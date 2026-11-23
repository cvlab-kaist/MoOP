"""Physical removal of pruned units from Alpamayo 1.5, and the recipe file format.

Keep-masks become real weight surgery: q_proj rows / o_proj columns per kept Q head, gate/up rows /
down columns per kept MLP channel. KV heads, k/v projections and the KV cache are never touched
(8 x 128 everywhere): the expert reads the VLM's per-layer cache, so its layout is a fixed interface.

Pruning Q heads unevenly across KV groups breaks the uniform group size that repeat_kv / enable_gqa
need, so the slim attention gathers K/V per kept Q head after the cache update (index_select on the
head dimension) and runs with num_key_value_groups = 1.

A recipe is `slim_meta.json` (kept indices per layer). MoOP does not rewrite weights, so the pruned
model is fully determined by the recipe and the pinned base weights: `load_slim` rebuilds it bit for
bit. `save_slim(..., write_state=True)` additionally writes the state_dict (about 17 GB), which only
consumers that cannot reach the base weights need (e.g. the AlpaSim driver container).
"""

import json
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from moop import MODEL_ID, MODEL_REV


def slim_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    """Stock Qwen3VLTextAttention.forward + K/V gather for ragged GQA."""
    input_shape = hidden_states.shape[:-1]  # (B, T)
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx,
                                                          cache_kwargs)

    # (B, 8, S, D) -> (B, n_keep, S, D): one K/V copy per kept Q head; groups become 1.
    # After the cache update, so the cache stays compact (8 KV heads).
    key_states = key_states.index_select(1, self.kv_head_index)
    value_states = value_states.index_select(1, self.kv_head_index)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _swap(linear, weight):
    """In-place Parameter swap (a fresh nn.Linear would silently be fp32)."""
    linear.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    linear.out_features, linear.in_features = weight.shape


def slice_layer(layer, kept_q, kept_mlp, head_dim):
    """Surgery on one decoder layer. kept_q / kept_mlp: ascending original indices."""
    assert len(kept_q) > 0
    attn = layer.self_attn
    assert attn.q_proj.bias is None and attn.o_proj.bias is None
    qi = torch.as_tensor(kept_q, dtype=torch.long, device=attn.q_proj.weight.device)
    rows = (qi[:, None] * head_dim + torch.arange(head_dim, device=qi.device)).reshape(-1)  # (n_keep*D,)
    _swap(attn.q_proj, attn.q_proj.weight.data[rows])
    _swap(attn.o_proj, attn.o_proj.weight.data[:, rows])
    mlp = layer.mlp
    mi = torch.as_tensor(kept_mlp, dtype=torch.long, device=mlp.gate_proj.weight.device)
    _swap(mlp.gate_proj, mlp.gate_proj.weight.data[mi])
    _swap(mlp.up_proj, mlp.up_proj.weight.data[mi])
    _swap(mlp.down_proj, mlp.down_proj.weight.data[:, mi])
    mlp.intermediate_size = mi.numel()


def bind_slim_attention(attn, kept_q, n_heads_orig, n_kv_heads):
    """kv_head_index[i] = KV head of the i-th kept Q head; groups collapse to 1."""
    per_group = n_heads_orig // n_kv_heads
    idx = torch.as_tensor(kept_q, dtype=torch.long, device=attn.k_proj.weight.device) // per_group
    attn.register_buffer("kv_head_index", idx, persistent=False)
    attn.num_key_value_groups = 1
    attn.forward = types.MethodType(slim_attn_forward, attn)


def apply_surgery(model, vlm_q, vlm_mlp, exp_q, exp_mlp):
    """Apply 0/1 keep-masks ((L, H) / (L, I) per tower) as physical removal. Returns the recipe meta."""
    meta = {"vlm": [], "expert": [], "kvonly_layers": []}
    for tower, layers, cfg, mq, mm in (
            ("vlm", model.vlm.model.language_model.layers, model.vlm.config.text_config, vlm_q, vlm_mlp),
            ("expert", model.expert.layers, model.expert.config, exp_q, exp_mlp)):
        for li, layer in enumerate(layers):
            kq = np.nonzero(mq[li])[0].tolist()
            km = np.nonzero(mm[li])[0].tolist()
            meta[tower].append({"q": kq, "mlp": km})
            slice_layer(layer, kq, km, cfg.head_dim)
            bind_slim_attention(layer.self_attn, kq, cfg.num_attention_heads, cfg.num_key_value_heads)
    return meta


def masks_from_meta(meta, model):
    """slim_meta.json -> the four keep-masks, shaped for `model` (the dense skeleton)."""
    tc, ec = model.vlm.config.text_config, model.expert.config
    vq = np.zeros((tc.num_hidden_layers, tc.num_attention_heads))
    vm = np.zeros((tc.num_hidden_layers, tc.intermediate_size))
    eq = np.zeros((ec.num_hidden_layers, ec.num_attention_heads))
    em = np.zeros((ec.num_hidden_layers, ec.intermediate_size))
    for (q, m), entries in (((vq, vm), meta["vlm"]), ((eq, em), meta["expert"])):
        for li, e in enumerate(entries):
            q[li, e["q"]] = 1.0
            m[li, e["mlp"]] = 1.0
    return vq, vm, eq, em


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def check_slim(model):
    """Every parameter must have stayed bf16 and contiguous through surgery."""
    for name, p in model.named_parameters():
        assert p.dtype == torch.bfloat16, f"{name}: {p.dtype}"
        assert p.is_contiguous(), name


def save_slim(model, meta, out_dir, write_state=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "slim_meta.json").write_text(json.dumps(meta))
    if write_state:
        torch.save(model.state_dict(), out_dir / "slim_state.pt")


def load_slim(ckpt_dir, device="cuda", revision=MODEL_REV):
    """Dense skeleton at the pinned revision -> surgery from slim_meta.json (-> optional state load)."""
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "slim_meta.json").read_text())
    if meta.get("kvonly_layers") or meta.get("vlm_hidden"):
        raise ValueError("recipe uses layer removal or width slicing, which this package does not support")
    model = Alpamayo1_5.from_pretrained(MODEL_ID, revision=revision, dtype=torch.bfloat16)
    apply_surgery(model, *masks_from_meta(meta, model))
    state = ckpt_dir / "slim_state.pt"
    if state.exists():
        model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
