# SPDX-License-Identifier: Apache-2.0
"""MoOP slim (structurally pruned) checkpoints for the AlpaSim Alpamayo 1.5 driver.

Installed by alpasim/install.py into src/driver/src/alpasim_driver/models/. A checkpoint directory is
slim if it holds `slim_meta.json` (kept Q heads / MLP channels per layer, written by
scripts/build_slim.py). The model is rebuilt from the pinned base weights and that recipe; a
`slim_state.pt` beside it is loaded strictly if present (not needed for MoOP, which does not rewrite
weights). Self-contained copy of moop/slim.py's load path: the driver image does not install moop.
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

MODEL_ID = "nvidia/Alpamayo-1.5-10B"
MODEL_REV = "7aba8293c09993f2e125c6819df05d7fa3e873ea"


def slim_attn_forward(self, hidden_states, position_embeddings, attention_mask,
                      past_key_values=None, cache_position=None, **kwargs):
    """Stock Qwen3VLTextAttention.forward + K/V gather for ragged GQA."""
    input_shape = hidden_states.shape[:-1]
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
    key_states = key_states.index_select(1, self.kv_head_index)
    value_states = value_states.index_select(1, self.kv_head_index)
    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout, scaling=self.scaling, **kwargs)
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _swap(linear, weight):
    linear.weight = nn.Parameter(weight.contiguous(), requires_grad=False)
    linear.out_features, linear.in_features = weight.shape


def _slice_layer(layer, kept_q, kept_mlp, cfg):
    attn = layer.self_attn
    qi = torch.as_tensor(kept_q, dtype=torch.long, device=attn.q_proj.weight.device)
    rows = (qi[:, None] * cfg.head_dim + torch.arange(cfg.head_dim, device=qi.device)).reshape(-1)
    _swap(attn.q_proj, attn.q_proj.weight.data[rows])
    _swap(attn.o_proj, attn.o_proj.weight.data[:, rows])
    mlp = layer.mlp
    mi = torch.as_tensor(kept_mlp, dtype=torch.long, device=mlp.gate_proj.weight.device)
    _swap(mlp.gate_proj, mlp.gate_proj.weight.data[mi])
    _swap(mlp.up_proj, mlp.up_proj.weight.data[mi])
    _swap(mlp.down_proj, mlp.down_proj.weight.data[:, mi])
    mlp.intermediate_size = mi.numel()
    per_group = cfg.num_attention_heads // cfg.num_key_value_heads
    attn.register_buffer("kv_head_index", qi // per_group, persistent=False)
    attn.num_key_value_groups = 1
    attn.forward = types.MethodType(slim_attn_forward, attn)


def is_slim_checkpoint(checkpoint_path) -> bool:
    return (Path(checkpoint_path) / "slim_meta.json").is_file()


def load_slim(ckpt_dir, device="cuda", attn_implementation="sdpa", revision=MODEL_REV):
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    ckpt_dir = Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "slim_meta.json").read_text())
    model = Alpamayo1_5.from_pretrained(MODEL_ID, revision=revision, dtype=torch.bfloat16,
                                        attn_implementation=attn_implementation)
    for layers, cfg, entries in ((model.vlm.model.language_model.layers, model.vlm.config.text_config, meta["vlm"]),
                                 (model.expert.layers, model.expert.config, meta["expert"])):
        for layer, e in zip(layers, entries):
            _slice_layer(layer, np.asarray(e["q"], dtype=np.int64), np.asarray(e["mlp"], dtype=np.int64), cfg)
    state = ckpt_dir / "slim_state.pt"
    if state.exists():
        model.load_state_dict(torch.load(state, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)
