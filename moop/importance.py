"""Per-clip Taylor importance for MoOP.

VLM (Q heads, MLP channels), both objectives from one teacher-forced forward:
  1. sample the model's own CoC (temperature 0.6, top-p 0.98) under the clip seed
  2. teacher-forced VLM forward over prompt + CoC, keeping the graph and the KV cache
  3. CE: backward of the CoC NLL                                   -> |dL_CE/dg|
  4. FM: flow-matching loss through the expert onto detached cache leaves, then one VLM backward
     seeded with the accumulated cache gradients                   -> |dL_FM/dg|

Expert (MLP channels, used by MoOP+E): per-denoising-step FM gate gradients |dL_s/dg| on the
dense model's cache, aggregated over steps by `znorm` (mean over steps of a within-layer z-score).
"""

import numpy as np
import torch

from moop.gates import (
    UnitGates,
    cache_tensors,
    coc_nll,
    expert_fm_grads,
    expert_fm_grads_stepwise,
    vlm_forward_with_grad,
)
from moop.model_io import build_inputs, gt_actions, run_rollout


def _gates(tower_layers, cfg):
    return UnitGates(tower_layers, cfg.num_attention_heads, cfg.head_dim, cfg.intermediate_size, "cuda")


def vlm_importance_clip(model, processor, data, seed, max_gen=256, fm_steps=10):
    """Returns ({coc_vlm_q, coc_vlm_mlp, traj_vlm_q, traj_vlm_mlp, traj_exp_q, traj_exp_mlp}, record).

    The model must have `model.vlm.enable_input_require_grads()` set: all weights are frozen and the
    gates sit downstream of k/v, so otherwise the early layers' cache carries no graph.
    """
    inputs = build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = run_rollout(model, inputs, max_generation_length=max_gen)
    coc_start, coc_end = prompt_len, roll["eos_pos"] + 1
    seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
    del roll

    x1 = gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    vlm_gates = _gates(model.vlm.model.language_model.layers, model.vlm.config.text_config)
    exp_gates = _gates(model.expert.layers, model.expert.config)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden, cache, rope_deltas = vlm_forward_with_grad(model, seq_tf, inputs["tokenized_data"])
    cache_t = cache_tensors(cache, len(model.vlm.model.language_model.layers))
    prefill = cache.get_seq_length()
    out = {}

    # CE objective
    nll = coc_nll(model, hidden, seq_tf, coc_start, coc_end)
    nll.backward(retain_graph=True)
    out["coc_vlm_q"], out["coc_vlm_mlp"] = vlm_gates.q_scores(), vlm_gates.mlp_scores()
    vlm_gates.zero_grads()
    for k, v in cache_t:
        k.grad = None
        v.grad = None
    del hidden, nll

    # FM objective: expert backward onto cache leaves, then one VLM backward from the cache grads
    fm_loss, grads, leaves = expert_fm_grads(model, cache, rope_deltas, x1, fm_steps, seed, prefill)
    out["traj_exp_q"], out["traj_exp_mlp"] = exp_gates.q_scores(), exp_gates.mlp_scores()
    ts, gs = [], []
    for (k, v), (gk, gv) in zip(cache_t, grads):
        for t_, g_ in ((k, gk), (v, gv)):
            if g_ is not None:
                ts.append(t_)
                gs.append(g_.to(t_.dtype))
    torch.autograd.backward(ts, gs)
    out["traj_vlm_q"], out["traj_vlm_mlp"] = vlm_gates.q_scores(), vlm_gates.mlp_scores()

    vlm_gates.remove()
    exp_gates.remove()
    del cache, cache_t, leaves, grads, ts, gs
    return out, {"coc_len": coc_end - coc_start, "prompt_len": prompt_len, "fm_loss": fm_loss}


def expert_step_importance_clip(model, processor, data, seed, max_gen=256, fm_steps=10):
    """Returns ({exp_q_abs_step (S, L, H), exp_mlp_abs_step (S, L, I)}, record) for one clip."""
    inputs = build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = run_rollout(model, inputs, max_generation_length=max_gen)
        coc_end = roll["eos_pos"] + 1
        seq_tf = roll["sequences"][:, :coc_end]  # (1, T) prompt + generated CoC
        del roll
        vout = model.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
    cache, rope_deltas = vout.past_key_values, vout.rope_deltas
    gates = _gates(model.expert.layers, model.expert.config)
    x1 = gt_actions(model, data, "cuda").to(torch.float32)  # (1, 64, 2)
    losses, q, mlp = expert_fm_grads_stepwise(model, cache, rope_deltas, x1, fm_steps, seed,
                                              cache.get_seq_length(), gates)
    gates.remove()
    del cache
    return ({"exp_q_abs_step": np.abs(q), "exp_mlp_abs_step": np.abs(mlp)},
            {"coc_len": coc_end - prompt_len, "fm_loss": float(np.mean(losses))})


def zscore_layers(a):
    """Within-layer z-score of an (L, U) map."""
    m = a.mean(1, keepdims=True)
    s = a.std(1, keepdims=True)
    return (a - m) / np.where(s > 0, s, 1.0)


def znorm(abs_step):
    """(S, L, U) per-step |grad| (mean over clips) -> (L, U): mean over steps of within-layer z-scores."""
    return np.mean([zscore_layers(a) for a in abs_step.astype(np.float64)], axis=0)
