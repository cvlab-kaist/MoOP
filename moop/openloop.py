"""Open-loop evaluation of one sample (paper Table 1 protocol).

The model writes its own CoC (temperature 0.6, top-p 0.98, seed = clip seed); prompt + CoC are then
re-prefilled teacher-forced and the expert denoises K trajectories from that cache with seeds
base + k. Per-sample ADE / FDE are stored at 1.6 / 3.2 / 6.4 s so minADE@K' for any K' <= K is a prefix.
"""

import numpy as np
import torch

from moop.data import clip_seed
from moop.metrics import ade_fde, coc_degenerate
from moop.model_io import build_inputs, denoise_with_cache, run_rollout

COT_END, TRAJ_FUTURE_START = 155678, 155681


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _paths(model, inputs, cache, rope_deltas, seeds):
    prefill = cache.get_seq_length()
    dev = inputs["input_ids"].device
    offset = torch.tensor([prefill], device=dev)
    prefix_mask = torch.ones(1, prefill, device=dev, dtype=torch.long)
    preds = []
    for s in seeds:
        action = denoise_with_cache(model, cache, rope_deltas, offset, prefix_mask, seed=s)  # crops back
        pred_xyz, _ = model.action_space.action_to_traj(
            action.float(), inputs["ego_history_xyz"][:, -1].float(), inputs["ego_history_rot"][:, -1].float())
        preds.append(pred_xyz[0, :, :2].cpu().numpy())
    return np.stack(preds)  # (K, 64, 2)


@torch.no_grad()
def eval_config_samples(model, inputs, seq_tf, coc_start, coc_end, gt_xy, seeds):
    """Teacher-forced re-prefill of prompt + CoC, then K denoisings. Returns (ade_k, fde_k, CoC NLL, pred_k)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.vlm.model(
            input_ids=seq_tf, attention_mask=torch.ones_like(seq_tf),
            pixel_values=inputs["tokenized_data"]["pixel_values"],
            image_grid_thw=inputs["tokenized_data"]["image_grid_thw"], use_cache=True,
        )
        logits = model.vlm.lm_head(out.last_hidden_state[:, coc_start - 1: coc_end - 1]).float()
        nll = torch.nn.functional.cross_entropy(logits[0], seq_tf[0, coc_start:coc_end]).item()
        pred_k = _paths(model, inputs, out.past_key_values, out.rope_deltas, seeds)
    ade, fde = ade_fde(pred_k, gt_xy)
    return ade, fde, nll, pred_k


@torch.no_grad()
def eval_from_generation_cache(model, inputs, roll, gt_xy, seeds):
    """K denoisings on the rollout's own KV cache (the official inference path)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_k = _paths(model, inputs, roll["past_key_values"], roll["rope_deltas"], seeds)
    ade, fde = ade_fde(pred_k, gt_xy)
    return ade, fde, pred_k


def evaluate_clip(model, processor, data, clip_id, seed=42, k=8, max_gen=256):
    """One open-loop row: per-sample ADE/FDE at three horizons, the generated CoC, its NLL."""
    inputs = build_inputs(model, processor, data, "cuda")
    prompt_len = inputs["input_ids"].shape[1]
    gt_xy = data["ego_future_xyz"][0, 0, :, :2].cpu().numpy()  # (64, 2)
    base = clip_seed(seed, clip_id)
    seeds = [base + i for i in range(k)]

    seed_all(base)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        roll = run_rollout(model, inputs, max_generation_length=max_gen)
    coc_end = roll["eos_pos"] + 1
    seq_gen = roll["sequences"][:, :coc_end].clone()
    gen_ids = roll["sequences"][0, prompt_len:coc_end].tolist()
    gen_coc = model.tokenizer.decode([t for t in gen_ids if t not in (COT_END, TRAJ_FUTURE_START)],
                                     skip_special_tokens=True)
    rec = {"clip_id": clip_id, "seed": base}
    try:
        ade_k, fde_k, nll_self, pred_k = eval_config_samples(model, inputs, seq_gen, prompt_len, coc_end,
                                                             gt_xy, seeds)
    except ValueError as e:
        # A degenerate CoC can contain a vision placeholder token; the teacher-forced re-prefill then has
        # one more image token than image features. For that clip only, drive on the generation cache.
        if "Image features and image tokens do not match" not in str(e):
            raise
        ade_k, fde_k, pred_k = eval_from_generation_cache(model, inputs, roll, gt_xy, seeds)
        nll_self = None
        rec["tf_fallback"] = "generation_cache"
    del roll
    rec.update({"ade_rollout_k": [round(float(x), 6) for x in ade_k],
                "fde_rollout_k": [round(float(x), 6) for x in fde_k],
                "nll_self": nll_self, "gen_coc": gen_coc, "gen_len": len(gen_ids),
                **{f"coc_{key}": v for key, v in coc_degenerate(gen_coc).items()}})
    for h in (16, 32):
        a_h, f_h = ade_fde(pred_k[:, :h], gt_xy[:h])
        rec.update({f"ade_rollout_k_h{h}": [round(float(x), 6) for x in a_h],
                    f"fde_rollout_k_h{h}": [round(float(x), 6) for x in f_h]})
    return rec
