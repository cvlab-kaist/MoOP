"""Expert MLP-channel importance for MoOP+E: per-denoising-step FM gradients, aggregated by znorm.

For each calibration clip (dense model, same CoC sample and noise seeds as compute_importance.py) the
signed per-step gate gradient of the flow-matching loss is taken at every one of the fm_steps
denoising steps (fresh noise per step). |g_s| is averaged over clips, then aggregated over steps as
the mean of within-layer z-scores (`znorm`).

Writes <out>/expert_importance.npz:
  exp_mlp_znorm (36, 8256), exp_q_znorm (36, 16)            used by the MoOP+E recipes
  exp_mlp_abs_step (10, 36, 8256), exp_q_abs_step (10, 36, 16)  mean over clips of |g_s|
plus expert_importance_sum.npz (for merging shards), config.json and metrics.json.

  python scripts/compute_expert_importance.py --manifest calib_100 --cache calib --out runs/expert_importance
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moop import MODEL_ID, MODEL_REV
from moop.data import clip_seed, load_cached, path_for, samples
from moop.importance import expert_step_importance_clip, znorm
from moop.model_io import load_base_model, set_attn_impl


def save(out_dir, acc, n, records):
    np.savez(out_dir / "expert_importance_sum.npz", n_clips=n, **acc)
    mean = {k: v / max(n, 1) for k, v in acc.items()}
    np.savez(out_dir / "expert_importance.npz", exp_mlp_znorm=znorm(mean["exp_mlp_abs_step"]),
             exp_q_znorm=znorm(mean["exp_q_abs_step"]), **mean)
    (out_dir / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="calib_100")
    ap.add_argument("--cache", default="calib")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--fm-steps", type=int, default=10)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = samples(args.manifest)[args.shard::args.n_shards]
    model, processor = load_base_model("cuda")
    set_attn_impl(model, "sdpa")
    (out_dir / "config.json").write_text(json.dumps({
        "model": MODEL_ID, "model_revision": MODEL_REV, "manifest": args.manifest,
        "clip_ids": [c for c, _ in clips], "seed": args.seed, "max_gen": args.max_gen,
        "fm_steps": args.fm_steps, "noise": "fresh draw per step", "aggregation": "znorm",
        "shard": args.shard, "n_shards": args.n_shards, "gpu": torch.cuda.get_device_name(0),
    }, indent=2))

    acc, records = {}, []
    for i, (clip_id, t0) in enumerate(clips):
        t = time.time()
        data = load_cached(path_for(args.cache, clip_id, t0))
        scores, rec = expert_step_importance_clip(model, processor, data, clip_seed(args.seed, clip_id),
                                                  args.max_gen, args.fm_steps)
        for k, v in scores.items():
            acc[k] = acc.get(k, 0.0) + v.astype(np.float64)
        rec["clip_id"] = clip_id
        records.append(rec)
        print(f"[{i + 1}/{len(clips)}] {clip_id} coc={rec['coc_len']} fm={rec['fm_loss']:.4f} "
              f"({time.time() - t:.0f}s)", flush=True)
        if (i + 1) % 10 == 0:
            save(out_dir, acc, i + 1, records)
    save(out_dir, acc, len(records), records)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
