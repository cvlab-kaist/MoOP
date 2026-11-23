"""VLM Taylor importance (CE and FM objectives) over a calibration set.

Writes <out>/importance.npz with the mean over clips of |dL/dg| per unit:
  coc_vlm_q (36, 32), coc_vlm_mlp (36, 12288)   CE objective
  traj_vlm_q, traj_vlm_mlp                       FM objective (through the KV cache)
  traj_exp_q (36, 16), traj_exp_mlp (36, 8256)   FM objective on the expert, summed over steps
plus importance_sum.npz (sums + clip count, for merging shards), config.json and metrics.json.

Paper setting: calib_100, seed 42, max_gen 256, fm_steps 10 (one 48 GB GPU, ~40 GB peak).

  python scripts/compute_importance.py --manifest calib_100 --cache calib --out runs/importance
  # sharded: --shard i --n-shards k on k GPUs, then scripts/merge_importance.py
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moop import MODEL_ID, MODEL_REV
from moop.data import clip_seed, load_cached, path_for, samples
from moop.importance import vlm_importance_clip
from moop.model_io import load_base_model, set_attn_impl


def save(out_dir, acc, n, records, args):
    np.savez(out_dir / "importance_sum.npz", n_clips=n, **acc)
    np.savez(out_dir / "importance.npz", **{k: v / max(n, 1) for k, v in acc.items()})
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
    model.vlm.enable_input_require_grads()

    (out_dir / "config.json").write_text(json.dumps({
        "model": MODEL_ID, "model_revision": MODEL_REV, "manifest": args.manifest,
        "clip_ids": [c for c, _ in clips], "seed": args.seed, "seed_rule": "sha256(f'{seed}:{clip_id}')[:4]",
        "max_gen": args.max_gen, "fm_steps": args.fm_steps, "shard": args.shard, "n_shards": args.n_shards,
        "gpu": torch.cuda.get_device_name(0),
    }, indent=2))

    acc, records = {}, []
    for i, (clip_id, t0) in enumerate(clips):
        t = time.time()
        data = load_cached(path_for(args.cache, clip_id, t0))
        torch.cuda.reset_peak_memory_stats()
        scores, rec = vlm_importance_clip(model, processor, data, clip_seed(args.seed, clip_id),
                                          args.max_gen, args.fm_steps)
        for k, v in scores.items():
            acc[k] = acc.get(k, 0.0) + v.astype(np.float64)
        rec.update(clip_id=clip_id, peak_gb=torch.cuda.max_memory_allocated() / 2**30)
        records.append(rec)
        print(f"[{i + 1}/{len(clips)}] {clip_id} coc={rec['coc_len']} fm={rec['fm_loss']:.4f} "
              f"peak={rec['peak_gb']:.1f}GB ({time.time() - t:.0f}s)", flush=True)
        if (i + 1) % 10 == 0:
            save(out_dir, acc, i + 1, records, args)
    save(out_dir, acc, len(records), records, args)
    print("saved ->", out_dir, flush=True)


if __name__ == "__main__":
    main()
