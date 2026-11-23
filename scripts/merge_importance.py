"""Merge sharded importance runs (compute_importance.py or compute_expert_importance.py).

  python scripts/merge_importance.py --shards runs/imp_s0 runs/imp_s1 ... --out runs/importance
"""

import argparse
import json
from pathlib import Path

import numpy as np

from moop.importance import znorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    shards = [Path(s) for s in args.shards]
    expert = (shards[0] / "expert_importance_sum.npz").exists()
    stem = "expert_importance" if expert else "importance"
    acc, n, records, clip_ids = {}, 0, [], []
    for s in shards:
        z = np.load(s / f"{stem}_sum.npz")
        n += int(z["n_clips"])
        for k in z.files:
            if k != "n_clips":
                acc[k] = acc.get(k, 0.0) + z[k]
        records += json.loads((s / "metrics.json").read_text())["per_clip"]
        clip_ids += json.loads((s / "config.json").read_text())["clip_ids"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mean = {k: v / n for k, v in acc.items()}
    if expert:
        mean = {"exp_mlp_znorm": znorm(mean["exp_mlp_abs_step"]),
                "exp_q_znorm": znorm(mean["exp_q_abs_step"]), **mean}
    np.savez(out / f"{stem}.npz", **mean)
    cfg = json.loads((shards[0] / "config.json").read_text())
    cfg.update(clip_ids=clip_ids, merged_from=[str(s) for s in shards], shard=None, n_shards=None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "metrics.json").write_text(json.dumps({"n_clips": n, "per_clip": records}, indent=2))
    print(f"merged {len(shards)} shards, {n} clips -> {out}")


if __name__ == "__main__":
    main()
