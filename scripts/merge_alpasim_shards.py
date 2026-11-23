"""Stitch sharded AlpaSim runs back into one run directory.

Scoring fields are taken verbatim from each shard's own aggregate/results-summary.json; nothing is
recomputed. A shard without aggregate/ did not finish and is refused. Rollout directories are hardlinked.

  python scripts/merge_alpasim_shards.py --runs-root runs/alpasim \
      --shards moop_u40_origin150_sh0 moop_u40_origin150_sh1 --out moop_u40_origin150 --expect-scenes 150
"""

import argparse
import json
import shutil
from pathlib import Path


def link_tree(src, dst):
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            continue
        try:
            q.hardlink_to(p)
        except OSError:
            shutil.copy2(p, q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-scenes", type=int, default=None)
    args = ap.parse_args()
    out_dir = args.runs_root / args.out
    merged, seen, base = [], {}, None
    for name in args.shards:
        run = args.runs_root / name
        summ = run / "aggregate" / "results-summary.json"
        if not summ.exists():
            raise SystemExit(f"{name}: no aggregate/results-summary.json -- the shard did not finish")
        d = json.loads(summ.read_text())
        if base is None:
            base = {k: v for k, v in d.items() if k not in ("rollouts", "metrics_results")}
        elif d.get("score_criteria") != base.get("score_criteria"):
            raise SystemExit(f"{name}: score_criteria differs from {args.shards[0]}")
        for r in d["rollouts"]:
            key = (r["clipgt_id"], r["rollout_id"])
            if key in seen:
                raise SystemExit(f"duplicate rollout {key} in {name} and {seen[key]}")
            seen[key] = name
            merged.append(r)
        link_tree(run / "rollouts", out_dir / "rollouts")
        print(f"{name}: {len(d['rollouts'])} rollouts")
    scenes = {r["clipgt_id"] for r in merged}
    if args.expect_scenes is not None and len(scenes) != args.expect_scenes:
        raise SystemExit(f"expected {args.expect_scenes} scenes, got {len(scenes)}")
    (out_dir / "aggregate").mkdir(parents=True, exist_ok=True)
    base.update(rollouts=merged, metrics_results=[], merged_from=list(args.shards))
    (out_dir / "aggregate" / "results-summary.json").write_text(json.dumps(base, indent=2))
    print(f"merged {len(args.shards)} shards -> {out_dir}: {len(merged)} rollouts over {len(scenes)} scenes")


if __name__ == "__main__":
    main()
