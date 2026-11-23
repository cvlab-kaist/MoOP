"""Split a scene list into k AlpaSim test suites (one per GPU) and print the suite names.

AlpaSim selects scenes by test suite; launching by uuid (never a bare scene_id) pins the render
release. Scenes are assigned round-robin in scene_id order so shards take similar wall-clock time.

  python scripts/make_shard_suites.py --scenes data/scenes/origin150.csv --k 4 \
      --name moop_u40_origin150 --sim-suites <alpasim>/data/scenes/sim_suites.csv --out suites.csv
"""

import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--sim-suites", required=True, help="AlpaSim's data/scenes/sim_suites.csv")
    ap.add_argument("--parent-suite", default="public_2601")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = pd.read_csv(args.scenes).sort_values("scene_id").reset_index(drop=True)
    su = pd.read_csv(args.sim_suites)
    own = set(su.loc[su.test_suite_id == args.parent_suite, "uuid"])
    bad = sorted(set(rows.uuid) - own)
    if bad:
        raise SystemExit(f"{len(bad)} uuids are not in {args.parent_suite} (e.g. {bad[:2]})")
    parts = []
    for i in range(args.k):
        part = rows.iloc[i::args.k].copy()
        part["test_suite_id"] = f"{args.name}_sh{i}"
        parts.append(part)
        print(f"{args.name}_sh{i} {len(part)}")
    pd.concat(parts)[["test_suite_id", "scene_id", "uuid"]].to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
