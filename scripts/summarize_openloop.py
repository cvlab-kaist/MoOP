"""Summarise open-loop runs: minADE@6 / minFDE@6 at 1.6 / 3.2 / 6.4 s with 95% bootstrap CIs, and paired
differences against a reference arm (clip bootstrap, 200k draws, seed 0, one shared resample).

  python scripts/summarize_openloop.py --manifest test_500 \
      --arms dense=runs/openloop/dense moop_u40=runs/openloop/moop_u40 --ref moop_u40
"""

import argparse
import json
from pathlib import Path

import numpy as np

from moop.data import samples
from moop.metrics import HORIZONS, boot_many, min_at_k


def load_rows(run_dir, name):
    rows = {}
    for p in sorted(Path(run_dir).glob(f"{name}_s*of*.json")):
        for r in json.loads(p.read_text()):
            rows[r["clip_id"]] = r
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--ref", default=None, help="arm the others are paired against")
    ap.add_argument("--out", default=None, help="write the summary as JSON here")
    args = ap.parse_args()
    name = Path(args.manifest).stem
    ids = [c for c, _ in samples(args.manifest)]
    arms = dict(a.split("=", 1) for a in args.arms)
    rows = {a: load_rows(d, name) for a, d in arms.items()}
    for a, r in rows.items():
        missing = [c for c in ids if c not in r]
        assert not missing, f"{a}: {len(missing)} of {len(ids)} clips missing"

    res = {}
    print(f"{name}: {len(ids)} clips, minADE@6 / minFDE@6, mean [95% CI]")
    for h in HORIZONS:
        for kind in ("ade", "fde"):
            v = {a: np.array([min_at_k(rows[a][c], kind, h) for c in ids]) for a in arms}
            series = dict(v)
            if args.ref:
                series.update({f"{a}-{args.ref}": v[a] - v[args.ref] for a in arms if a != args.ref})
            ci = boot_many(series)
            res[f"{kind}@{h}"] = {k: {"mean": float(np.mean(s)), "ci": ci[k][:2], "p": ci[k][2]}
                                  for k, s in series.items()}
            line = "  ".join(f"{k} {np.mean(s):.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k, s in series.items())
            print(f"min{kind.upper()}@6 {h}: {line}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
