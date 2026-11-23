"""Closed-loop summary of merged AlpaSim runs (paper Table 2 conventions).

Scene-score family, per arm (95% scene-cluster bootstrap, 200k draws, seed 0, one shared resample):
  score       AlpaSim scene score: 0 on an at-fault collision or off-road, else min(progress / 0.8, 1)
  at_fault    rollout had an at-fault (front / lateral) collision
  offroad     rollout went off-road
  progress    progress_clipped_rel
Aborted rollouts (AlpaSim's route sanity check) count as score 0, no event, progress 0.
Paired differences vs --ref use the same resample; score / progress as ref - arm, rates as arm - ref
(positive favours the reference).

AlpaSim score (Alpamayo-R1 Sec. 6.1): km driven per event, for 'at-fault' (front / lateral collision or
off-road) and 'all' (any collision or off-road). Within a scene, rollouts are ranked into indices 1..n
by run_uuid + rollout_id; per index, total distance / total events / 1000; reported as mean +- std over
indices. Aborted rollouts carry no distance and drop out, as in AlpaSim.

  python scripts/summarize_closedloop.py --arms dense=runs/alpasim/dense_origin150 \
      moop_u40=runs/alpasim/moop_u40_origin150 --ref moop_u40
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["score", "at_fault", "offroad", "progress"]
HIGHER = {"score", "progress"}


def load(run_dir):
    rs = json.loads((Path(run_dir) / "aggregate" / "results-summary.json").read_text())["rollouts"]
    rows = []
    for r in rs:
        m = r.get("metrics") or {}
        rows.append({
            "scene": r["clipgt_id"], "key": f"{r.get('run_uuid')}_{r['rollout_id']}",
            "aborted": r.get("failure_reason") not in (None, "collision_at_fault", "offroad"),
            "score": r["score"] or 0.0, "at_fault": float((m.get("collision_at_fault") or 0) > 0),
            "offroad": float((m.get("offroad") or 0) > 0), "progress": m.get("progress_clipped_rel") or 0.0,
            "dist": m.get("dist_traveled_m"),
            "e_all": float((m.get("offroad_or_collision") or 0) > 0),
            "e_af": float((m.get("offroad_or_collision_at_fault") or 0) > 0)})
    return pd.DataFrame(rows)


def km_per_event(df, idx=None, scenes=None):
    """AlpaSim score (all, at-fault): mean over rollout indices of km / events. idx (B, S) resamples scenes."""
    d = df[df["dist"].notna()].copy()
    d["i"] = d.groupby("scene")["key"].rank(method="dense").astype(int) - 1
    n_i = int(d["i"].max()) + 1
    pos = {s: j for j, s in enumerate(scenes)}
    A = np.zeros((len(scenes), n_i, 3))  # (S, idx, [dist m, all events, at-fault events])
    for r in d.itertuples():
        A[pos[r.scene], r.i] += (r.dist, r.e_all, r.e_af)
    a = A if idx is None else A[idx]  # (..., S, idx, 3)
    s = a.sum(-3)
    with np.errstate(divide="ignore"):
        per = s[..., 0:1] / s[..., 1:3] / 1000  # (..., idx, 2)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=merged_run_dir")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--draws", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms)
    data = {a: load(d) for a, d in arms.items()}
    scenes = sorted(set().union(*[set(d.scene) for d in data.values()]))
    for a, d in data.items():
        assert set(d.scene) == set(scenes), f"{a}: scene set differs"
    V = {(a, m): d.groupby("scene")[m].mean().reindex(scenes).to_numpy() for a, d in data.items() for m in METRICS}
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, len(scenes), (args.draws, len(scenes)))  # (B, S)

    def ci(x):
        b = x[idx].mean(1)
        return float(x.mean()), *np.percentile(b, [2.5, 97.5])

    f = lambda c, sig=False: (f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]" + ("*" if c[1] > 0 or c[2] < 0 else "")
                              if sig else f"{c[0]:.3f} [{c[1]:.3f}, {c[2]:.3f}]")
    print(f"{len(scenes)} scenes; scene-cluster bootstrap {args.draws:,} draws, seed {args.seed}")
    print("| arm | rollouts | aborted | " + " | ".join(METRICS) + " | AlpaSim score at-fault | all |")
    print("|---" * (len(METRICS) + 5) + "|")
    for a, d in data.items():
        per = km_per_event(d, scenes=scenes)  # (idx, 2)
        print(f"| {a} | {len(d)} | {int(d.aborted.sum())} | " + " | ".join(f(ci(V[a, m])) for m in METRICS)
              + f" | {per[:, 1].mean():.2f} ± {per[:, 1].std(ddof=1):.2f} | {per[:, 0].mean():.2f} ± "
                f"{per[:, 0].std(ddof=1):.2f} |")
    if args.ref:
        print(f"\npaired vs {args.ref} (score/progress = ref - arm, rates = arm - ref; * CI excludes 0)")
        bref = km_per_event(data[args.ref], idx, scenes).mean(-2)  # (B, 2)
        pref = km_per_event(data[args.ref], scenes=scenes).mean(0)
        for a in arms:
            if a == args.ref:
                continue
            cells = []
            for m in METRICS:
                d = V[args.ref, m] - V[a, m] if m in HIGHER else V[a, m] - V[args.ref, m]
                cells.append(f(ci(d), sig=True))
            ba = km_per_event(data[a], idx, scenes).mean(-2)
            pa = km_per_event(data[a], scenes=scenes).mean(0)
            for j in (1, 0):  # at-fault, all
                lo, hi = np.percentile(bref[:, j] - ba[:, j], [2.5, 97.5])
                cells.append(f((pref[j] - pa[j], lo, hi), sig=True))
            print(f"| {a} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
