"""Score LingoQA predictions with Lingo-Judge and summarise with segment-cluster bootstrap CIs.

Lingo-Judge (HF `wayveai/Lingo-Judge`) is imported from the LingoQA repository's benchmark/judge.py
(Wayve non-commercial licence; clone https://github.com/wayveai/LingoQA and set $LINGOQA_BENCH to its
benchmark/ directory). An answer is correct if the judge's logit is > 0 against either reference.
CIs resample whole segments (5 questions each), 200k draws, seed 0.

  python scripts/score_lingoqa.py --arms dense=runs/lingoqa/dense moop_u40=runs/lingoqa/moop_u40 --ref moop_u40
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

from moop.metrics import boot_many


def score_run(judge, run_dir):
    run_dir = Path(run_dir)
    scored = run_dir / "scored.json"
    if scored.exists():
        return json.loads(scored.read_text())
    rows = json.loads((run_dir / "rows.json").read_text())
    out = []
    for r in rows:
        s = float(judge.compute([r["question"]], [list(r["references"])], [r["answer"]])[0])
        out.append({"question_id": r["question_id"], "segment_id": r["segment_id"], "score": s,
                    "correct": s > 0.0})
    scored.write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, help="name=run_dir")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    sys.path.insert(0, os.environ["LINGOQA_BENCH"])
    from judge import LingoJudge

    judge = LingoJudge().eval().to(args.device)
    arms = dict(a.split("=", 1) for a in args.arms)
    scored = {a: pd.DataFrame(score_run(judge, d)).set_index(["question_id", "segment_id"]) for a, d in arms.items()}
    keys = sorted(set.intersection(*[set(s.index) for s in scored.values()]))
    assert len(keys) == 500, f"{len(keys)} questions common to all arms, expected 500"
    acc = {a: scored[a].loc[keys, "correct"].astype(float).to_numpy() for a in arms}
    series = dict(acc)
    if args.ref:
        series.update({f"{a}-{args.ref}": acc[a] - acc[args.ref] for a in arms if a != args.ref})
    ci = boot_many(series, groups=[k[1] for k in keys])
    for k, v in series.items():
        print(f"{k:28s} {100 * v.mean():6.1f}%  [{100 * ci[k][0]:.1f}, {100 * ci[k][1]:.1f}]")


if __name__ == "__main__":
    main()
