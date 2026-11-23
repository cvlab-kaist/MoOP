"""Re-derive the OOD sample lists (ood_val, ood_train) from the PhysicalAI-AV dataset files.

A clip enters if `reasoning/ood_reasoning.parquet` has an event carrying both `event_start_timestamp`
and curated CoC text; the sample starts at that timestamp (the moment the OOD situation occurs), and
the clip's chunk comes from `clip_index.parquet`. ood_val = split 'val' (262 clips), ood_train =
split 'train' (1,271 clips). The shipped lists in data/manifests/ were produced by this rule.

  python scripts/make_ood_manifest.py --dataset-dir <PhysicalAI-AV root> --split val --out ood_val.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def first_event(events):
    if not isinstance(events, str):
        return None
    for e in json.loads(events):
        if "event_start_timestamp" in e and "coc" in e:
            return e
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--split", choices=["val", "train"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    ci = pd.read_parquet(root / "clip_index.parquet")
    ood = pd.read_parquet(root / "reasoning" / "ood_reasoning.parquet")
    rows = []
    for cid, r in ood.iterrows():
        e = first_event(r["events"])
        if cid not in ci.index or e is None or r["split"] != args.split:
            continue
        rows.append({"clip_id": cid, "t0_us": int(e["event_start_timestamp"]),
                     "chunk": int(ci.loc[cid, "chunk"]), "cluster": r["event_cluster"], "split": r["split"]})
    pd.DataFrame(rows).to_parquet(args.out, index=False)
    print(f"{len(rows)} clips -> {args.out}")


if __name__ == "__main__":
    main()
