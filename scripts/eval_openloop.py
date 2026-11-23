"""Open-loop evaluation (paper Table 1): minADE@6 / minFDE@6 on Test500, OOD-Val, OOD-Train.

  python scripts/eval_openloop.py --model dense        --manifest test_500 --cache test --out runs/openloop/dense
  python scripts/eval_openloop.py --model runs/moop_u40 --manifest ood_val --cache ood  --out runs/openloop/moop_u40
  # shards: --shard i --n-shards k (one GPU each); rows go to <out>/<manifest>_s<i>of<k>.json, resumable
Then summarise with scripts/summarize_openloop.py. k = 8 samples are stored; the paper reports the first 6.
Deterministic kernels are on; results are bit-identical on the same GPU model, and differ by about
+-0.02 m across GPU generations, so compare arms on one GPU type.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from moop import MODEL_REV
from moop.data import load_cached, path_for, samples
from moop.model_io import load_base_model, set_attn_impl
from moop.openloop import evaluate_clip


def set_determinism():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'dense' or a directory holding slim_meta.json")
    ap.add_argument("--manifest", required=True, help="test_500 | ood_val | ood_train | path")
    ap.add_argument("--cache", required=True, help="cache set the manifest was built into (test | ood)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-gen", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.manifest).stem
    rows_path = out / f"{name}_s{args.shard}of{args.n_shards}.json"
    clips = samples(args.manifest)[args.shard::args.n_shards][: args.limit]
    set_determinism()
    if args.model == "dense":
        model, processor = load_base_model("cuda")
    else:
        from alpamayo1_5 import helper

        from moop.slim import load_slim
        model = load_slim(args.model, device="cuda")
        processor = helper.get_processor(model.tokenizer)
    set_attn_impl(model, "sdpa")
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "model_revision": MODEL_REV, "manifest": args.manifest, "k": args.k,
        "seed": args.seed, "max_gen": args.max_gen, "gpu": torch.cuda.get_device_name(0),
        "deterministic": True, "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"]}, indent=2))

    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["clip_id"] for r in rows}
    for i, (clip_id, t0) in enumerate(clips):
        if clip_id in done:
            continue
        t = time.time()
        rec = evaluate_clip(model, processor, load_cached(path_for(args.cache, clip_id, t0)), clip_id,
                            args.seed, args.k, args.max_gen)
        rows.append(rec)
        if len(rows) % 10 == 0:
            rows_path.write_text(json.dumps(rows))
        print(f"[{i + 1}/{len(clips)}] {clip_id[:8]} minADE@6 {min(rec['ade_rollout_k'][:6]):.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
    rows_path.write_text(json.dumps(rows))
    a = np.array([min(r["ade_rollout_k"][:6]) for r in rows])
    print(f"{name}: {len(rows)} clips, minADE@6 (6.4 s) mean {a.mean():.4f}")


if __name__ == "__main__":
    main()
