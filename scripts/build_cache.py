"""Build the per-clip sample cache for a manifest from the PhysicalAI-AV dataset.

Clips are streamed one at a time from the Hugging Face hub (`nvidia/PhysicalAI-Autonomous-Vehicles`,
gated; log in with `huggingface-cli login` first); only the needed members of each chunk move. About
6 MB per sample. Resumable: samples already on disk are skipped.

  python scripts/build_cache.py --manifest calib_100 --cache calib
  python scripts/build_cache.py --manifest test_500  --cache test
  python scripts/build_cache.py --manifest ood_val   --cache ood
  python scripts/build_cache.py --manifest ood_train --cache ood
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from moop.data import cache_root, load_manifest, path_for, write_sample

_AVDI = None


def _avdi(local_dir):
    global _AVDI
    if _AVDI is None:
        import physical_ai_av
        _AVDI = physical_ai_av.PhysicalAIAVDatasetInterface(local_dir=local_dir)
    return _AVDI


class _ClipFeatures:
    """Dataset interface with get_clip_feature memoised, so the windows of one clip share a download."""

    def __init__(self, avdi):
        self._avdi, self._memo = avdi, {}

    def __getattr__(self, name):
        return getattr(self._avdi, name)

    def get_clip_feature(self, clip_id, feature, maybe_stream=False):
        if (clip_id, feature) not in self._memo:
            self._memo[clip_id, feature] = self._avdi.get_clip_feature(clip_id, feature,
                                                                        maybe_stream=maybe_stream)
        return self._memo[clip_id, feature]


def build_clip(task):
    clip_id, t0s, chunk, cache, local_dir = task
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    avdi = _ClipFeatures(_avdi(local_dir))
    status = []
    for t0 in t0s:
        out = path_for(cache, clip_id, t0)
        if out.exists():
            status.append({"clip_id": clip_id, "status": "skip"})
            continue
        try:
            d = load_physical_aiavdataset(clip_id, t0_us=t0, maybe_stream=True, avdi=avdi)
        except Exception as e:  # noqa: BLE001  per-clip failures are recorded, not fatal
            status.append({"clip_id": clip_id, "status": "error", "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        write_sample(out, d, clip_id, t0, chunk)
        status.append({"clip_id": clip_id, "status": "ok"})
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="manifest name (data/manifests/<name>.parquet) or path")
    ap.add_argument("--cache", required=True, help="cache set name under $MOOP_CACHE")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--local-dir", default=None, help="local PhysicalAI-AV mirror, if any")
    args = ap.parse_args()

    man = load_manifest(args.manifest)
    tasks = [(c, [int(t) for t in g.t0_us], int(g.chunk.iloc[0]), args.cache, args.local_dir)
             for c, g in man.groupby("clip_id", sort=False)]
    print(f"{len(man)} samples -> {cache_root() / args.cache}", flush=True)
    done, errors, t0 = [], [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_clip, t) for t in tasks]
        for k, fut in enumerate(as_completed(futs), 1):
            for r in fut.result():
                (errors if r["status"] == "error" else done).append(r)
            if k % 25 == 0 or k == len(tasks):
                print(f"[{k}/{len(tasks)} clips] ok {len(done)}  errors {len(errors)}  "
                      f"{(time.time() - t0) / 60:.1f} min", flush=True)
    (cache_root() / args.cache / f"errors_{args.manifest.split('/')[-1]}.json").write_text(
        json.dumps(errors, indent=2))
    print(f"done: {len(done)} samples, {len(errors)} errors", flush=True)


if __name__ == "__main__":
    main()
