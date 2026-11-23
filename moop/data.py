"""Sample lists and the per-clip sample cache.

Every evaluation and calibration set is a manifest under `data/manifests/` (clip_id, t0_us, chunk).
Samples are read from a per-clip npz cache built once from the PhysicalAI-AV dataset with
`scripts/build_cache.py`: the 16 frames the model consumes (4 cameras x 4 frames) stored as JPEG
(quality 95) plus the ego trajectory, about 6 MB per clip. The reported numbers were produced from
this cache, so frames go through the same JPEG round trip here.

The cache root is `$MOOP_CACHE` (default `./cache`); a set lives in `<root>/<name>/samples/`.
"""

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
KEYS = ("image_frames", "camera_indices", "ego_history_xyz", "ego_history_rot",
        "ego_future_xyz", "ego_future_rot")


def cache_root():
    return Path(os.environ.get("MOOP_CACHE", REPO / "cache"))


def clip_seed(global_seed, clip_id):
    """Per-clip seed derived from the clip id, so results do not depend on order or sharding."""
    h = hashlib.sha256(f"{global_seed}:{clip_id}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def load_manifest(name_or_path):
    """A manifest by name (`calib_100`, `test_500`, `ood_val`, `ood_train`) or by path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = MANIFESTS / f"{name_or_path}.parquet"
    return pd.read_parquet(p)


def samples(name_or_path):
    """[(clip_id, t0_us)] in manifest order."""
    df = load_manifest(name_or_path)
    return list(zip(df["clip_id"], df["t0_us"].astype(int)))


def path_for(cache, clip_id, t0_us):
    return cache_root() / cache / "samples" / f"{clip_id}__t0_{t0_us}.npz"


def load_cached(path, device=None):
    """Rebuild the dataset loader's return dict from a cached npz.

    Returns image_frames (n_cam, n_frame, 3, H, W) uint8 and the ego tensors with their original
    (1, 1, T, ...) shapes, i.e. a drop-in for `alpamayo1_5.load_physical_aiavdataset`.
    """
    z = np.load(path, allow_pickle=True)
    n_cam, n_frame = int(z["n_cam"]), int(z["n_frame"])
    H, W = int(z["H"]), int(z["W"])
    frames = np.empty((n_cam, n_frame, 3, H, W), dtype=np.uint8)
    for c in range(n_cam):
        for f in range(n_frame):
            img = np.asarray(Image.open(io.BytesIO(z["jpeg_bytes"][c * n_frame + f])))
            frames[c, f] = np.transpose(img, (2, 0, 1))  # (H, W, 3) -> (3, H, W)
    out = {
        "image_frames": torch.from_numpy(frames),
        "camera_indices": torch.from_numpy(z["camera_indices"]),
        "ego_history_xyz": torch.from_numpy(z["ego_history_xyz"]),
        "ego_history_rot": torch.from_numpy(z["ego_history_rot"]),
        "ego_future_xyz": torch.from_numpy(z["ego_future_xyz"]),
        "ego_future_rot": torch.from_numpy(z["ego_future_rot"]),
        "t0_us": int(z["t0_us"]),
        "clip_id": str(z["clip_id"]),
    }
    if device is not None:
        for k in KEYS:
            out[k] = out[k].to(device)
    return out


def write_sample(out, d, clip_id, t0_us, chunk):
    """Write one dataset sample (loader dict) as a cache npz: 16 JPEGs at quality 95 + ego tensors."""
    frames = d["image_frames"].numpy()  # (n_cam, n_frame, 3, H, W) uint8
    n_cam, n_frame = frames.shape[0], frames.shape[1]
    jpegs = []
    for c in range(n_cam):
        for f in range(n_frame):
            buf = io.BytesIO()
            Image.fromarray(np.transpose(frames[c, f], (1, 2, 0))).save(buf, format="JPEG", quality=95)
            jpegs.append(buf.getvalue())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:  # np.savez appends ".npz" to a path lacking it
        np.savez(
            fh,
            jpeg_bytes=np.array(jpegs, dtype=object),
            n_cam=np.int32(n_cam), n_frame=np.int32(n_frame),
            H=np.int32(frames.shape[3]), W=np.int32(frames.shape[4]),
            camera_indices=d["camera_indices"].numpy(),
            ego_history_xyz=d["ego_history_xyz"].numpy(),
            ego_history_rot=d["ego_history_rot"].numpy(),
            ego_future_xyz=d["ego_future_xyz"].numpy(),
            ego_future_rot=d["ego_future_rot"].numpy(),
            t0_us=np.int64(t0_us), clip_id=str(clip_id), chunk=np.int32(chunk),
        )
    tmp.rename(out)
