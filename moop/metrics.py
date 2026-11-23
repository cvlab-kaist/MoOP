"""Trajectory errors, CoC degeneracy, and the bootstrap confidence intervals used in the paper."""

from itertools import groupby

import numpy as np

HORIZONS = {"1.6s": "_h16", "3.2s": "_h32", "6.4s": ""}   # 10 Hz: 16 / 32 / 64 waypoints
K = 6                 # minADE@6 / minFDE@6: the first 6 of the 8 stored samples (seeds base + k)
N_BOOT = 200_000
CHUNK = 10_000


def ade_fde(pred_xy, gt_xy):
    """Per-sample ADE (mean L2 over steps) and FDE (endpoint L2). pred_xy (K, T, 2), gt_xy (T, 2)."""
    err = np.linalg.norm(pred_xy - gt_xy[None], axis=2)  # (K, T)
    return err.mean(1), err[:, -1]


def min_at_k(row, kind, horizon="6.4s", k=K):
    """minADE@k or minFDE@k of one open-loop row (kind 'ade' or 'fde')."""
    return float(min(row[f"{kind}_rollout_k{HORIZONS[horizon]}"][:k]))


def coc_degenerate(text):
    """Degenerate CoC: empty, or 'soup' (>5% non-ASCII, <50% unique words, or >300 characters)."""
    s = str(text).strip()
    words = s.split()
    uniq = len(set(words)) / max(len(words), 1)
    nonascii = sum(ord(ch) > 127 for ch in s) / max(len(s), 1)
    empty = len(s) == 0
    soup = (not empty) and (nonascii > 0.05 or uniq < 0.5 or len(s) > 300)
    run = max((len(list(g)) for _, g in groupby(s.replace(" ", ""))), default=0)
    return {"empty": bool(empty), "soup": bool(soup), "degenerate": bool(empty or soup),
            "degenerate_strict": bool(empty or soup or run >= 10), "len": len(s)}


def boot_many(series, n_boot=N_BOOT, seed=0, groups=None):
    """Bootstrap the means of aligned series with one shared set of draws.

    series: {name: (N,)}. groups=None resamples items; otherwise whole clusters (groups (N,) labels) are
    resampled and the mean is the ratio of resampled sums to resampled counts. Returns
    {name: (lo, hi, p)} with the 95% percentile interval and the two-sided bootstrap p of mean = 0
    (meaningful for paired differences).
    """
    names = list(series)
    D = np.stack([np.asarray(series[k], dtype=float) for k in names], 1)  # (N, M)
    if groups is not None:
        keys, inv = np.unique(np.asarray(groups), return_inverse=True)
        S = np.zeros((len(keys), D.shape[1]))  # (G, M) per-cluster sums
        np.add.at(S, inv, D)
        cnt = np.bincount(inv, minlength=len(keys)).astype(float)  # (G,)
    else:
        S, cnt = D, np.ones(len(D))
    n = len(S)
    rng = np.random.RandomState(seed)
    means = []
    for s in range(0, n_boot, CHUNK):
        c = min(CHUNK, n_boot - s)
        idx = rng.randint(0, n, size=(c, n))  # (c, n)
        w = np.bincount((idx + n * np.arange(c)[:, None]).ravel(), minlength=c * n).reshape(c, n)  # (c, n)
        means.append((w @ S) / (w @ cnt)[:, None])  # (c, M)
    m = np.concatenate(means)  # (n_boot, M)
    lo, hi = np.percentile(m, [2.5, 97.5], axis=0)
    p = np.minimum(1.0, 2 * np.minimum((m <= 0).mean(0), (m >= 0).mean(0)))
    return {k: (float(lo[i]), float(hi[i]), float(p[i])) for i, k in enumerate(names)}
