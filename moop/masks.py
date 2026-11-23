"""Importance -> keep-masks for the MoOP recipes.

VLM: every layer loses the same number of Q heads and MLP channels (uniform allocation). Within a
layer the units with the lowest score are cut (np.argsort order). The MoOP score is the max of the two
objectives' within-layer ranks:  s = max(rank(I_FM), rank(I_CE)),  rank in [0, 1] per layer.
The FM-only / CE-only ablations use the raw score of one objective.

Expert (MoOP+E): Q heads and KV are kept; per layer the lowest MLP channels under the step-aggregated
FM score (`znorm`) are cut.

Budgets (Alpamayo 1.5: 36 VLM layers x 32 Q heads x 12288 channels; 36 expert layers x 8256 channels):
  u40: 13 heads + 4898 channels cut per VLM layer   -> 2,657,452,032 parameters (24.0% of the model)
  u50: 16 heads + 6144 channels cut per VLM layer   -> 3,321,888,768 parameters
  e93.75: 7740 of 8256 expert MLP channels cut per layer (516 kept)
`l0keep` keeps all of layer 0's Q heads and cuts the same number of parameters as extra MLP channels
spread evenly over the layers (one head = 2*128*4096 = 1,048,576 parameters, one channel = 3*4096 =
12,288). The paper recipe is the plain one; at the cut, several of layer 0's heads tie in rank, which
makes a re-measured plain recipe sensitive to measurement noise, and l0keep removes that sensitivity.

Ties: the last VLM layer's FM scores are exactly zero (its outputs never reach the KV cache the
expert reads), so which of its MLP channels an FM-based recipe cuts is decided by np.argsort's order
among equal values. That order changed between numpy releases; the recipes were built with numpy
2.2.6, which pyproject.toml pins. The shipped `recipes/*/slim_meta.json` are the exact masks.
"""

import numpy as np

BUDGETS = {"u40": (13, 4898), "u50": (16, 6144)}
EXPERT_CUTS = {"e93.75": 7740}
RECIPES = {
    # name: (VLM score, budget, layer-0 exemption, expert cut)
    "moop_u40": ("moop", "u40", False, None),
    "moop_e_u40": ("moop", "u40", False, "e93.75"),
    "fm_only_u40": ("fm", "u40", False, None),
    "ce_only_u40": ("ce", "u40", False, None),
    "moop_u40_l0keep": ("moop", "u40", True, None),
    "moop_u50": ("moop", "u50", False, None),
    "moop_e_u50": ("moop", "u50", False, "e93.75"),
    "fm_only_u50": ("fm", "u50", False, None),
    "ce_only_u50": ("ce", "u50", False, None),
}


def rank_norm(scores):
    """Per-layer rank in [0, 1]. (L, U) -> (L, U)."""
    out = np.zeros_like(scores, dtype=float)
    n = scores.shape[1]
    for i in range(scores.shape[0]):
        out[i] = np.argsort(np.argsort(scores[i])) / max(n - 1, 1)
    return out


def vlm_scores(imp, kind):
    """(L, H) Q-head and (L, I) MLP-channel scores for 'moop', 'fm' or 'ce'."""
    if kind == "moop":
        return (np.maximum(rank_norm(imp["traj_vlm_q"]), rank_norm(imp["coc_vlm_q"])),
                np.maximum(rank_norm(imp["traj_vlm_mlp"]), rank_norm(imp["coc_vlm_mlp"])))
    key = {"fm": "traj", "ce": "coc"}[kind]
    return imp[f"{key}_vlm_q"], imp[f"{key}_vlm_mlp"]


def cut_lowest(scores, counts):
    """Keep-mask (1 = keep) dropping the counts[l] lowest-scoring units of each layer l."""
    mask = np.ones_like(scores, dtype=float)
    for li, k in enumerate(counts):
        if k > 0:
            mask[li, np.argsort(scores[li])[:k]] = 0.0
    return mask


def vlm_cut_counts(budget, n_layers, l0keep, head_dim=128, hidden=4096):
    """Per-layer (Q heads, MLP channels) to cut."""
    n_q, n_m = BUDGETS[budget]
    cq = np.full(n_layers, n_q, dtype=int)
    cm = np.full(n_layers, n_m, dtype=int)
    if l0keep:
        n_extra, _ = divmod(n_q * 2 * head_dim * hidden, 3 * hidden)   # 13 heads -> 1109 channels
        extra = np.full(n_layers, n_extra // n_layers)
        extra[: n_extra - extra.sum()] += 1
        cq[0] = 0
        cm += extra
    return cq, cm


def build_masks(recipe, imp, expert_imp=None, vlm_shape=(36, 32, 12288), expert_shape=(36, 16, 8256)):
    """-> (vlm_q (L, H), vlm_mlp (L, I), exp_q (Le, He), exp_mlp (Le, Ie)) keep-masks for a recipe."""
    kind, budget, l0keep, ecut = RECIPES[recipe]
    n_layers = vlm_shape[0]
    sq, sm = vlm_scores(imp, kind)
    cq, cm = vlm_cut_counts(budget, n_layers, l0keep)
    vq, vm = cut_lowest(sq, cq), cut_lowest(sm, cm)
    eq = np.ones(expert_shape[:2])
    em = np.ones((expert_shape[0], expert_shape[2]))
    if ecut is not None:
        if expert_imp is None:
            raise ValueError(f"{recipe} needs the expert importance (znorm)")
        em = cut_lowest(expert_imp["exp_mlp_znorm"], np.full(expert_shape[0], EXPERT_CUTS[ecut]))
    return vq, vm, eq, em
