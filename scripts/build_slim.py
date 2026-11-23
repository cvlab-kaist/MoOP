"""Build a pruned model from importance scores (or a shipped recipe) and write its recipe.

  # from importance scores
  python scripts/build_slim.py --recipe moop_u40 \
      --importance recipes/importance/importance.npz --out runs/moop_u40
  python scripts/build_slim.py --recipe moop_e_u40 \
      --importance recipes/importance/importance.npz \
      --expert-importance recipes/importance/expert_importance.npz --out runs/moop_e_u40
  # from a shipped recipe (exact paper masks)
  python scripts/build_slim.py --from-recipe recipes/moop_u40 --out runs/moop_u40

Writes <out>/slim_meta.json (kept indices; the model is rebuilt from it by moop.slim.load_slim),
config.json and summary.txt. --save-state also writes the ~17 GB state_dict (needed by the AlpaSim
driver container, which does not load the base weights itself).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from moop import MODEL_ID, MODEL_REV
from moop.masks import RECIPES, build_masks
from moop.slim import apply_surgery, check_slim, masks_from_meta, n_params, save_slim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES))
    ap.add_argument("--from-recipe", help="directory holding a slim_meta.json to rebuild")
    ap.add_argument("--importance", help="importance.npz from compute_importance.py")
    ap.add_argument("--expert-importance", help="expert_importance.npz (MoOP+E recipes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-state", action="store_true")
    ap.add_argument("--masks-only", action="store_true", help="write slim_meta.json without loading the model")
    args = ap.parse_args()
    if bool(args.recipe) == bool(args.from_recipe):
        ap.error("give exactly one of --recipe / --from-recipe")

    if args.from_recipe:
        meta_in = json.loads((Path(args.from_recipe) / "slim_meta.json").read_text())
        name = meta_in.get("recipe", Path(args.from_recipe).name)
    else:
        imp = dict(np.load(args.importance))
        eimp = dict(np.load(args.expert_importance)) if args.expert_importance else None
        vq, vm, eq, em = build_masks(args.recipe, imp, eimp)
        name = args.recipe

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.masks_only:
        if args.from_recipe:
            ap.error("--masks-only applies to --recipe")
        meta = {"vlm": [{"q": np.nonzero(vq[i])[0].tolist(), "mlp": np.nonzero(vm[i])[0].tolist()}
                        for i in range(len(vq))],
                "expert": [{"q": np.nonzero(eq[i])[0].tolist(), "mlp": np.nonzero(em[i])[0].tolist()}
                           for i in range(len(eq))],
                "kvonly_layers": [], "recipe": name, "model_revision": MODEL_REV}
        (out / "slim_meta.json").write_text(json.dumps(meta))
        print(f"{name}: masks -> {out / 'slim_meta.json'}")
        return

    import torch
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    # built on the CPU; the attention implementation does not touch the weights
    model = Alpamayo1_5.from_pretrained(MODEL_ID, revision=MODEL_REV, dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    before = n_params(model)
    if args.from_recipe:
        vq, vm, eq, em = masks_from_meta(meta_in, model)
    meta = apply_surgery(model, vq, vm, eq, em)
    check_slim(model)
    after = n_params(model)
    meta.update(recipe=name, model_revision=MODEL_REV,
                params={"dense": before, "slim": after, "removed": before - after})
    save_slim(model, meta, out, write_state=args.save_state)
    (out / "config.json").write_text(json.dumps({
        "recipe": name, "importance": args.importance, "expert_importance": args.expert_importance,
        "from_recipe": args.from_recipe, "model": MODEL_ID, "model_revision": MODEL_REV}, indent=2))
    summary = (f"{name}: removed {before - after:,} of {before:,} parameters "
               f"({100 * (before - after) / before:.2f}%)\n")
    (out / "summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
