"""Check that a built recipe (slim_meta.json) has exactly the kept units of the paper recipe.

  python scripts/verify_recipe.py runs/moop_u40/slim_meta.json            # recipe name read from the file
  python scripts/verify_recipe.py runs/x/slim_meta.json --recipe moop_u40
"""

import argparse
import hashlib
import json
from pathlib import Path

CHECKSUMS = Path(__file__).resolve().parents[1] / "recipes" / "checksums.json"


def recipe_hash(meta):
    """sha256 over the kept Q-head / MLP-channel indices of both towers."""
    blob = json.dumps({"vlm": meta["vlm"], "expert": meta["expert"]}, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta")
    ap.add_argument("--recipe")
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    name = args.recipe or meta.get("recipe")
    ref = json.loads(CHECKSUMS.read_text())[name]
    got = recipe_hash(meta)
    ok = got == ref["sha256"]
    print(f"{name}: {'MATCH' if ok else 'MISMATCH'} ({got[:16]}... vs {ref['sha256'][:16]}..., "
          f"source: {ref['source']})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
