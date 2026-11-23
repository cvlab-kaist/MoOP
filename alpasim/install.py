"""Install MoOP slim-checkpoint support into an AlpaSim checkout (https://github.com/NVlabs/alpasim).

  python alpasim/install.py --alpasim /path/to/alpasim [--handover-fix]

1. copies alpamayo1_5_slim.py into the driver's model package
2. routes a checkpoint directory holding slim_meta.json to it in alpamayo1_5_model.py
3. adds the `pair_a15` topology (driver on one GPU, renderer + physics + trafficsim on another; our
   launcher puts all of them on the same card per shard)
4. --handover-fix applies the hand-over braking fix backported from NVlabs/alpasim main (2026-08-11);
   the u50 closed-loop runs used it, the paper's u40 runs did not (see alpasim/README.md)
Idempotent: steps already applied are skipped.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = """        model = Alpamayo1_5.from_pretrained(
            checkpoint_path,
            dtype=self.DTYPE,
            attn_implementation=_ATTN_IMPLEMENTATION,
        ).to(device)
"""
NEW = """        if is_slim_checkpoint(checkpoint_path):
            logger.info("Slim (structurally pruned) checkpoint detected at %s", checkpoint_path)
            model = load_slim(checkpoint_path, device=device, attn_implementation=_ATTN_IMPLEMENTATION)
        else:
            model = Alpamayo1_5.from_pretrained(
                checkpoint_path,
                dtype=self.DTYPE,
                attn_implementation=_ATTN_IMPLEMENTATION,
            ).to(device)
"""
IMPORT_ANCHOR = "from ..schema import ModelConfig\n"
IMPORT = "from .alpamayo1_5_slim import is_slim_checkpoint, load_slim\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpasim", required=True)
    ap.add_argument("--handover-fix", action="store_true")
    args = ap.parse_args()
    root = Path(args.alpasim)
    models = root / "src" / "driver" / "src" / "alpasim_driver" / "models"
    shutil.copy(HERE / "alpamayo1_5_slim.py", models / "alpamayo1_5_slim.py")
    f = models / "alpamayo1_5_model.py"
    src = f.read_text()
    if IMPORT not in src:
        assert IMPORT_ANCHOR in src, f"{f}: import anchor not found; patch by hand (see alpasim/README.md)"
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT, 1)
    if "is_slim_checkpoint(checkpoint_path)" not in src:
        assert OLD in src, f"{f}: from_pretrained block not found; patch by hand (see alpasim/README.md)"
        src = src.replace(OLD, NEW, 1)
    f.write_text(src)
    shutil.copy(HERE / "pair_a15.yaml", root / "src" / "wizard" / "configs" / "topology" / "pair_a15.yaml")
    print("slim loader + pair_a15 topology installed")
    if args.handover_fix:
        r = subprocess.run(["patch", "-p1", "--forward", "-i", str(HERE / "handover_fix.patch")], check=False,
                           cwd=root)
        print("handover fix", "applied" if r.returncode == 0 else "NOT applied (already present or conflicts)")


if __name__ == "__main__":
    main()
