#!/usr/bin/env python
"""VB-CGCD phase 1 check.

  python scripts/check_phase1.py OUT_DIR [--ckpt PATH]

OUT_DIR holds dino-vitb16-sl/*.npy and the dino_vitb16_*/ batch folders
(e.g. features/<run_id>). Default checkpoint: OUT_DIR/ft_ckpt.pt.
Exit code 0 only if every check passes.
"""
import math, sys
from pathlib import Path
import numpy as np

MODEL, RAW, BS = "dino-vitb16-sl", "dino_vitb16", 128   # BS = extraction --batch-size
SPLITS = {"train": ("", 50000), "test": ("test_", 10000)}
N_CLASS, DIM = 100, 768
CKPT_PARAMS = 888_676   # 221,184 BOFT + 76,900 head + 590,592 pooler (PATCH 5)

fails = []


def check(ok, msg):
    print(("  PASS " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def main():
    argv = sys.argv[1:]
    if not argv:
        sys.exit(__doc__)
    out = Path(argv[0])
    ckpt = Path(argv[argv.index("--ckpt") + 1]) if "--ckpt" in argv else out / "ft_ckpt.pt"

    for split, (pre, n) in SPLITS.items():
        f, l = out / MODEL / f"{pre}features-{MODEL}.npy", out / MODEL / f"{pre}labels-{MODEL}.npy"
        if not (f.exists() and l.exists()):
            check(False, f"{split}: merged files missing in {out / MODEL}")
            continue
        X, y = np.load(f), np.load(l)
        check(X.shape == (n, DIM), f"{split} features shape {X.shape}")
        check(y.shape == (n,), f"{split} labels shape {y.shape}")
        check(bool(np.isfinite(X).all()), f"{split} features all finite")
        check(float(np.abs(X).max()) <= 1.0, f"{split} features inside [-1, 1] (tanh pooler), max |x| = {np.abs(X).max():.4f}")
        u, c = np.unique(y, return_counts=True)
        check(len(u) == N_CLASS and bool((c == n // N_CLASS).all()),
              f"{split} labels: {len(u)} classes, {c.min()}..{c.max()} per class (want {n // N_CLASS})")
        for kind in ("features", "labels"):
            d = out / f"{RAW}_{pre}{kind}"
            cnt = len(list(d.glob("*.npy"))) if d.exists() else 0
            check(cnt == math.ceil(n / BS), f"{d.name}: {cnt} batch files (want {math.ceil(n / BS)}; more = stale files merged)")

    if ckpt.exists():
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        state = ck["state"]
        total = sum(t.numel() for t in state.values())
        pooler = [k for k in state if "pooler.dense" in k]
        check(total == CKPT_PARAMS, f"checkpoint params {total:,} (want {CKPT_PARAMS:,}), epoch {ck.get('epoch')}")
        check(len(pooler) == 2, f"checkpoint pooler keys: {len(pooler)} (want 2)")
    else:
        check(False, f"checkpoint missing: {ckpt}")

    print(f"\n{'OK' if not fails else 'FAILED'}: {len(fails)} failed check(s)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()