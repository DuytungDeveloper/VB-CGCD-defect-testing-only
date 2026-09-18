"""Which images the fine-tuning step sees, and the validation folds.

Two knobs, both off by default (see scripts/run_phase1_extract.sh):
  FT_LABELED_IDX  splits/c100_labeled_hf_idx.npy -> fine-tune only on the 400 imgs/class
                  that phase 2 treats as labeled (fixes D9; upstream uses 500/class).
  FT_FOLD         F1..F5 -> hold out 10 classes as fake-novel for (lambda, tau) search.

The index file is class-ordered: 400 entries for class 0, then class 1, ... (see D17).
Inside a class the order is the merged .npy row order, so "first 320" is deterministic.
"""
import numpy as np

PER_CLASS = 400
TRAIN_PER_CLASS = 320          # Q1b: 320 train / 80 val per fake-known class
N_BASE = 50
FOLDS = {f"F{i + 1}": tuple(range(i * 10, i * 10 + 10)) for i in range(5)}


def labeled_subset(idx_path, fold=None):
    """Return dict with train_idx / val_idx / novel_idx as HF train indices.

    fold=None: train_idx = all 20000 labeled indices, the other two empty.
    fold="F5": classes 40..49 are fake-novel (novel_idx, 4000), the other 40 classes
               give train_idx (12800) and val_idx (3200).
    """
    idx = np.load(idx_path)
    if idx.shape != (N_BASE * PER_CLASS,):
        raise ValueError(f"{idx_path}: expected {N_BASE * PER_CLASS} indices, got {idx.shape}")
    per_class = idx.reshape(N_BASE, PER_CLASS)

    if fold is None:
        return {"train_idx": per_class.reshape(-1), "val_idx": np.array([], dtype=idx.dtype),
                "novel_idx": np.array([], dtype=idx.dtype)}
    if fold not in FOLDS:
        raise ValueError(f"unknown fold {fold!r}, expected one of {sorted(FOLDS)}")

    novel = list(FOLDS[fold])
    known = [k for k in range(N_BASE) if k not in novel]
    return {"train_idx": per_class[known, :TRAIN_PER_CLASS].reshape(-1),
            "val_idx": per_class[known, TRAIN_PER_CLASS:].reshape(-1),
            "novel_idx": per_class[novel, :].reshape(-1)}


def hf_to_npy_row(n=50000, extract_bs=128):
    """Map HF train index -> row in the merged .npy (see D17 and scripts/make_labeled_idx.py).

    Row order = shard files sorted as strings; shard b = HF rows extract_bs*b ...
    Returns an array `row` with row[hf_index] = npy_row.
    """
    import numpy as np
    n_shards = -(-n // extract_bs)
    names = sorted(f"labels_{b}.npy" for b in range(n_shards))
    order = [int(f[len("labels_"):-len(".npy")]) for f in names]
    P = np.concatenate([np.arange(b * extract_bs, min((b + 1) * extract_bs, n)) for b in order])
    row = np.empty(n, dtype=np.int64)
    row[P] = np.arange(n)
    return row
