#!/usr/bin/env python
"""Write the HF train indices that phase 2 treats as labeled (CIFAR-100, 50 base classes).

Phase 2 labeled set = first 400 rows per class 0..49 of the merged .npy (utils/dataloader.py:309).
Merged row order = shard files sorted as strings (feature_extractor/dino-cifar100.py:44-45);
shard b = HF rows 128*b .. 128*b+127 (DataLoader, no shuffle, batch 128). See D17.

Usage (repo root):
    python scripts/make_labeled_idx.py --check features/<run_id>/dino-vitb16-sl
Output: splits/c100_labeled_hf_idx.npy, int64, 20000 = 50 classes x 400,
        class 0 first; inside a class, merged .npy row order.
"""
import argparse, hashlib, os
import numpy as np
from datasets import load_dataset

EXTRACT_BS, N_BASE, PER_CLASS = 128, 50, 400
OUT = "splits/c100_labeled_hf_idx.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="merged feature folder; its labels must equal hf[P]")
    a = ap.parse_args()

    hf = np.array(load_dataset("uoft-cs/cifar100", split="train")["fine_label"]).astype(np.int64)
    n = len(hf)
    n_shards = -(-n // EXTRACT_BS)

    # same order as sorted(os.listdir(labels_dir)) in merge_npy
    names = sorted(f"labels_{b}.npy" for b in range(n_shards))
    order = [int(f[len("labels_"):-len(".npy")]) for f in names]
    P = np.concatenate([np.arange(b * EXTRACT_BS, min((b + 1) * EXTRACT_BS, n)) for b in order])
    assert np.array_equal(np.sort(P), np.arange(n)), "P is not a permutation"
    L = hf[P]

    m = os.path.basename(a.check.rstrip("/"))
    merged = np.load(f"{a.check}/labels-{m}.npy").astype(np.int64)
    assert np.array_equal(merged, L), "merged labels != hf[P]: extraction order differs"

    idx = np.concatenate([P[np.where(L == k)[0][:PER_CLASS]] for k in range(N_BASE)]).astype(np.int64)
    assert len(idx) == N_BASE * PER_CLASS and len(np.unique(idx)) == len(idx), "size / duplicates"
    assert np.array_equal(hf[idx], np.repeat(np.arange(N_BASE), PER_CLASS)), "class blocks wrong"

    naive = np.concatenate([np.where(hf == k)[0][:PER_CLASS] for k in range(N_BASE)])
    print(f"shards {n_shards}, first in order {order[:5]}")
    print(f"checks OK: P permutation, hf[P] == {a.check} labels, {len(idx)} unique, 400 per class 0..49")
    print(f"overlap with naive first-400-per-class (HF order): {len(np.intersect1d(idx, naive))} / {len(idx)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and np.array_equal(np.load(OUT), idx):
        print(f"{OUT} unchanged")
    else:
        np.save(OUT, idx)
        print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
