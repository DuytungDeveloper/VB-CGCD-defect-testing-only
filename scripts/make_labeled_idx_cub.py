#!/usr/bin/env python
"""Write the HF train indices that phase 2 treats as labeled (CUB-200 t5, 100 base classes x 25 images).

Phase 2 labeled set = first 25 rows per class 0..99 of the merged .npy (utils/dataloader.py:309, dataloaders/cub200.py
num_labeled 2500 / base 100). Merged row order = shard files sorted as strings (feature_extractor/dino-cub200.py:44-45);
shard b = HF rows 128*b .. 128*b+127 (DataLoader, no shuffle, batch 128). See D17. Labels as in dino-cub200.py:105-109:
np.unique over the "text" strings, so the base classes are the first 100 names in alphabetical order.

Checks (each one stops the script when it fails):
  1. the labels of every --check folder equal hf[P] (P = row order of merge_npy);
  2. the 2500 rows equal the S0 labeled rows built by the loader itself (StrictPerClassIncrementalLoader as in
     dataloaders/cub200.py makeT5Loader), compared as a multiset of (feature row, label) on the FIRST --check folder;
  3. 2500 unique indices, 25 per class 0..99, class 0 first.
Duplicate feature rows are counted and reported; check 2 is exact also when they exist.

Usage (repo root, conda env vbcgcd; the first folder must hold features-* and test_features-*):
    python scripts/make_labeled_idx_cub.py --check features/<cub_ft_up_r0>/dino-vitb16-sl features/<other> ...
Output: splits/cub_labeled_hf_idx.npy, int64, 2500 = 100 classes x 25, class 0 first; inside a class, merged .npy order.
Prints the sha256 of the output and of the "text" column (data-order fingerprint: the code does not pin the HF revision).
"""
import argparse, collections, hashlib, os, sys
import numpy as np

EXTRACT_BS, N_BASE, PER_CLASS = 128, 100, 25
INCREMENT, NOVEL_INC, KNOWN_INC = 20, 25, 5                 # dataloaders/cub200.py makeT5Loader
OUT = "splits/cub_labeled_hf_idx.npy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs="+", required=True, help="merged feature folders; their labels must equal hf[P]")
    a = ap.parse_args()
    assert os.path.exists("main.py"), "run from the repo root"
    sys.path.insert(0, os.getcwd())

    from datasets import load_dataset
    ds = load_dataset("cassiekang/cub200_dataset", split="train")
    text = list(ds["text"])
    names = np.unique(np.array(text))
    hf = np.searchsorted(names, np.array(text)).astype(np.int64)          # = idx[t] of dino-cub200.py
    n = len(hf)
    n_shards = -(-n // EXTRACT_BS)
    print(f"HF train: {n} rows, {len(names)} classes, fingerprint {ds._fingerprint}")
    print(f"text column sha256 {hashlib.sha256(chr(10).join(text).encode()).hexdigest()}")

    shard = sorted(f"labels_{b}.npy" for b in range(n_shards))           # same order as sorted(os.listdir()) in merge_npy
    order = [int(f[len("labels_"):-len(".npy")]) for f in shard]
    P = np.concatenate([np.arange(b * EXTRACT_BS, min((b + 1) * EXTRACT_BS, n)) for b in order])
    assert np.array_equal(np.sort(P), np.arange(n)), "P is not a permutation"
    L = hf[P]

    for c in a.check:                                                     # check 1
        m = os.path.basename(c.rstrip("/"))
        merged = np.load(f"{c}/labels-{m}.npy").astype(np.int64)
        assert np.array_equal(merged, L), f"{c}: merged labels != hf[P], extraction order differs"
    print(f"check 1 OK: {n_shards} shards, first in order {order[:5]}; labels of {len(a.check)} folders == hf[P]")

    rows = np.concatenate([np.where(L == k)[0][:PER_CLASS] for k in range(N_BASE)])
    idx = P[rows].astype(np.int64)
    assert len(idx) == N_BASE * PER_CLASS and len(np.unique(idx)) == len(idx), "size / duplicates"   # check 3
    assert np.array_equal(hf[idx], np.repeat(np.arange(N_BASE), PER_CLASS)), "class blocks wrong"

    c = a.check[0].rstrip("/")                                            # check 2
    m = os.path.basename(c)
    X = np.load(f"{c}/features-{m}.npy")
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    dup = cnt[inv] > 1
    chosen = np.zeros(n, bool); chosen[rows] = True
    straddle = sum(1 for g in np.unique(inv[dup]) if 0 < chosen[inv == g].sum() < (inv == g).sum())
    print(f"feature rows of {c}: {int(dup.sum())} rows in duplicate groups, {straddle} groups straddle the 25-row cut")

    from utils.dataloader import StrictPerClassIncrementalLoader
    loader = StrictPerClassIncrementalLoader(data_dir=os.path.dirname(c), pretrained_model_name=m, base=N_BASE,
                                             increment=INCREMENT, num_labeled=N_BASE * PER_CLASS,
                                             num_novel_inc=NOVEL_INC, num_known_inc=KNOWN_INC)
    s0 = next(iter(loader.train_dataloader()))
    key = lambda x, y: [r.tobytes() + int(v).to_bytes(4, "little") for r, v in zip(np.asarray(x, np.float32), y)]
    got = collections.Counter(key(s0._x, np.asarray(s0._y).astype(np.int64)))
    want = collections.Counter(key(X[rows], L[rows]))
    assert got == want, f"loader S0 labeled set != chosen rows ({sum((got - want).values())} rows differ)"
    print(f"check 2 OK: loader S0 = {len(s0._x)} rows = the {len(rows)} chosen rows (multiset of feature row, label)")

    naive = np.concatenate([np.where(hf == k)[0][:PER_CLASS] for k in range(N_BASE)])
    print(f"check 3 OK: {len(idx)} unique, {PER_CLASS} per class 0..{N_BASE - 1}")
    print(f"overlap with naive first-{PER_CLASS}-per-class (HF order): {len(np.intersect1d(idx, naive))} / {len(idx)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and np.array_equal(np.load(OUT), idx):
        print(f"{OUT} unchanged")
    else:
        np.save(OUT, idx)
        print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
