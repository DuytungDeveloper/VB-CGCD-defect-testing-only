#!/usr/bin/env python
"""Rank the (lambda, tau) grid by the primary criterion only: k-means, n_init=10 (D18).

Much faster than eval_fold.py (no GMM sweep). Prints one line per run, then a sorted table.
Usage:  python scripts/pick_lambda.py F5 features/*__f5_*/dino-vitb16-sl
"""
import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "feature_extractor")
from ft_data import hf_to_npy_row, labeled_subset   # noqa: E402

IDX, SEEDS = "splits/c100_labeled_hf_idx.npy", 5


def acc(yt, yp):
    cm = confusion_matrix(yt, yp)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum() * 100


def main():
    fold, dirs = sys.argv[1], sys.argv[2:]
    sub = labeled_subset(IDX, fold)
    rows = []
    for d in dirs:
        m = os.path.basename(d.rstrip("/"))
        F = np.load(f"{d}/features-{m}.npy")
        L = np.load(f"{d}/labels-{m}.npy").astype(int)
        row = hf_to_npy_row(len(L))
        tr, nv = row[sub["train_idx"]], row[sub["novel_idx"]]
        X = PCA(n_components=384).fit(F[tr]).transform(F[nv])
        y = L[nv]
        a = [acc(y, KMeans(n_clusters=len(np.unique(y)), n_init=10, random_state=s).fit_predict(X))
             for s in range(SEEDS)]
        tag = os.path.basename(os.path.dirname(d.rstrip("/"))).split("__")[-1]
        rows.append((float(np.mean(a)), float(np.std(a, ddof=1) / np.sqrt(len(a))), tag))
        print(f"{tag:20s} km {rows[-1][0]:6.2f} +-{rows[-1][1]:.3f}", flush=True)

    print()
    for v, se, tag in sorted(rows, reverse=True):
        print(f"{v:6.2f} +-{se:.3f}  {tag}")
    base = [r for r in rows if "_l0_" in r[2]]
    if len(base) == 2:
        print(f"\nnoise floor (two identical lambda=0 runs): {abs(base[0][0] - base[1][0]):.3f}")
    if base:
        b = np.mean([r[0] for r in base])
        print(f"lambda=0 mean {b:.2f}; deltas: " +
              ", ".join(f"{t.split('_')[1]} {v - b:+.2f}" for v, _, t in rows if "_l0_" not in t))


if __name__ == "__main__":
    main()
