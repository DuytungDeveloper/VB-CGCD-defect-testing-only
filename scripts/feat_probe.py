#!/usr/bin/env python
"""Diagnostic probes of a feature folder: k-means (k known) cluster accuracy on all / base / novel
train rows, silhouette estimate of k, and a linear probe trained with ALL train labels (upper bound,
never part of the protocol). Usage: python scripts/feat_probe.py FEATURE_DIR [--n_base 5] [--kmax 16] [--no_sil]"""
import argparse, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


def cacc(y, c):
    k = max(y.max(), c.max()) + 1; M = np.zeros((k, k), int); np.add.at(M, (y, c), 1)
    r, cc = linear_sum_assignment(-M); return 100 * M[r, cc].sum() / len(y)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--n_base", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=16); ap.add_argument("--no_sil", action="store_true")
    a = ap.parse_args(); m = os.path.basename(a.dir.rstrip("/"))
    X, y = np.load(f"{a.dir}/features-{m}.npy"), np.load(f"{a.dir}/labels-{m}.npy").astype(int)
    Xt, yt = np.load(f"{a.dir}/test_features-{m}.npy"), np.load(f"{a.dir}/test_labels-{m}.npy").astype(int)
    K, nb = y.max() + 1, a.n_base
    print(a.dir, X.shape)
    for name, mask, off, k in (("all", y >= 0, 0, K), ("base", y < nb, 0, nb), ("novel", y >= nb, nb, K - nb)):
        Z = PCA(min(128, mask.sum() - 1), random_state=0).fit_transform(X[mask])
        c = KMeans(k, n_init=10, random_state=0).fit_predict(Z)
        print(f"  k-means {name:5s} (k={k:2d}) cluster acc {cacc(y[mask] - off, c):5.1f}%")
    if not a.no_sil:
        Z = PCA(128, random_state=0).fit_transform(X)
        sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)) for k in range(2, a.kmax + 1)}
        print(f"  silhouette best k = {max(sil, key=sil.get)} (true {K})")
    p = LogisticRegression(max_iter=5000).fit(X, y).predict(Xt)
    rec = [100 * (p[yt == k] == k).mean() for k in range(K)]
    print(f"  linear probe acc {100 * (p == yt).mean():5.1f}% | balanced {np.mean(rec):5.1f}% | per class {[round(r) for r in rec]}")


if __name__ == "__main__":
    main()
