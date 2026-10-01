#!/usr/bin/env python
"""D23 / A: compare learned stage-0 parameters with the data in the same PCA space.
Usage: python scripts/ckpt_vs_data.py TAG [TAG ...]   (run from the repo root, env vbcgcd)"""
import glob, json, sys
import numpy as np
from sklearn.decomposition import PCA


def arg(cmd, name, cast):
    i = len(cmd) - 1 - cmd[::-1].index(name)          # last occurrence wins, as in argparse
    return cast(cmd[i + 1])


for tag in sys.argv[1:]:
    run = glob.glob(f"runs/*__{tag}")[0]
    cfg = json.load(open(f"{run}/config.json"))
    cmd, feat = cfg["command"], next(iter(cfg["inputs"]))
    d, base = arg(cmd, "--num_dim", int), arg(cmd, "--base", int)
    lr = arg(cmd, "--lr", float) if "--lr" in cmd else 4e-6
    m = feat.rstrip("/").split("/")[-1]
    X, y = np.load(f"{feat}/features-{m}.npy"), np.load(f"{feat}/labels-{m}.npy").astype(int)
    ds = arg(cmd, "--dataset", str)
    if ds == "gc10":       # dataloaders/gc10.py: S0 = first n_use[c] rows of class c (file order), rest = pool
        sp = json.load(open("splits/gc10_split.json")); n_use = [sp["n_use"][c] for c in sp["classes"]]
    elif ds == "neudet":   # dataloaders/neudet.py: first 192 rows per class
        n_use = [192] * base
    else:
        raise SystemExit(f"{tag}: --dataset {ds} not supported")
    keep = np.concatenate([np.flatnonzero(y == k)[:n_use[k]] for k in range(base)])
    Z = PCA(d, svd_solver="full").fit_transform(X[keep]); yz = y[keep]
    xbar = np.stack([Z[yz == k].mean(0) for k in range(base)])
    ev_data = np.stack([np.linalg.eigvalsh(np.cov(Z[yz == k].T)) for k in range(base)])
    ck = f"{run}/artifacts/saved_models"
    mu = np.load(f"{ck}/stage0class_means.npy")[:base]
    S = np.load(f"{ck}/stage0class_covariances.npy")[:base]
    ev = np.linalg.eigvalsh((S + S.transpose(0, 2, 1)) / 2)
    r = np.linalg.norm(mu - xbar, axis=1) / np.linalg.norm(xbar, axis=1)
    big = np.abs(mu) > 0.5 * np.abs(mu).max()           # near-saturated coordinates
    agree = (np.sign(mu) == np.sign(xbar))[big].mean()  # ~1.0 = refit PCA aligned with the run's PCA
    B = 0.0334 * lr / 4e-6                              # Adam step budget (D23)
    print(f"{tag}: n_S0={len(yz)} lr={lr:g} budget={B:.4f} | max|mu|={np.abs(mu).max():.4f} "
          f"({np.abs(mu).max() / B:.2f} x budget) | r_mu mean={r.mean():.3f} [{r.min():.3f}, {r.max():.3f}] "
          f"| sign agree={agree:.2f} | eig Sigma [{ev.min():.3f}, {ev.max():.3f}] "
          f"vs data eig min {ev_data.min():.2f}, mean {ev_data.mean(1).min():.1f}-{ev_data.mean(1).max():.1f} | neg eig {int((ev < 0).sum())}")