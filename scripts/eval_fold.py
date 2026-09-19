#!/usr/bin/env python
"""Validation metric for the (lambda, tau) search (Q2). CPU only, ~1 min per run.

Reads the features of a phase-1 validation run and its fold, then reports:
  novel_acc  main criterion: PCA(384) fit on the 40 fake-known TRAIN features (mimics phase 2,
             where PCA is fit on labeled S0 only), transform the 4000 fake-novel images,
             GaussianMixture(k=10) over 5 seeds, cluster accuracy via Hungarian; mean and std.
  known_acc  constraint: CE head (saved in ft_ckpt.pt) on the 3200 held-out val images.
Usage:
    python scripts/eval_fold.py features/<run_id>/dino-vitb16-sl F5 [--out runs/<run_id>/fold_metrics.json]
"""
import argparse, json, os, sys
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, "feature_extractor")
from ft_data import hf_to_npy_row, labeled_subset   # noqa: E402

IDX = "splits/c100_labeled_hf_idx.npy"
N_DIM, N_GMM_SEEDS = 384, 20


def cluster_acc(y_true, y_pred):
    """Hungarian accuracy: best one-to-one map from cluster id to class id."""
    cm = confusion_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    return cm[r, c].sum() / cm.sum() * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir")
    ap.add_argument("fold")
    ap.add_argument("--out")
    a = ap.parse_args()

    m = os.path.basename(a.feat_dir.rstrip("/"))
    F = np.load(f"{a.feat_dir}/features-{m}.npy")
    L = np.load(f"{a.feat_dir}/labels-{m}.npy").astype(int)
    row = hf_to_npy_row(len(L))
    sub = labeled_subset(IDX, a.fold)
    tr, va, nv = (row[sub[k]] for k in ("train_idx", "val_idx", "novel_idx"))

    pca = PCA(n_components=N_DIM).fit(F[tr])           # fit on fake-known train only
    Xn, yn = pca.transform(F[nv]), L[nv]
    k = len(np.unique(yn))
    # primary: same clustering as phase 2 (GMM, full covariance, 384 dims). NOTE: 739k params
    # per 4000 samples -> bimodal solutions, and ELBO anti-correlates with accuracy (r = -0.999
    # measured on c100_consmoke), so never use n_init. Same seed set for every config (paired).
    accs = [cluster_acc(yn, GaussianMixture(n_components=k, random_state=s).fit_predict(Xn))
            for s in range(N_GMM_SEEDS)]
    # secondary, more stable: diagonal covariance (768x fewer params) and k-means (paper C.3 text)
    accs_diag = [cluster_acc(yn, GaussianMixture(n_components=k, covariance_type="diag",
                                                 random_state=s).fit_predict(Xn))
                 for s in range(N_GMM_SEEDS)]
    # primary since 2026-09-19: k-means n_init=10 is the best phase-2 config (D18).
    accs_km = [cluster_acc(yn, KMeans(n_clusters=k, n_init=10, random_state=s).fit_predict(Xn))
               for s in range(N_GMM_SEEDS)]
    acc_km = float(np.mean(accs_km))

    ck = f"{os.path.dirname(a.feat_dir.rstrip('/'))}/ft_ckpt.pt"
    st = torch.load(ck, map_location="cpu", weights_only=True)["state"]
    W, b = st["head.weight"].numpy(), st["head.bias"].numpy()
    known_acc = float((np.argmax(F[va] @ W.T + b, axis=1) == L[va]).mean() * 100)

    out = {"fold": a.fold, "feat_dir": a.feat_dir,
           "novel_acc": float(np.mean(accs)), "novel_acc_std": float(np.std(accs, ddof=1)),
           "novel_acc_se": float(np.std(accs, ddof=1) / np.sqrt(len(accs))),
           "novel_acc_median": float(np.median(accs)), "n_gmm_seeds": len(accs),
           "novel_acc_seeds": [round(x, 2) for x in accs],
           "novel_acc_diag": float(np.mean(accs_diag)),
           "novel_acc_diag_se": float(np.std(accs_diag, ddof=1) / np.sqrt(len(accs_diag))),
           "novel_acc_kmeans": acc_km,
           "novel_acc_kmeans_se": float(np.std(accs_km, ddof=1) / np.sqrt(len(accs_km))),
           "novel_acc_kmeans_min": float(np.min(accs_km)),
           "novel_acc_kmeans_max": float(np.max(accs_km)),
           "known_acc": known_acc, "n_train": len(tr), "n_val": len(va), "n_novel": len(nv)}
    print(json.dumps(out, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
