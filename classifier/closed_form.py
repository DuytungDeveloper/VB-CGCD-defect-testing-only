"""Closed-form shrinkage estimator for the per-class Gaussians (improvement 1 redesign, "B").

For every class k of the current pass (n_k samples in the PCA space, d dims):
    mu_k    = sample mean
    S_k     = (1/n_k) sum_i (x_i - mu_k)(x_i - mu_k)^T            (MLE)
    nu_k    = tr(S_k) / d
    Sigma_k = (1 - a_k) S_k + a_k nu_k I,   a_k = Ledoit-Wolf analytic shrinkage
Same as the MAP under an inverse-Wishart prior with Psi = kappa nu_k I and a = kappa / (n_k + kappa),
i.e. kappa_hat = n_k a / (1 - a): the prior strength is estimated from the data (empirical Bayes).
Edge cases (fixed before running): n_k == 1 -> Sigma = nu_pool I (pooled average variance of the pass);
n_k == 0 -> the slot keeps its value in `prev` (the pass-F estimate during the S-pass).
Slots not in `classes` are copied unchanged from `prev` (old classes stay fixed, D12).
Sources: Ledoit & Wolf (2004, J. Multivariate Anal.); Friedman (1989, RDA, JASA); Murphy (2012, Sec. 4.6)."""
import numpy as np
from sklearn.covariance import LedoitWolf


def closed_form_params(X, y, classes, prev=None, num_classes=None):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    d = X.shape[1]
    if prev is None:                       # S0: unseen slots keep SVI's initial values (mu = 0, Sigma = I)
        means = np.zeros((num_classes, d))
        covs = np.tile(np.eye(d), (num_classes, 1, 1))
    else:
        means = np.array(prev["class_means"], dtype=np.float64)
        covs = np.array(prev["class_covs"], dtype=np.float64)
    info = {}
    for k in classes:
        Xk = X[y == k]
        n = len(Xk)
        if n >= 2:
            lw = LedoitWolf().fit(Xk)
            means[k], covs[k] = lw.location_, lw.covariance_
            a = float(lw.shrinkage_)
            info[k] = (n, a, n * a / (1 - a) if a < 1 else float("inf"))
        elif n == 1:
            means[k] = Xk[0]
            covs[k] = _pooled_nu(X, y, classes) * np.eye(d)
            info[k] = (1, 1.0, float("inf"))
        else:
            info[k] = (0, None, None)      # no samples: keep prev
    return {"class_means": means, "class_covs": covs}, info


def _pooled_nu(X, y, classes):
    """tr(pooled within-class covariance) / d over the classes with >= 2 samples."""
    r = np.concatenate([X[y == k] - X[y == k].mean(0) for k in classes if (y == k).sum() >= 2])
    return float((r ** 2).sum() / (len(r) * X.shape[1]))
