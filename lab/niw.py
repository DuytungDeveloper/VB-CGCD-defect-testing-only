"""Normal-Inverse-Wishart (NIW) posterior of one Gaussian per class, and its two classifiers (E06.04, QD27).

Why: step 06 fits each class by SGD on the likelihood (D11) and hardly leaves its start (D23). With the conjugate
prior NIW(m0, k0, nu0, Psi0) the posterior of (mu, Sigma) of a class is known in closed form, so session 0 needs no
optimiser. For n rows with mean xb and scatter S (Murphy 2012, sec. 4.6.3; Gelman et al. 2013, BDA3 sec. 3.6):
    kn = k0 + n,  vn = nu0 + n,  mn = (k0 m0 + n xb) / kn,  Psi_n = Psi0 + S + (k0 n / kn) (xb - m0)(xb - m0)'
    posterior mean of Sigma:  Psi_n / (vn - d - 1)
    posterior predictive:     Student-t, vn - d + 1 degrees of freedom, location mn,
                              scale matrix Psi_n (kn + 1) / (kn (vn - d + 1))
Prior used here (pre-registration of E06.04): m0 = mean of all rows, k0 = K0, Psi0 = off * P with off = nu0 - d - 1
and P the pooled within-class covariance (scatter of all classes / (N - C)), so the prior mean of Sigma is P and a
large off gives LDA. off is the only free number; it is chosen on the labeled rows by leave-one-out (loo_curve).
The marginal likelihood is not used to choose it: with a small k0 the prior predictive of one row rewards a weak
prior without measuring fit (O'Hagan 1995; Berger & Pericchi 1996); log_evidence is here for checks only.

All functions take plain arrays, work in float64, read no file and draw no random number.
Self-test on generated data: python lab/niw.py
"""
import numpy as np
from scipy.special import gammaln, multigammaln

K0 = 0.01                                                   # kappa0: nearly flat prior on the class mean
OFFS = (2, 5, 10, 25, 50, 100, 250, 1000, 5000, 100000)     # grid of nu0 - d - 1 (pre-registration of E06.04)


def class_stats(z, y):
    """Per-class counts, means and scatter matrices. Returns dict(classes, n, mean, scatter, m0, N, d)."""
    z, y = np.asarray(z, np.float64), np.asarray(y).astype(int)
    classes = np.unique(y)
    n = np.array([(y == c).sum() for c in classes])
    mean = np.stack([z[y == c].mean(0) for c in classes])
    scatter = np.stack([(z[y == c] - m).T @ (z[y == c] - m) for c, m in zip(classes, mean)])
    return dict(classes=classes, n=n, mean=mean, scatter=scatter, m0=z.mean(0), N=len(z), d=z.shape[1])


def prior_cov(st, k, loo_class=False):
    """Pooled within-class covariance used as the prior mean of Sigma of class index k: all classes (default), or
    all classes but k (loo_class: the prior a class would get if it were new, as for novel classes in E09.08)."""
    C = len(st["n"])
    if loo_class:
        return (st["scatter"].sum(0) - st["scatter"][k]) / (st["N"] - st["n"][k] - (C - 1))
    return st["scatter"].sum(0) / (st["N"] - C)


def fit(z, y, off, k0=K0, loo_class=False):
    """NIW posterior of every class. off = nu0 - d - 1 > 0.
    Returns dict(classes, d, mn (C, d), kn (C,), vn (C,), chol (C, d, d) of Psi_n, logdet (C,) of Psi_n)."""
    st = class_stats(z, y)
    d, C = st["d"], len(st["n"])
    mn, kn, vn, chol, logdet = [], [], [], [], []
    for k in range(C):
        n, dm = st["n"][k], st["mean"][k] - st["m0"]
        psi = off * prior_cov(st, k, loo_class) + st["scatter"][k] + (k0 * n / (k0 + n)) * np.outer(dm, dm)
        L = np.linalg.cholesky(psi)
        mn.append((k0 * st["m0"] + n * st["mean"][k]) / (k0 + n)); kn.append(k0 + n); vn.append(d + 1 + off + n)
        chol.append(L); logdet.append(2 * np.log(np.diag(L)).sum())
    return dict(classes=st["classes"], d=d, mn=np.stack(mn), kn=np.array(kn), vn=np.array(vn),
                chol=np.stack(chol), logdet=np.array(logdet))


def _t_logpdf(q, logdet_psi, kn, vn, d):
    """Student-t predictive log density from q = (x - mn)' Psi_n^-1 (x - mn) and log det Psi_n."""
    v = vn - d + 1
    c = kn * v / (kn + 1)                                   # scale matrix = Psi_n / c
    return (gammaln((v + d) / 2) - gammaln(v / 2) - d / 2 * np.log(v * np.pi)
            - 0.5 * (logdet_psi - d * np.log(c)) - (v + d) / 2 * np.log1p(q * c / v))


def log_density(model, x, kind):
    """(len(x), C) log densities of the rows of x under every class.
    kind "gauss": Gaussian with mean mn and covariance Psi_n / (vn - d - 1) (plug-in of the posterior mean; A2).
    kind "t": the posterior predictive Student-t (A3)."""
    x, d = np.asarray(x, np.float64), model["d"]
    out = np.empty((len(x), len(model["classes"])))
    for k in range(out.shape[1]):
        q = (np.linalg.solve(model["chol"][k], (x - model["mn"][k]).T) ** 2).sum(0)
        if kind == "gauss":
            c = model["vn"][k] - d - 1
            out[:, k] = -0.5 * (model["logdet"][k] - d * np.log(c)) - 0.5 * q * c - d / 2 * np.log(2 * np.pi)
        elif kind == "t":
            out[:, k] = _t_logpdf(q, model["logdet"][k], model["kn"][k], model["vn"][k], d)
        else:
            raise ValueError(kind)
    return out


def predict(model, x, kind):
    """Class label of every row: argmax of log_density (equal class priors, as the pipeline)."""
    return model["classes"][log_density(model, x, kind).argmax(1)]


def loo_curve(z, y, offs=OFFS, k0=K0, loo_class=False):
    """Leave-one-out log predictive density of the labeled rows, summed, for every off in offs.
    For row i of class k everything is recomputed without that row: the class (n - 1 rows), the pooled covariance
    and m0; then the Student-t predictive of that posterior is evaluated at the row. Removing one row changes
    Psi_n by a rank-2 term, Psi' = A - a uu' + c ww' (u = x - xb_k, w = new class mean - new m0), so one Cholesky
    of A per class and off is enough (matrix inversion and determinant lemmas). Returns an array (len(offs),)."""
    z, y = np.asarray(z, np.float64), np.asarray(y).astype(int)
    st = class_stats(z, y)
    d, N, C = st["d"], st["N"], len(st["n"])
    assert st["n"].min() >= 2, "leave-one-out needs at least 2 rows per class"
    St = st["scatter"].sum(0)
    out = np.zeros(len(offs))
    for j, off in enumerate(offs):
        for k, cls in enumerate(st["classes"]):
            n, xb, x = st["n"][k], st["mean"][k], z[y == cls]
            if loo_class:
                A = off * (St - st["scatter"][k]) / (N - n - (C - 1)) + st["scatter"][k]
                a = n / (n - 1)
            else:
                A = off * St / (N - 1 - C) + st["scatter"][k]
                a = (off / (N - 1 - C) + 1) * n / (n - 1)
            u = x - xb
            m0 = st["m0"] - (x - st["m0"]) / (N - 1)                 # m0 without the row
            w = xb - u / (n - 1) - m0                                # class mean without the row, minus m0
            kn, vn, c = k0 + n - 1, d + 1 + off + n - 1, k0 * (n - 1) / (k0 + n - 1)
            r = x - (k0 * m0 + (n - 1) * (xb - u / (n - 1))) / kn    # row minus the posterior mean
            L = np.linalg.cholesky(A)
            lu, lw, lr = (np.linalg.solve(L, v.T) for v in (u, w, r))
            m11, m22, m12 = -1 / a + (lu * lu).sum(0), 1 / c + (lw * lw).sum(0), (lu * lw).sum(0)
            b1, b2 = (lu * lr).sum(0), (lw * lr).sum(0)
            det = m11 * m22 - m12 ** 2                               # 2 x 2 capacitance matrix; det(C) = -a c
            assert (det < 0).all(), "leave-one-out: Psi without the row is not positive definite"
            q = (lr * lr).sum(0) - (m22 * b1 ** 2 - 2 * m12 * b1 * b2 + m11 * b2 ** 2) / det
            logdet = 2 * np.log(np.diag(L)).sum() + np.log(a * c) + np.log(-det)
            out[j] += _t_logpdf(q, logdet, kn, vn, d).sum()
    return out


def choose(curve, offs=OFFS):
    """off with the highest curve value; ties go to the larger off."""
    curve = np.asarray(curve)
    return offs[len(curve) - 1 - int(np.argmax(curve[::-1]))]


def log_evidence(x, m0, k0, nu0, psi0):
    """log p(x) of the rows of one class under NIW(m0, k0, nu0, psi0) (Murphy 2007, eq. 266). For checks only."""
    x = np.asarray(x, np.float64)
    n, d = x.shape
    xb = x.mean(0)
    psin = psi0 + (x - xb).T @ (x - xb) + (k0 * n / (k0 + n)) * np.outer(xb - m0, xb - m0)
    ld = lambda a: np.linalg.slogdet(a)[1]
    return (-n * d / 2 * np.log(np.pi) + multigammaln((nu0 + n) / 2, d) - multigammaln(nu0 / 2, d)
            + nu0 / 2 * ld(psi0) - (nu0 + n) / 2 * ld(psin) + d / 2 * (np.log(k0) - np.log(k0 + n)))


def log_sequential(x, m0, k0, nu0, psi0):
    """Sum over the rows of the Student-t predictive of each row given the rows before it. Equals log_evidence."""
    x = np.asarray(x, np.float64)
    d, tot = x.shape[1], 0.0
    for i in range(len(x)):
        n = i
        xb = x[:i].mean(0) if i else np.zeros(d)
        S = (x[:i] - xb).T @ (x[:i] - xb) if i else np.zeros((d, d))
        psin = psi0 + S + (k0 * n / (k0 + n)) * np.outer(xb - m0, xb - m0)
        mn = (k0 * m0 + n * xb) / (k0 + n)
        L = np.linalg.cholesky(psin)
        q = (np.linalg.solve(L, x[i] - mn) ** 2).sum()
        tot += _t_logpdf(q, 2 * np.log(np.diag(L)).sum(), k0 + n, nu0 + n, d)
    return float(tot)


def _selftest():
    import time
    rng = np.random.default_rng(0)
    ok = True

    def report(name, good, detail):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'OK  ' if good else 'FAIL'} {name}: {detail}")

    # 1. the one-dimensional example of the notes: rows 1, 2, 3, 2 with m0 = 0, k0 = 1, nu0 = 3, Psi0 = 1
    x = np.array([[1.0], [2.0], [3.0], [2.0]])
    xb, S = x.mean(0), ((x - x.mean(0)) ** 2).sum()
    mn, psin = (0 + 4 * xb[0]) / 5, 1 + S + (1 * 4 / 5) * xb[0] ** 2
    report("1-D example", abs(mn - 1.6) < 1e-12 and abs(psin - 6.2) < 1e-12 and abs(psin / (7 - 2) - 1.24) < 1e-12,
           f"mn {mn:.4f}, Psi_n {psin:.4f}, posterior mean of the variance {psin / 5:.4f}")

    # 2. evidence = sequential predictive, also when n < d
    for d, n in ((3, 9), (12, 5), (40, 25)):
        x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + rng.normal(size=d)
        a = rng.normal(size=(d, d)); psi0 = a @ a.T + np.eye(d)
        m0, nu0 = rng.normal(size=d), d + 3.5
        e, s = log_evidence(x, m0, 0.3, nu0, psi0), log_sequential(x, m0, 0.3, nu0, psi0)
        report(f"evidence = sequential predictive (d {d}, n {n})", abs(e - s) <= 1e-8 * abs(e), f"{e:.6f} vs {s:.6f}")

    # 3. fast leave-one-out = direct recomputation, both priors
    C, d, n = 6, 10, 7
    mu = rng.normal(size=(C, d)) * 2
    z = np.concatenate([mu[c] + rng.normal(size=(n + c, d)) @ rng.normal(size=(d, d)) * 0.4 for c in range(C)])
    y = np.concatenate([np.full(n + c, 10 + c) for c in range(C)])
    for loo_class in (False, True):
        worst = 0.0
        for off in (2, 50, 5000):
            direct = 0.0
            for i in range(len(z)):
                keep = np.arange(len(z)) != i
                m = fit(z[keep], y[keep], off, loo_class=loo_class)
                k = int(np.where(m["classes"] == y[i])[0][0])
                direct += log_density(m, z[i:i + 1], "t")[0, k]
            fast = loo_curve(z, y, (off,), loo_class=loo_class)[0]
            worst = max(worst, abs(fast - direct) / abs(direct))
        report(f"fast leave-one-out = direct (prior from {'the other classes' if loo_class else 'all classes'})",
               worst <= 1e-8, f"largest relative difference {worst:.1e}")

    # 4. a very strong pooled prior is LDA: same labels as one pooled covariance with the class means
    zt = np.concatenate([mu[c] + rng.normal(size=(40, d)) @ rng.normal(size=(d, d)) * 0.4 for c in range(C)])
    st = class_stats(z, y)
    P = st["scatter"].sum(0) / (st["N"] - C)
    lda = st["classes"][np.stack([-(np.linalg.solve(np.linalg.cholesky(P), (zt - m).T) ** 2).sum(0)
                                  for m in st["mean"]], 1).argmax(1)]
    m = fit(z, y, 1e9)
    same = (predict(m, zt, "gauss") == lda).mean(), (predict(m, zt, "t") == lda).mean()
    report("off -> infinity is LDA (classes of different sizes)", min(same) >= 0.99, f"agreement {same[0]:.3f} (gauss), {same[1]:.3f} (t)")

    # 5. choose: ties go to the larger off
    report("choose", choose([1.0, 3.0, 3.0, 2.0], (2, 5, 10, 25)) == 10, "ties go to the larger off")

    # 6. time of the leave-one-out curve at the size of CUB-200 session 0
    C, d, n = 100, 384, 25
    z = rng.normal(size=(C * n, d)).astype(np.float32) + np.repeat(rng.normal(size=(C, d)), n, 0).astype(np.float32)
    y = np.repeat(np.arange(C), n)
    t0 = time.time()
    curve = loo_curve(z, y)
    report("leave-one-out curve at CUB size (100 classes x 25 rows, d 384, 10 offs)", np.isfinite(curve).all(),
           f"{time.time() - t0:.0f} s, chooses off = {choose(curve)}")
    print("niw self-test:", "all OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _selftest() else 1)
