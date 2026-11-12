"""Variants of the steps of lab/replay.py v2 (QD20, QD21; lab/README.vi.md sections 4 and 5).

A variant is a function vNNx_<name>(st) -> dict for step NN, run through scripts/exp.py with
  python lab/replay.py run --from <run> --start NN --at NN=vNNx_<name>[:key=value,...][+next function]
Rules:
- st comes without the true_* keys (the runner removes them): a variant never sees true labels.
- A variant returns new keys only; the runner refuses keys of other steps and true_* keys (_check_out).
- Call helpers of lab/ by imported name (from replay import f), never as replay.f, and name a component with
  f.__name__, never with a bare string: the hash chain follows names only, so code it cannot see could change
  without making the snapshots stale (checked 2026-10-20).
- At step 09 the last function must leave ids and cluster with ids in [0, increment) (_check_09); results with
  another number of clusters go to keys of their own.
- Log lines: never print "(array([" or a line starting with "|": scripts/extract_vbcgcd.py parses those.
"""
import numpy as np
from sklearn.cluster import KMeans
from clustering.kmeans import KMeansCluster
from landscape import canon
from niw import K0, loo_curve, choose, class_stats, prior_cov, log_evidence


def v09a_overcluster(st, k=20):
    """09a Over-clustering (E09.04, E09.05): k-means with k > increment clusters on the raw pool features, with the
    class and n_init of step 09 (main.Clustering_alg("kmeans") is KMeansCluster). Why: k-means tends to make clusters
    of similar size (uniform effect; Xiong, Wu & Chen 2009, IEEE TSMC-B), so with k = increment the rare novel classes
    are absorbed into the clusters of large ones; more clusters give them room (E09.05: alive at k = 20).
    In: pool_x, offset, args.kmeans_n_init. Out: sub_ids (0..k-1), sub_cluster (has predict_raw), sub_inertia.
    It never writes ids or cluster, so alone it cannot feed steps 10-14: score it at tier 1 (lab/score.py) or follow it
    with a merge back to increment clusters (v09b, E09.06).
    Draws the global RNG (random_state=None): run it first at step 09, so it starts from the step's RNG state (CRN)."""
    cl = KMeansCluster(num_classes=k, label_offset=st["offset"], n_init=st["args"].kmeans_n_init)
    cl.fit(st["pool_x"])
    ids = np.asarray(cl.predict_raw(st["pool_x"]))
    print(f"v09a_overcluster: k={k} sizes={np.bincount(ids, minlength=k).tolist()} inertia={cl.model.inertia_}")
    return dict(sub_ids=ids, sub_cluster=cl, sub_inertia=float(cl.model.inertia_))


def multi_init(x, k, m, random_state=None, marks=None):
    """m single k-means++ inits in a row (KMeans(n_init=1), other parameters default) from one RNG stream, kept by the
    rule of KMeans.fit (scikit-learn 1.5.2): the first init, then replace when the inertia is strictly lower and the
    partition differs. random_state None draws the global numpy RNG, as step 09 does; the first 10 inits are then the
    10 inits of the pipeline's n_init 10 from the same state (landscape.decompose; E09.07 check C1).
    marks: numbers of inits after which the kept solution is recorded (default (m,)); the largest must be m.
    Returns ({mark: (fitted KMeans, canonical labels, inertia)}, inertias of the m inits). No labels are read."""
    marks = tuple(marks or (m,))
    assert max(marks) == m and min(marks) >= 1, f"marks {marks}: between 1 and m = {m}, the largest equal to m"
    best, kept, inertias = None, {}, np.empty(m)
    for i in range(m):
        km = KMeans(n_clusters=k, n_init=1, random_state=random_state).fit(x)
        lab, inertias[i] = canon(km.labels_), km.inertia_
        if best is None or (km.inertia_ < best[2] and not np.array_equal(lab, best[1])):
            best = (km, lab, float(km.inertia_))
        if i + 1 in marks:
            kept[i + 1] = best
    return kept, inertias


def cluster_of(model, k, offset):
    """A KMeansCluster around an already fitted KMeans, so that steps 10-14 can call predict_raw on it."""
    cl = KMeansCluster(num_classes=k, label_offset=offset, n_init=1)
    cl.model = model
    return cl


def v09c_multi_init(st, m=1000):
    """09c Many k-means inits, lowest inertia kept (E09.07; Q09.12, QD26): step 09 with m single inits instead of
    n_init 10 (multi_init). Why: best-of-10 changes with the init (09-cluster); a deeper minimum is more stable, and
    E09.06 should stand on a stable k-means. The opposite may hold: the SSE objective prefers splitting a large class
    to isolating a rare one (uniform effect; Xiong, Wu & Chen 2009), so lower inertia need not mean better labels.
    In: pool_x, offset, args.increment. Out: ids, cluster, inertia, as step 09 (replaces s09_cluster).
    Draws the global RNG: run it first at step 09, so it starts from the step's RNG state (CRN)."""
    k = st["args"].increment
    kept, inertias = multi_init(st["pool_x"], k, m, marks=sorted({min(10, m), m}))
    km, _, j = kept[m]
    cl = cluster_of(km, k, st["offset"])
    ids = np.asarray(cl.predict_raw(st["pool_x"]))
    print(f"v09c_multi_init: m={m} sizes={np.bincount(ids, minlength=k).tolist()} inertia={j} "
          f"inertia_after_10={kept[min(10, m)][2]} min_single={inertias.min()}")
    return dict(ids=ids, cluster=cl, inertia=j)


CUB_T5 = dict(dataset="cub200", base=100, increment=20, num_classes=200, num_dim=384)   # upstream run.sh; QD27


def v04a_cub_t5(st):
    """04a Sessions of the CUB-200 t5 protocol, cut down to what steps 05-08 and the first online session need (QD27;
    E06.04). It calls main.Dataloader with load_mode t5, so the split is the one of dataloaders/cub200.py
    (StrictPerClassIncrementalLoader: 100 base classes, 25 labeled images each; 20 novel classes per session with 25
    images each, plus 5 images of every seen class). The loader builds all six sessions at once, so the global RNG
    leaves this step in the state main.py has before the PCA of S0.
    In: args (dataset cub200, base 100, increment 20; args.load_mode is not read). Out: the keys of step 04 for S0 and
    session 1: s0_x, s0_y, pool_x, test0_x, test1_x, test_old_x, test_all_x, offset, and the true_* labels.
    Not a variant for --at: it returns true_* keys, which the runner refuses from a variant (_check_out), and
    replay.new_state accepts load_mode t1 only. A runner calls it directly, in place of s04_loader, after setting the
    seeds in the order of main.py; the default s05-s08 then run on its output.
    Known properties of this split (notes of 2026-11-05): base classes are the first 100 names in alphabetical order;
    the pool of old images is not consumed between sessions (D16); the fine-tuned features have seen every training
    image of the base classes, also the old images of the pool (leak of the D9 kind)."""
    import main
    from types import SimpleNamespace
    a = st["args"]
    assert a.dataset == "cub200" and (a.base, a.increment) == (CUB_T5["base"], CUB_T5["increment"]), "CUB-200 t5: base 100, increment 20"
    tr, te, te_old, te_all = (list(it) for it in main.Dataloader(SimpleNamespace(**{**vars(a), "load_mode": "t5"})))
    assert len(tr) == 6 and len(te) == 6, f"t5: S0 and five online sessions, got {len(tr)} and {len(te)}"
    y = lambda d: np.asarray(d._y)
    out = dict(s0_x=tr[0]._x, s0_y=y(tr[0]), pool_x=tr[1]._x, test0_x=te[0]._x, test1_x=te[1]._x,
               test_old_x=te_old[1]._x, test_all_x=te_all[1]._x, offset=a.base,
               true_pool_y=y(tr[1]), true_test0_y=y(te[0]), true_test1_y=y(te[1]),
               true_test_old_y=y(te_old[1]), true_test_all_y=y(te_all[1]))
    n0 = np.bincount(out["s0_y"].astype(int), minlength=a.base)
    yp = out["true_pool_y"].astype(int)
    assert set(np.unique(out["s0_y"]).astype(int)) == set(range(a.base)) and n0.min() == n0.max(), "S0: every base class, same count"
    assert np.array_equal(out["true_test0_y"], out["true_test_old_y"]) and set(np.unique(out["true_test1_y"]).astype(int)) == \
        set(range(a.base, a.base + a.increment)), "test sets of S0 and of session 1"
    print(f"v04a_cub_t5: S0 {len(out['s0_y'])} images = {a.base} classes x {n0[0]}; pool 1 {len(yp)} images "
          f"({int((yp < a.base).sum())} old, {int((yp >= a.base).sum())} novel); test S0 {len(out['true_test0_y'])}, "
          f"novel test {len(out['true_test1_y'])}, all test {len(out['true_test_all_y'])}")
    return out


# ---------------------------------------------------------------- E09.06: merge the sub-clusters of 09a back to increment
class MergedCluster:
    """Cluster model of 09b for steps 10-14: nearest of the sub-cluster centres of 09a (predict_raw of the sub-cluster
    model, raw features), then the map sub-cluster -> merged cluster."""

    def __init__(self, sub_cluster, mapping):
        self.sub_cluster, self.mapping = sub_cluster, np.asarray(mapping, dtype=np.int64)
        self.num_classes = int(self.mapping.max()) + 1

    def predict_raw(self, features):
        return self.mapping[np.asarray(self.sub_cluster.predict_raw(features)).astype(int)]


def merge_prior(st):
    """NIW prior of every cluster in 09b (E09.06): m0 = mean of z_pool, k0 = niw.K0, nu0 = d + 1 + off,
    Psi0 = off * pooled within-class covariance of the labeled S0 classes (all classes), with off chosen by
    leave-one-out on S0 under the leave-one-class-out prior (niw.loo_curve(..., loo_class=True), niw.choose: the rule
    of off_lc in E06.04). Reads z_s0, s0_y (labels of S0, not of the pool) and z_pool. Returns a dict."""
    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    off = choose(loo_curve(z0, y0, loo_class=True))
    cs = class_stats(z0, y0)
    d = cs["d"]
    return dict(m0=np.asarray(st["z_pool"], np.float64).mean(0), k0=K0, nu0=d + 1 + off, psi0=off * prior_cov(cs, 0),
                off=int(off), d=d)


def agglomerate(n_sub, k, pair_cost):
    """Greedy merge of the sub-clusters 0..n_sub-1 down to k groups: at each step the pair of current groups with the
    lowest pair_cost(members_a, members_b) is merged (ties: the first pair in sorted order). Costs are cached by the
    member tuples, so a cached value can never be stale. Returns (list of member lists, steps), steps = list of
    (members_a, members_b, cost) in merge order."""
    groups, cache, steps = {i: (i,) for i in range(n_sub)}, {}, []
    while len(groups) > k:
        keys, best = sorted(groups), None
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                key = (groups[a], groups[b])
                if key not in cache:
                    cache[key] = float(pair_cost(groups[a], groups[b]))
                if best is None or cache[key] < best[0]:
                    best = (cache[key], a, b)
        c, a, b = best
        steps.append((groups[a], groups[b], c))
        groups[a] = tuple(sorted(groups[a] + groups.pop(b)))
    return list(groups.values()), steps


def merge_costs(st, sub_ids, prior):
    """Pair costs of the merge rules of E09.06 over member tuples of sub-clusters (lower = merge first):
    bf = minus the log Bayes factor of NIW marginal likelihoods (niw.log_evidence; Murphy 2007, eq. 266) in z_pool;
    ward = n_a n_b / (n_a + n_b) |mu_a - mu_b|^2 on the raw pool features (Ward 1963);
    bhatt = Bhattacharyya distance (paper Eq. 9) between the NIW posterior-mean Gaussians (mean m_n, covariance
    Psi_n / (nu_n - d - 1)) in z_pool; bhatt_mean = its mean term only. Returns {rule: pair_cost}."""
    z, x = np.asarray(st["z_pool"], np.float64), np.asarray(st["pool_x"], np.float64)
    sub_ids = np.asarray(sub_ids).astype(int)
    m0, k0, nu0, psi0, d = (prior[key] for key in ("m0", "k0", "nu0", "psi0", "d"))
    idx = lambda g: np.flatnonzero(np.isin(sub_ids, g))
    ev, gauss = {}, {}

    def evid(g):
        if g not in ev:
            ev[g] = log_evidence(z[idx(g)], m0, k0, nu0, psi0)
        return ev[g]

    def post(g):
        if g not in gauss:
            zz = z[idx(g)]
            n, xb = len(zz), zz.mean(0)
            psin = psi0 + (zz - xb).T @ (zz - xb) + (k0 * n / (k0 + n)) * np.outer(xb - m0, xb - m0)
            gauss[g] = ((k0 * m0 + n * xb) / (k0 + n), psin / (nu0 + n - d - 1))
        return gauss[g]

    def bhatt(a, b, mean_only):
        (ma, ca), (mb, cb) = post(a), post(b)
        L = np.linalg.cholesky((ca + cb) / 2)
        dm = np.linalg.solve(L, ma - mb)
        t = dm @ dm / 8
        if mean_only:
            return t
        ld = lambda c: 2 * np.log(np.diag(np.linalg.cholesky(c))).sum()
        return t + 0.5 * (2 * np.log(np.diag(L)).sum() - 0.5 * (ld(ca) + ld(cb)))

    def ward(a, b):
        ia, ib = idx(a), idx(b)
        na, nb = len(ia), len(ib)
        return na * nb / (na + nb) * ((x[ia].mean(0) - x[ib].mean(0)) ** 2).sum()

    return {"bf": lambda a, b: -(evid(tuple(sorted(a + b))) - evid(a) - evid(b)), "ward": ward,
            "bhatt": lambda a, b: bhatt(a, b, False), "bhatt_mean": lambda a, b: bhatt(a, b, True)}


def merged_ids(sub_ids, groups):
    """Map sub-cluster -> merged cluster, merged clusters numbered by first appearance in the pool."""
    sub_ids = np.asarray(sub_ids).astype(int)
    first = {i: np.flatnonzero(sub_ids == i)[0] if (sub_ids == i).any() else len(sub_ids) for i in range(sub_ids.max() + 1)}
    order = sorted(groups, key=lambda g: min(first[i] for i in g))
    mapping = np.empty(sub_ids.max() + 1, dtype=np.int64)
    for new, g in enumerate(order):
        mapping[list(g)] = new
    return mapping


def v09b_merge(st, rule="bf", prior=None):
    """09b Merge the sub-clusters of 09a back to increment clusters (E09.06, pre-registered at 61b9eb9): greedy
    agglomeration, at each step the pair with the largest log Bayes factor under the NIW prior of merge_prior (rule
    "bf"; "ward", "bhatt", "bhatt_mean" are the report-only rules, tier 1 only). Test images: nearest sub-cluster
    centre (raw features), then the map. No pool labels, no test data, no RNG.
    In: sub_ids, sub_cluster (09a), z_pool, pool_x, z_s0, s0_y. Out: ids, cluster (MergedCluster), merge_steps,
    merge_off; prior can be passed to skip the leave-one-out (same snapshot)."""
    prior = prior or merge_prior(st)
    sub = np.asarray(st["sub_ids"]).astype(int)
    groups, steps = agglomerate(sub.max() + 1, st["args"].increment, merge_costs(st, sub, prior)[rule])
    mapping = merged_ids(sub, groups)
    print(f"v09b_merge: rule={rule} off={prior['off']} sizes={np.bincount(mapping[sub]).tolist()}")
    return dict(ids=mapping[sub], cluster=MergedCluster(st["sub_cluster"], mapping), merge_steps=steps,
                merge_off=prior["off"])
