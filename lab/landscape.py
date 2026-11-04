"""k-means landscape of step 09 (QD26; notebooks 09-cluster and 09d-ownership): many single k-means++ inits on the pool
of one snapshot, without true labels, so that the solution of the pipeline (n_init 10) can be placed among them.

Label-free by construction: from the snapshots it reads only pool_x (step 04) and ids, inertia, rng_in (step 09); keys
true_* are dropped on load and never used. Camera and coil, if needed, come from file names in the notebooks.
Results are cached on this machine in runs/<run>/landscape/<key>.npz (git ignores *.npz); the key hashes the snapshot
(hash chain up to 09, bytes of pool_x), the code of the functions that make the cache, and the parameters.

  singles(run, n)      n single inits (KMeans(n_init=1)) from an own RNG stream, 1 OpenMP thread
  permuted(run, m)     m pipeline runs (n_init 10) from rng_in of step 09 with the rows permuted (r0 s4 in QD26)
  decompose(run)       the 10 inits of the pipeline's n_init 10, replayed one by one from rng_in
  keep_rule, canon, ari, modes, best_of_weights, good_turing, instability: analysis helpers (no RNG, no cache)
"""
import hashlib, inspect, io, json, time, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from replay import snapshots, _fresh, _run_dir    # noqa: E402  (lab/ and the repo root are on sys.path)


# ---------------------------------------------------------------- inputs of one snapshot, label-free
def pool(run):
    """Label-free inputs of step 09 of a run: pool_x (n, d) float, the pipeline's ids (n,), inertia, rng_in, K, and
    h09 (hash chain up to step 09). Refuses stale snapshots (replay._fresh)."""
    blobs = snapshots(run)
    _fresh(blobs, "09")
    o4 = {k: v for k, v in blobs["04"]["out"].items() if not k.startswith("true_")}
    o9 = {k: v for k, v in blobs["09"]["out"].items() if not k.startswith("true_")}
    x = np.asarray(o4["pool_x"])
    ids = np.asarray(o9["ids"]).astype(int)
    return dict(x=x, ids=ids, inertia=float(o9["inertia"]), rng_in=blobs["09"]["rng_in"],
                k=int(blobs["04"]["meta"]["args"]["increment"]), h09=blobs["09"]["hash"])


def canon(lab):
    """Cluster ids relabelled by order of first appearance (uint8): equal arrays <=> same partition."""
    lab = np.asarray(lab)
    _, first = np.unique(lab, return_index=True)
    order = np.argsort(first)
    mp = np.empty(lab.max() + 1, dtype=np.uint8)
    mp[np.unique(lab)[order]] = np.arange(len(order), dtype=np.uint8)
    return mp[lab]


def _same_clustering(a, b):
    """sklearn's _is_same_clustering: a and b are the same partition up to relabelling."""
    return np.array_equal(canon(a), canon(b))


def keep_rule(labels, inertias):
    """Index kept by KMeans.fit over inits in this order (scikit-learn 1.5.2): replace when inertia is strictly lower
    and the partition differs."""
    best = 0
    for i in range(1, len(inertias)):
        if inertias[i] < inertias[best] and not _same_clustering(labels[i], labels[best]):
            best = i
    return best


# ---------------------------------------------------------------- k-means runs (the only RNG use)
def _kmeans1(x, k, random_state):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=1, random_state=random_state).fit(x)
    return km.labels_, float(km.inertia_), int(km.n_iter_)


def _seed(run, what):
    return zlib.crc32(f"landscape|{what}|{run}".encode())        # own stream; E09.07 uses 907000 + ... (QD26)


def _singles_compute(x, k, n, seed):
    from threadpoolctl import threadpool_limits
    rs = np.random.RandomState(seed)
    L = np.empty((n, len(x)), dtype=np.uint8)
    J, it = np.empty(n), np.empty(n, dtype=np.int32)
    with threadpool_limits(1):
        for i in range(n):
            lab, J[i], it[i] = _kmeans1(x, k, rs)
            L[i] = canon(lab)
    return dict(L=L, J=J, n_iter=it)


def _permuted_compute(x, k, m, seed, rng_in):
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    prm = np.random.RandomState(seed)
    L = np.empty((m, len(x)), dtype=np.uint8)
    J = np.empty(m)
    with threadpool_limits(1):
        for i in range(m):
            p = prm.permutation(len(x))
            np.random.set_state(rng_in)                                # the same RNG state every time
            km = KMeans(n_clusters=k, n_init=10, random_state=None).fit(x[p])
            back = np.empty(len(x), dtype=int)
            back[p] = km.labels_                                       # labels in the snapshot's row order
            L[i], J[i] = canon(back), float(km.inertia_)
    return dict(L=L, J=J)


_CODE = (canon, _kmeans1, _seed, _singles_compute, _permuted_compute)


def _key(p, what, params):
    h = hashlib.sha256()
    h.update(json.dumps([p["h09"], what, params], sort_keys=True).encode())
    h.update(np.ascontiguousarray(p["x"]).tobytes())
    for f in _CODE:
        h.update(inspect.getsource(f).encode())
    return h.hexdigest()[:12]


def _cached(run, what, params, compute):
    """Load runs/<run>/landscape/<what>_<key>.npz, or compute it and write it. Returns (dict, meta)."""
    p = pool(run)
    path = _run_dir(run) / "landscape" / f"{what}_{_key(p, what, params)}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))
    import sklearn
    t0 = time.time()
    out = compute(p)
    meta = dict(run=run, what=what, params=params, seconds=round(time.time() - t0, 1), sklearn=sklearn.__version__,
                made=time.strftime("%Y-%m-%d %H:%M"))
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, meta=json.dumps(meta), **out)
    path.write_bytes(buf.getvalue())
    return out, meta


def singles(run, n=10000):
    """n single k-means++ inits on the pool of run: L (n, N) uint8 canonical labels, J (n,) inertia, n_iter (n,)."""
    return _cached(run, "singles", dict(n=n), lambda p: _singles_compute(p["x"], p["k"], n, _seed(run, "singles")))


def permuted(run, m=1000):
    """m pipeline runs (KMeans n_init 10) from rng_in of step 09 with the rows randomly permuted: L (m, N), J (m,)."""
    return _cached(run, "permuted", dict(m=m),
                   lambda p: _permuted_compute(p["x"], p["k"], m, _seed(run, "permuted"), p["rng_in"]))


def _job(a):
    fn, run, kw = a
    return run, fn(run, **kw)[1]


def run_all(runs, fn=singles, workers=None, **kw):
    """fn(run, **kw) for many runs in parallel processes (each one OpenMP thread); returns {run: meta}."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return dict(ex.map(_job, [(fn, r, kw) for r in runs]))


def decompose(run):
    """The pipeline's KMeans(n_init=10) replayed as 10 KMeans(n_init=1) from rng_in of step 09 (same global stream).
    Returns labels (10, N) canonical, inertias (10,), kept index (keep_rule), the RNG state after, and the result of
    KMeans(n_init=10) from the same state with its RNG state after, for the check."""
    from sklearn.cluster import KMeans
    p = pool(run)
    np.random.set_state(p["rng_in"])
    km = KMeans(n_clusters=p["k"], n_init=10, random_state=None).fit(p["x"])
    after10 = np.random.get_state()
    np.random.set_state(p["rng_in"])
    labs, js = [], []
    for _ in range(10):
        lab, j, _ = _kmeans1(p["x"], p["k"], None)
        labs.append(canon(lab)); js.append(j)
    after1 = np.random.get_state()
    return dict(labels=np.array(labs), inertias=np.array(js), kept=keep_rule(labs, js), after=after1,
                full_labels=canon(km.labels_), full_inertia=float(km.inertia_), full_after=after10,
                pipeline=canon(p["ids"]), pipeline_inertia=p["inertia"])


# ---------------------------------------------------------------- analysis helpers (deterministic)
def ari(a, b, k=None):
    """Adjusted Rand index (Hubert & Arabie 1985) of two label arrays with values < k, from the contingency table."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    k = k or int(max(a.max(), b.max())) + 1
    n = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype(np.float64)
    c2 = lambda v: (v * (v - 1) / 2).sum()
    s, sa, sb, t = c2(n), c2(n.sum(1)), c2(n.sum(0)), len(a) * (len(a) - 1) / 2
    e = sa * sb / t
    return float((s - e) / ((sa + sb) / 2 - e)) if (sa + sb) / 2 != e else 1.0


def modes(L, J, tau=0.95):
    """Solution modes: distinct partitions sorted by inertia; each joins the first leader with ARI >= tau, else leads a
    new mode. Returns mode (n,) per init, leaders (indices into L), and per mode: inertia of the leader, count."""
    keys = [r.tobytes() for r in L]
    first = {}
    for i, kb in enumerate(keys):
        first.setdefault(kb, i)
    distinct = sorted(first.values(), key=lambda i: (J[i], i))
    k = int(L.max()) + 1
    leaders, of = [], {}
    for i in distinct:
        for m, l in enumerate(leaders):
            if ari(L[i], L[l], k) >= tau:
                of[keys[i]] = m
                break
        else:
            of[keys[i]] = len(leaders)
            leaders.append(i)
    mode = np.array([of[kb] for kb in keys])
    return dict(mode=mode, leaders=np.array(leaders), J=J[leaders], count=np.bincount(mode, minlength=len(leaders)))


def best_of_weights(J, m):
    """Probability that best-of-m (m draws with replacement from the n singles) keeps single i: order statistics of
    the minimum, P(rank r) = ((n - r + 1)^m - (n - r)^m) / n^m, ties broken by index."""
    n = len(J)
    r = np.empty(n)
    r[np.lexsort((np.arange(n), J))] = np.arange(1, n + 1)
    return ((n - r + 1) / n) ** m - ((n - r) / n) ** m


def good_turing(mode):
    """Good-Turing missing mass n1 / n (Good 1953): share of probability in modes not seen yet."""
    c = np.bincount(mode)
    return float((c == 1).sum() / len(mode))


def instability(L, w=None, chunk=500):
    """Per-image instability d_i = mean_j 2 c_ij (1 - c_ij), c_ij = (weighted) share of inits with i, j together
    (co-association; Fred & Jain 2005). mean_i d_i = 1 - E[Rand index] of two independent draws (Rand 1971)."""
    n, N = L.shape
    w = np.full(n, 1 / n) if w is None else np.asarray(w) / np.sum(w)
    k = int(L.max()) + 1
    C = np.zeros((N, N))
    for s in range(0, n, chunk):
        O = np.eye(k, dtype=np.float32)[L[s:s + chunk]]                 # (c, N, k)
        Ow = O * np.sqrt(w[s:s + chunk])[:, None, None].astype(np.float32)
        H = Ow.transpose(1, 0, 2).reshape(N, -1)
        C += H @ H.T
    D = 2 * C * (1 - C)
    np.fill_diagonal(D, 0)
    return D.sum(1) / (N - 1)
