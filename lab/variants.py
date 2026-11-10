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
