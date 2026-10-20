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
from clustering.kmeans import KMeansCluster


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
