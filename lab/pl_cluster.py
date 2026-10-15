"""Tier-1 evaluation of the clustering slot (M10): scores of a pool partition, used by
the notebooks in lab/ (tag nb-v1: notes/nb/12_cluster.ipynb) and by the PL experiments. True labels enter only here, for scoring;
they never reach a clustering method."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score


def score(y, ids, names, n_old):
    """Tier-1 scores of a pool partition; y = true class ids (scoring only), ids = cluster ids. Flat dict.

    Ownership: class c owns cluster k if c is the plurality class in k, old classes included (PL-E01);
    ties go to the lower class id. best(c) = max n_ck / n_c over the clusters c owns (0 if none).
    bal_acc_hungarian: one-to-one match novel class -> cluster on novel images (Kuhn 1955), mean recall.
    """
    y, ids = np.asarray(y).astype(int), np.asarray(ids).astype(int)
    assert ids.min() >= 0, "negative cluster id (noise label?): map it to a valid cluster id first"
    C, K = len(names), int(ids.max()) + 1
    n = np.zeros((C, K), dtype=int)
    np.add.at(n, (y, ids), 1)                                        # n[c, k] = images of class c in cluster k
    n_c, n_k = n.sum(1), n.sum(0)
    owner = n.argmax(0)                                              # plurality class of each cluster
    out = {"K": K, "ami": float(adjusted_mutual_info_score(y, ids)),
           "size_cv": float(n_k.std(ddof=1) / n_k.mean()),
           "purity_mean": float((n.max(0) / n_k).mean()),
           "old_in_novel_owned": int(n[:n_old][:, owner >= n_old].sum())}
    nov = np.arange(n_old, C)
    rows, cols = linear_sum_assignment(-n[nov])                      # one-to-one: novel class -> cluster
    out["bal_acc_hungarian"] = float((n[nov][rows, cols] / n_c[nov][rows]).mean())
    for c in nov:
        own = np.flatnonzero(owner == c)
        out[f"owned_{names[c]}"] = int(len(own))
        out[f"best_{names[c]}"] = float((n[c, own] / n_c[c]).max()) if len(own) else 0.0
        # PL-E01 accompanying measures: total recall over the owned clusters, purity of the best owned cluster
        out[f"recall_owned_{names[c]}"] = float(n[c, own].sum() / n_c[c])
        k_best = own[np.argmax(n[c, own])] if len(own) else -1
        out[f"purity_best_{names[c]}"] = float(n[c, k_best] / n_k[k_best]) if len(own) else 0.0
    return out


def _capture(seed, features):
    """Replay the pipeline up to M10 (lab/replay.py, stop="cluster") and capture the global NumPy RNG
    state at the entry of KMeansCluster.fit, from outside: replay.py is unchanged (tag nb-v1: 12_cluster, cell 8)."""
    from replay import replay
    from clustering.kmeans import KMeansCluster
    states, orig = [], KMeansCluster.fit

    def rec(self, *a, **kw):
        states.append(np.random.get_state())                # read-only: does not consume the RNG
        return orig(self, *a, **kw)

    KMeansCluster.fit = rec
    try:
        r = replay(seed, features=features, stop="cluster")
    finally:
        KMeansCluster.fit = orig
    assert len(states) == 1, len(states)
    return r, states[0]


def _ref_check(seed, tag, y, ids5, inertia5):
    """Check I1 against the pipeline run runs/*__<tag>_s<seed>: its contingency table and inertia if the run
    printed them (--print_cluster_xtab; PL-E01), else its cluster sizes (PL-E01b). None if there is no such run."""
    import re
    from pathlib import Path
    from sklearn.metrics.cluster import contingency_matrix
    from replay import _reference
    rd = sorted(Path("runs").glob(f"*__{tag}_s{seed}"))
    if not rd:
        return None
    ref_inertia, ref_table, _ = _reference(seed, tag=tag)
    if ref_table is not None:
        return bool(contingency_matrix(y, ids5).tolist() == ref_table
                    and abs(inertia5 - ref_inertia) / ref_inertia < 1e-6)
    text = (rd[0] / "console.log").read_text(errors="replace")
    i = text.index("(array([")
    arrays = re.findall(r"array\(\[([^\]]*)\]", text[i:text.index("]))", i) + 3])
    assert len(arrays) == 2, arrays
    ref_sizes = sorted((int(v) for v in arrays[1].replace(",", " ").split()), reverse=True)
    return bool(sorted(np.bincount(ids5).tolist(), reverse=True) == ref_sizes)


def main():
    """Tier-1 run of the clustering slot: k-means with K' clusters on the S1 pool, every K' from the
    pipeline's RNG state at M10 (PL-E01, PL-E01b). Writes a flat metrics.json when run through exp.py."""
    import argparse, json, os
    from pathlib import Path
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--features", default=None, help="features/<run>/<model>; default: first line of runs/inputs.txt")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30])
    ap.add_argument("--ref-tag", required=True, help="pipeline runs used by check I1, e.g. gc10_frozen_km_evhung_d64_head")
    a = ap.parse_args()
    assert 5 in a.ks, "K' = 5 is needed for checks I1 and I3"
    r, state = _capture(a.seed, a.features)
    from clustering.kmeans import KMeansCluster
    st, args = r["stages"][1], r["args"]
    X, raw = st["train"]._x, np.asarray(st["raw"])
    y = np.asarray(st["train"]._y).astype(int)
    names = [c.split("_")[0] for c in json.load(open("splits/gc10_split.json"))["classes"]]
    comp = dict(zip(names, np.bincount(y, minlength=len(names)).tolist()))
    expected = {**{nm: 10 for nm in names[:args.base]},
                "silk": 420, "inclusion": 135, "rolled": 17, "crease": 32, "waist": 94}
    out = {"seed": a.seed, "I2_ok": int(comp == expected)}
    for k in a.ks:
        np.random.set_state(state)                           # every K' starts from the pipeline's RNG state
        cl = KMeansCluster(num_classes=k, label_offset=args.base, n_init=args.kmeans_n_init)
        cl.fit(X)
        ids = np.asarray(cl.predict_raw(X))
        sc = score(y, ids, names, n_old=args.base)
        sc["inertia"] = float(cl.model.inertia_)
        out.update({f"k{k}_{key}": v for key, v in sc.items()})
        if k == 5:
            ref = _ref_check(a.seed, a.ref_tag, y, ids, sc["inertia"])
            out["I1_internal_ok"] = int(np.array_equal(ids, raw))
            out["I1_has_ref"] = int(ref is not None)
            out["I1_ok"] = int(out["I1_internal_ok"] == 1 and ref is not False)
            out["I3_ok"] = int(sc["best_waist"] >= 0.5)
    out["checks_ok"] = int(out["I1_ok"] and out["I2_ok"] and out["I3_ok"])
    for key, v in out.items():
        print(f"{key:30s} {v}")
    if os.environ.get("EXP_RUN_DIR"):
        (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root: clustering/, main.py
    main()
