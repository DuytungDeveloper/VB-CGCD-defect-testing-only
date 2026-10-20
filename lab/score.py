"""Tier-1 evaluation (QD18; lab/README.vi.md sections 5 and 9): the output of one step scored with the true labels,
for scoring only, to screen methods fast; a ranking at tier 1 must be confirmed at tier 2 (full pipeline, exp.py runs).
Apart from the check commands of replay.py, this is the only place in lab/ that reads the true labels of the pool.

  python lab/score.py 09 --from <v2 run> [--ks 5 10 15 20 30]     through scripts/exp.py: writes metrics.json
"""
import argparse, importlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "lab")):
    if p not in sys.path:
        sys.path.insert(0, p)
from replay import load, snapshots, _fresh          # noqa: E402
from variants import v09a_overcluster               # noqa: E402


def score09(y, ids, names, n_old):
    """Tier-1 scores of a pool partition; y = true class ids (scoring only), ids = cluster ids. Flat dict.
    Same code as lab/pl_cluster.py score() (E09.04, E09.05), which it replaces.

    Ownership: class c owns cluster k if c is the plurality class in k, old classes included (E09.04);
    ties go to the lower class id. best(c) = max n_ck / n_c over the clusters c owns (0 if none).
    bal_acc_hungarian: one-to-one match novel class -> cluster on novel images (Kuhn 1955), mean recall.
    ami: adjusted mutual information (Vinh, Epps & Bailey 2010); purity: Manning, Raghavan & Schuetze 2008, 16.3.
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
        # E09.04 accompanying measures: total recall over the owned clusters, purity of the best owned cluster
        out[f"recall_owned_{names[c]}"] = float(n[c, own].sum() / n_c[c])
        k_best = own[np.argmax(n[c, own])] if len(own) else -1
        out[f"purity_best_{names[c]}"] = float(n[c, k_best] / n_k[k_best]) if len(own) else 0.0
    return out


def tier1_09(run, ks=(5, 10, 15, 20, 30)):
    """Tier 1 of step 09 on a v2 run, as E09.04 and E09.05 did with replay v1: for each K' in ks, v09a_overcluster with
    K' clusters from the RNG state at the entry of step 09 in the run's snapshots (common random numbers), scored
    with the true pool labels. Flat dict, keys k<K'>_<score> as the metrics.json of E09.05 (gc10_pl01b_*).
    Checks: C1 at K' = increment the partition equals the run's own step 09 (same function, same RNG state);
    C2 pool composition: 10 old images per base class, n_use of the split file per novel class (E09.04);
    C3 positive control (E09.04): waist_folding has best >= 0.5 at K' = increment."""
    st = load(run, "04", check=False)      # new_state first: imports torch, jax, main in the order of the v2 runs
    blobs = snapshots(run)
    _fresh(blobs, "09")                    # steps 04-09 of the run must match the code now
    a = st["args"]
    assert a.dataset == "gc10", "tier 1 of step 09 is written for GC10 (checks C2, C3 use its classes)"
    assert a.increment in ks, f"K' = {a.increment} is needed for checks C1 and C3"
    split = json.load(open(ROOT / importlib.import_module(f"dataloaders.{a.dataset}").SPLIT))
    names = [c.split("_")[0] for c in split["classes"]]
    y = np.asarray(st["true_pool_y"]).astype(int)
    view = {k: v for k, v in st.items() if not k.startswith("true_")}          # what a variant gets
    out = {"seed": a.seed}
    for k in ks:
        np.random.set_state(blobs["09"]["rng_in"])
        r = v09a_overcluster(view, k=k)
        sc = score09(y, r["sub_ids"], names, n_old=a.base)
        sc["inertia"] = r["sub_inertia"]
        out.update({f"k{k}_{key}": v for key, v in sc.items()})
        if k == a.increment:
            out["C1_ok"] = int(np.array_equal(r["sub_ids"], blobs["09"]["out"]["ids"]))
            out["C3_ok"] = int(sc["best_waist"] >= 0.5)
    expected = [10] * a.base + [split["n_use"][c] for c in split["classes"][a.base:]]
    out["C2_ok"] = int(np.bincount(y, minlength=len(names)).tolist() == expected)
    out["checks_ok"] = int(all(out.get(c) == 1 for c in ("C1_ok", "C2_ok", "C3_ok")))
    return out


def main(argv):
    p = argparse.ArgumentParser(prog="score.py", description="tier-1 scores of one step of a replay v2 run")
    p.add_argument("step", choices=["09"])
    p.add_argument("--from", dest="run", required=True, help="run id or tag of a v2 run (runs/<run>/snap)")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30], help="numbers of clusters K'")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: features/, splits/)"
    out = tier1_09(o.run, o.ks)
    for key, v in out.items():
        print(f"{key:30s} {v}")
    if os.environ.get("EXP_RUN_DIR"):
        (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
