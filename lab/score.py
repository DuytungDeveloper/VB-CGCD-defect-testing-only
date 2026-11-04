"""Tier-1 evaluation (QD18; lab/README.vi.md sections 5 and 9): the output of one step scored with the true labels,
for scoring only, to screen methods fast; a ranking at tier 1 must be confirmed at tier 2 (full pipeline, exp.py runs).
Apart from the check commands of replay.py, this is the only place in lab/ that reads the true labels of the pool.

  python lab/score.py 09 --from <v2 run> [--ks 5 10 15 20 30]     through scripts/exp.py: writes metrics.json
  python lab/score.py 03                                             E03.02: linear probes on the GC10 features
"""
import argparse, importlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "lab")):
    if p not in sys.path:
        sys.path.insert(0, p)
from replay import load, snapshots, _fresh, predict_default, _jnp          # noqa: E402
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


# ---------------------------------------------------------------- loss ledger (QD24; notebooks 10-14)
def ledger(run):
    """Loss ledger (QD24; design fixed in 10-naming): for each image, whether it is still right after each step, from
    the snapshots of one run; true labels for scoring only. Columns are added by the notebooks 10-14.
    Sets: 'test1' (novel test set), 'pool' (pool of session 1; its old images get NaN at 09 and 10, a value at 12).
    Columns: '09' the image's cluster has the image's class as pool owner (plurality class of the cluster in the pool,
    old classes included, ties to the lower id, as score09): many-to-one, a diagnostic, not a usable naming;
    '10' the image's cluster carries the image's class under pi_test (the pipeline's naming, QD15);
    '11' (test1 only, counterfactual) the classifier with the pass-F Gaussians (params_f) puts the image in the slot
    of its named cluster, as if the pipeline stopped after pass F;
    '12' (pool only) the old/new filter of step 12 (keep): an old image is right when it is dropped, a novel image when
    it is kept and its cluster carries its class (as '10'), i.e. it trains the slot named after its own class in pass S.
    Returns {set: DataFrame with y (true id), novel (bool), cluster, and one float column per step: 1 right, 0 wrong,
    NaN not defined}."""
    import pandas as pd
    b = snapshots(run)
    _fresh(b, "12")
    o4, o9, o10 = b["04"]["out"], b["09"]["out"], b["10"]["out"]
    off, K = int(o4["offset"]), int(b["04"]["meta"]["args"]["increment"])
    yp, ids = np.asarray(o4["true_pool_y"]).astype(int), np.asarray(o9["ids"]).astype(int)
    y1 = np.asarray(o4["true_test1_y"]).astype(int)
    n = np.zeros((int(max(yp.max(), y1.max())) + 1, K), dtype=int)
    np.add.at(n, (yp, ids), 1)                                       # pool: class x cluster
    owner = n.argmax(0)
    name = np.array([off + o10["pi_test"][k] for k in range(K)])     # true id named by each cluster
    out = {}
    for s, y, cl in (("test1", y1, np.asarray(o9["cluster"].predict_raw(o4["test1_x"])).astype(int)), ("pool", yp, ids)):
        nov = y >= off
        df = pd.DataFrame({"y": y, "novel": nov, "cluster": cl})
        df["09"] = np.where(nov, (owner[cl] == y).astype(float), np.nan)
        df["10"] = np.where(nov, (name[cl] == y).astype(float), np.nan)
        out[s] = df
    import jax.numpy as jnp
    st = load(run, "11", check=False)                                # workbench: classifier of step 08, params_f
    predict_default(st["clf"])
    slot = np.asarray(st["clf"]._predict(jnp.asarray(st["z_test1"]), _jnp(st["params_f"]))[0])
    t, lut = out["test1"], np.asarray(o10["lut"])
    t["11"] = np.where(t["novel"], (slot == lut[t["y"].to_numpy()]).astype(float), np.nan)
    p, keep = out["pool"], np.asarray(b["12"]["out"]["keep"]).astype(bool)
    assert keep.shape == (len(p),), f"keep: one bool per pool image, got shape {keep.shape}"
    named = name[p["cluster"].to_numpy()] == p["y"].to_numpy()
    p["12"] = np.where(p["novel"], (keep & named).astype(float), (~keep).astype(float))
    return out


def ledger_summary(df, names, steps):
    """Per novel class of a ledger set: images right after each step, and the flows between consecutive steps
    (lost = right -> wrong, gained = wrong -> right); last row 'balanced' = mean recall of the novel classes per step."""
    import pandas as pd
    rows = {}
    for c, g in df[df["novel"]].groupby("y"):
        r = {"n": len(g)}
        for i, s in enumerate(steps):
            r[s] = int(g[s].sum())
            if i:
                p = steps[i - 1]
                r[f"lost_{p}_{s}"] = int(((g[p] == 1) & (g[s] == 0)).sum())
                r[f"gained_{p}_{s}"] = int(((g[p] == 0) & (g[s] == 1)).sum())
        rows[names[c]] = r
    t = pd.DataFrame(rows).T.astype(float)
    t.loc["balanced"] = {s: float((t[s] / t["n"]).mean()) for s in steps}
    return t


# ---------------------------------------------------------------- step 03: linear probe (E03.02)
SETS_03 = {"fz": ("gc10_fz_cls5first_img80pct", "dino-vitb16", "gc10_frozen_km_evhung_d64_head"),
           **{f"r{r}": (f"gc10_ft_cls5first_img80pct_r{r}", "dino-vitb16-sl", f"gc10_ft_cls5first_img80pct_r{r}_km_evhung_d64")
              for r in range(3)}}                                   # name: (feature run tag, model, phase-2 run group)
TEST_N_03 = {"rolled_pit": 6, "crease": 11, "inclusion": 43, "silk_spot": 124, "waist_folding": 30}   # E03.02/C3


def wilson(k, n, z=1.96):
    """95% Wilson score interval of k successes in n trials (Wilson 1927)."""
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z / d * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return c - h, c + h


def pipeline_recall_03(group, n_classes):
    """Per-class recall on the S1 test set of the main.py runs <group>_s0..4: the 'confusion stage 1' lines summed over the
    seeds (rows true ids, columns predicted ids; as scripts/cm_summary.py). Returns correct, total per class, number of runs."""
    import re
    pat = re.compile(r"confusion stage 1: rows=true cols=pred (\[\[.*\]\])")
    logs = sorted(f for f in (ROOT / "runs").glob(f"*__{group}_s[0-4]/console.log"))
    cms = [np.array(json.loads(pat.search(f.read_text(errors="replace")).group(1))) for f in logs]
    assert len(cms) == 5 and all(c.shape == (n_classes, n_classes) for c in cms), f"{group}: {len(cms)} confusion matrices"
    C = sum(cms)
    return C.diagonal(), C.sum(1), len(cms)


def tier1_03():
    """E03.02: per-class linear probes on the GC10 features of SETS_03, against the per-class recall of the pipeline.
    Train rows: roles S0, reserve, S1 (lab/data_io.py), true labels of all 10 classes (scoring only); test: 455 images.
    Probes (scikit-learn): lr = LogisticRegression(max_iter=5000), lr_bal = the same with class_weight="balanced",
    knn = 20 nearest neighbours by cosine distance, pca_lr_bal = lr_bal on PCA(64, random_state=0) fit on the S0 rows
    (num_dim of step 05). Checks: C1 feature hashes = config.json of the phase-2 runs; C3 novel test counts = TEST_N_03;
    C4 lr on all train rows of fz reproduces the logged probe accuracy 80.2% (LOG, 20260930_232359__gc10_extract_frozen)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    import data_io
    spec = importlib.util.spec_from_file_location("exp", ROOT / "scripts" / "exp.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    split = json.load(open(ROOT / data_io.SPLITS["gc10"]))
    names, n_old = split["classes"], split["n_base"]
    img = data_io.images("gc10")
    tr = img[img["part"] == "train"].sort_values("row")
    rows = tr["row"][tr["role"].isin(["S0", "reserve", "S1"])].to_numpy()
    s0 = tr["row"][tr["role"] == "S0"].to_numpy()
    out, checks = {}, {}
    for name, (tag, model, group) in SETS_03.items():
        F = next((ROOT / "features").glob(f"*__{tag}")) / model
        X, y = np.load(F / f"features-{model}.npy"), np.load(F / f"labels-{model}.npy").astype(int)
        Xt, yt = np.load(F / f"test_features-{model}.npy"), np.load(F / f"test_labels-{model}.npy").astype(int)
        assert np.array_equal(y[tr["row"]], tr["cls"]), f"{name}: labels file and data_io disagree"
        cfg = json.loads(next((ROOT / "runs").glob(f"*__{group}_s0")).joinpath("config.json").read_text())
        checks[f"C1_{name}"] = int(exp.hash_path(F) in cfg["inputs"].values())
        checks[f"C3_{name}"] = int(all(int((yt == names.index(c)).sum()) == n for c, n in TEST_N_03.items()))
        pca = PCA(64, random_state=0).fit(X[s0])
        probes = {"lr": (LogisticRegression(max_iter=5000), X, Xt),
                  "lr_bal": (LogisticRegression(max_iter=5000, class_weight="balanced"), X, Xt),
                  "knn": (KNeighborsClassifier(20, metric="cosine"), X, Xt),
                  "pca_lr_bal": (LogisticRegression(max_iter=5000, class_weight="balanced"), pca.transform(X), pca.transform(Xt))}
        corr_p, tot_p, n_runs = pipeline_recall_03(group, len(names))
        for pname, (clf, A, At) in probes.items():
            pred = clf.fit(A[rows], y[rows]).predict(At)
            rec = []
            for c in range(n_old, len(names)):
                k, n = int(((pred == c) & (yt == c)).sum()), int((yt == c).sum())
                lo, hi = wilson(k, n)
                rec.append(k / n)
                out.update({f"{pname}_{name}_{names[c]}_k": k, f"{pname}_{name}_{names[c]}_n": n,
                            f"{pname}_{name}_{names[c]}_lo": round(float(lo), 4), f"{pname}_{name}_{names[c]}_hi": round(float(hi), 4)})
            out[f"{pname}_{name}_bal_novel"] = round(float(np.mean(rec)), 4)
        rec = []
        for c in range(n_old, len(names)):
            r = corr_p[c] / tot_p[c]
            n = int((yt == c).sum())                                 # the same n test images in every seed
            lo, hi = wilson(round(r * n), n)
            rec.append(r)
            out.update({f"pipe_{name}_{names[c]}_recall": round(float(r), 4), f"pipe_{name}_{names[c]}_lo": round(float(lo), 4),
                        f"pipe_{name}_{names[c]}_hi": round(float(hi), 4)})
        out[f"pipe_{name}_bal_novel"] = round(float(np.mean(rec)), 4)
        out[f"pipe_{name}_runs"] = n_runs
        for c in ("rolled_pit", "crease", "inclusion", "silk_spot", "waist_folding"):   # decision rule of E03.02
            lo, p = out[f"lr_bal_{name}_{c}_lo"], out[f"lr_bal_{name}_{c}_k"] / out[f"lr_bal_{name}_{c}_n"]
            out[f"type_{name}_{c}"] = ("D" if lo > out[f"pipe_{name}_{c}_hi"] else "") + ("R" if p < 0.5 else "") or "U"
        if name == "fz":
            acc = float((LogisticRegression(max_iter=5000).fit(X, y).predict(Xt) == yt).mean())
            out["C4_fz_lr_all_rows_acc"] = round(100 * acc, 1)
            checks["C4"] = int(round(100 * acc, 1) == 80.2)
    out.update(checks)
    out["checks_ok"] = int(all(v == 1 for v in checks.values()))
    return out


def main(argv):
    p = argparse.ArgumentParser(prog="score.py", description="tier-1 scores of one step of a replay v2 run")
    p.add_argument("step", choices=["03", "09"])
    p.add_argument("--from", dest="run", help="09: run id or tag of a v2 run (runs/<run>/snap)")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30], help="numbers of clusters K'")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: features/, splits/)"
    assert o.step == "03" or o.run, "step 09 needs --from <run>"
    out = tier1_03() if o.step == "03" else tier1_09(o.run, o.ks)
    for key, v in out.items():
        print(f"{key:30s} {v}")
    if os.environ.get("EXP_RUN_DIR"):
        (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
