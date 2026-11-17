"""Tier-1 evaluation (QD18; lab/README.vi.md sections 5 and 9): the output of one step scored with the true labels,
for scoring only, to screen methods fast; a ranking at tier 1 must be confirmed at tier 2 (full pipeline, exp.py runs).
Apart from the check commands of replay.py, this is the only place in lab/ that reads the true labels of the pool.

  python lab/score.py 09 --from <v2 run> [--ks 5 10 15 20 30]     through scripts/exp.py: writes metrics.json
  python lab/score.py 03                                             E03.02: linear probes on the GC10 features
  python lab/score.py e0907 --from <v2 run> [--k-first 0 --k-last 10 --m 1000]   E09.07, one snapshot (exp.py)
  python lab/score.py e0907-summary                                  E09.07: decision table from the 15 runs
  python lab/score.py e0604 --from <gc10 v2 run> | --dataset cub200 --features F --ft R --seed S [--checks-only]
                                                                     E06.04, one chain (exp.py)
  python lab/score.py e0604-summary                                  E06.04: decision table from the 30 runs
  python lab/score.py e0906 --from <v2 run> [--k-first 0 --k-last 10]   E09.06, one snapshot (exp.py)
  python lab/score.py e0906-summary                                  E09.06: decision table from the 15 runs
  python lab/score.py e1101 --from <v2 run> --k-first 0 --k-last 3     E11.01, one snapshot (exp.py)
  python lab/score.py e1101-summary                                  E11.01: decision table from the 25 runs
  python lab/score.py cub-tb                                         CUB t5: T, B of QD28 from the default chains
  python lab/score.py e1401 --prefix cub_chain_ft[_l25] [--checks-only]   E14.01, the 15 chains of one feature set (exp.py)
  python lab/score.py e1401-summary                                  E14.01: table from the two runs
  python lab/score.py e1402 --prefix cub_chain_ft[_l25] [--checks-only]   E14.02, the 15 chains of one feature set (exp.py)
  python lab/score.py e1402-summary                                  E14.02: tables from the two runs
  python lab/score.py e1402b --prefix cub_chain_ft[_l25] [--checks-only]  E14.02 amendment: tables 4 and 5 (exp.py)
  python lab/score.py e1402b-summary                                 E14.02 amendment: tables from the two runs
  python lab/score.py e1403 --prefix cub_chain_ft[_l25] [--checks-only]   E14.03: old-slot competition, B, C, D (exp.py)
  python lab/score.py e1403-summary                                  E14.03: tables from the two runs
  python lab/score.py e1405 --prefix cub_chain_ft[_l25] [--checks-only]   E14.05: mean ladder Mk, Mc, M1, shared cov (exp.py)
  python lab/score.py e1405-summary                                  E14.05: tables from the two runs
  python lab/score.py e1406 --prefix cub_chain_ft[_l25] [--checks-only]   E14.06: NIW posterior for every slot (exp.py)
  python lab/score.py e1406-summary                                  E14.06: tables from the two runs
  python lab/score.py e1202 --prefix cub_chain_ft[_l25] [--checks-only]   E12.02: unlabeled old/new filter, model N (exp.py)
  python lab/score.py e1202-summary                                  E12.02: tables from the two runs
"""
import argparse, contextlib, importlib, importlib.util, io, json, os, re, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "lab")):
    if p not in sys.path:
        sys.path.insert(0, p)
from replay import load, snapshots, _fresh, _component, _jnp, chain, _check_09, _run_dir, s09_cluster   # noqa: E402
from variants import v09a_overcluster, v09b_merge, merge_prior, multi_init, cluster_of, merged_ids, stop_at_step   # noqa: E402
from niw import log_evidence                         # noqa: E402
from landscape import canon                         # noqa: E402


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
    """Loss ledger (QD24) of one run from its snapshots (fresh up to step 14): ledger_state after step 14."""
    _fresh(snapshots(run), "14")
    return ledger_state(load(run, "14", check=False))


def ledger_state(st):
    """Loss ledger (QD24; design fixed in 10-naming): for each image, whether it is still right after each step, from a
    state after step 14 (a run's snapshots through ledger(), or a state in memory, as E09.07 will build it);
    true labels for scoring only. Columns are added by the notebooks 10-14.
    Sets: 'test1' (novel test set); 'pool' (pool of session 1; its old images get NaN at 09 and 10, a value at 12);
    'test_old' (old test set of session 1, the images of the Old row of the S1 table; at t1 the S0 test set).
    Columns: '08' (test_old only) the S0 classifier (params0) puts the image in its own slot;
    '09' the image's cluster has the image's class as pool owner (plurality class of the cluster in the pool,
    old classes included, ties to the lower id, as score09): many-to-one, a diagnostic, not a usable naming;
    '10' the image's cluster carries the image's class under pi_test (the pipeline's naming, QD15);
    '11' (counterfactual) the classifier with the pass-F Gaussians (params_f) puts the image in the slot of its named
    cluster (test1) or in its own slot (test_old), as if the pipeline stopped after pass F;
    '12' (pool only) the old/new filter of step 12 (keep): an old image is right when it is dropped, a novel image when
    it is kept and its cluster carries its class (as '10'), i.e. it trains the slot named after its own class in pass S;
    '14' the final classifier (params1) puts the image in the slot of its class, as the S1 table of step 14 scores it.
    slot_<k> (test sets): the slot chosen by the classifier of column k, in the cluster space of the S1 table (lut):
    below offset an old class, offset + c the slot of cluster c; it splits the errors into old->new and old->old,
    new->old and new->new. The classifier is the run's own (st['predict'], as in steps 12 and 14).
    Checks: column 14 reproduces the Novel and Old rows of the S1 table (acc1), column 08 the S0 table (acc0).
    Returns {set: DataFrame with y (true id), novel (bool), cluster (test1, pool), one float column per step (1 right,
    0 wrong, NaN not defined), and the slot columns}."""
    import pandas as pd
    import jax.numpy as jnp
    off, K = int(st["offset"]), int(st["args"].increment)
    yp, ids = np.asarray(st["true_pool_y"]).astype(int), np.asarray(st["ids"]).astype(int)
    y1, yo = np.asarray(st["true_test1_y"]).astype(int), np.asarray(st["true_test_old_y"]).astype(int)
    assert (y1 >= off).all() and (yo < off).all(), "test1 holds the novel classes only, test_old the old ones only"
    n = np.zeros((int(max(yp.max(), y1.max())) + 1, K), dtype=int)
    np.add.at(n, (yp, ids), 1)                                       # pool: class x cluster
    owner = n.argmax(0)
    name = np.array([off + st["pi_test"][k] for k in range(K)])      # true id named by each cluster
    out = {}
    for s, y, cl in (("test1", y1, np.asarray(st["cluster"].predict_raw(st["test1_x"])).astype(int)), ("pool", yp, ids)):
        nov = y >= off
        df = pd.DataFrame({"y": y, "novel": nov, "cluster": cl})
        df["09"] = np.where(nov, (owner[cl] == y).astype(float), np.nan)
        df["10"] = np.where(nov, (name[cl] == y).astype(float), np.nan)
        out[s] = df
    clf, lut = st["clf"], np.asarray(st["lut"])
    _component(st["predict"])(clf)                                   # the run's classifier, as in steps 12 and 14
    slot = lambda key, z: np.asarray(clf._predict(jnp.asarray(z), _jnp(st[key]))[0]).astype(int)
    t, o = out["test1"], pd.DataFrame({"y": yo, "novel": False})
    zo = st["pca"].transform(st["test_old_x"])                       # as clf.test projects it (pre_processing)
    for k, key in (("08", "params0"), ("11", "params_f"), ("14", "params1")):
        o[f"slot_{k}"] = slot(key, zo)
        o[k] = (o[f"slot_{k}"] == lut[yo]).astype(float)
        if k != "08":
            t[f"slot_{k}"] = slot(key, st["z_test1"])
            t[k] = (t[f"slot_{k}"] == lut[y1]).astype(float)
    out["test_old"] = o
    p, keep = out["pool"], np.asarray(st["keep"]).astype(bool)
    assert keep.shape == (len(p),), f"keep: one bool per pool image, got shape {keep.shape}"
    named = name[p["cluster"].to_numpy()] == p["y"].to_numpy()
    p["12"] = np.where(p["novel"], (keep & named).astype(float), (~keep).astype(float))
    assert np.array_equal(st["test_old_x"], st["test0_x"]), "t1: the old test set of session 1 is the S0 test set"
    for col, df, acc, row in (("14", t, "acc1", "Novel"), ("14", o, "acc1", "Old"), ("08", o, "acc0", "All")):
        right = int(df[col].sum())
        assert round(st[acc][row] * len(df) / 100) == right, \
            f"column {col}: {right}/{len(df)} right, but {acc} {row} = {st[acc][row]}"
    return out


def ledger_summary(df, names, steps, novel=True):
    """Per class of a ledger set, the novel classes (novel=True) or the old ones (novel=False; set 'test_old'): images
    right after each step, and the flows between consecutive steps (lost = right -> wrong, gained = wrong -> right);
    last row 'balanced' = mean recall of those classes per step."""
    import pandas as pd
    rows = {}
    for c, g in df[df["novel"] == novel].groupby("y"):
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


# ---------------------------------------------------------------- E09.07: many k-means inits at step 09, tier 2
ESTOP_0907 = re.compile(r"Step (\d+):.*early_stop_flag = (True|False)")    # as scripts/extract_vbcgcd.py


def _tail_0907(st):
    """Steps 10-14 (default code) on a state that holds a partition (ids, cluster) after step 08; returns the log text
    of the passes F and S. No snapshots, no global RNG (RNG_STEPS), SVI from PRNGKey(0)."""
    logs = {}
    for a, b in (("10", "11"), ("12", "13"), ("14", "14")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            chain(st, start=a, upto=b)
        logs[b] = buf.getvalue()
    return logs


def _measures_0907(st, names, logs):
    """Flat measures of one finished chain (state after step 14) for E09.07, from the loss ledger (ledger_state):
    bal14 = balanced novel at tier 2 (QD25; mean recall of the novel classes on the novel test set, names of QD15),
    bal10 (M1), bal14_qd14 (M2: count Hungarian on slot_14 itself, scored on the same images), stop_S (M3: the step of
    the early_stop_flag line of pass S; n_epochs when the pass ran to the end), S1_Old, S1_New, S1_All, the old test
    images right at 08 and 14 with the flows (C4), balanced accuracy over all classes, images right per novel class."""
    led = ledger_state(st)
    t, o = led["test1"], led["test_old"]
    off, K = int(st["offset"]), int(st["args"].increment)
    rec = t.groupby("y")["14"].mean()
    m = {"bal14": float(rec.mean()), "bal10": float(t.groupby("y")["10"].mean().mean()),
         "S1_Old": 100 * float(o["14"].mean()), "S1_New": 100 * float(t["14"].mean()), "S1_All": float(st["acc1"]["All"]),
         "bal_all": float(np.concatenate([o.groupby("y")["14"].mean().to_numpy(), rec.to_numpy()]).mean()),
         "new14": int(t["14"].sum()), "old08": int(o["08"].sum()), "old14": int(o["14"].sum()),
         "lost08_14": int(((o["08"] == 1) & (o["14"] == 0)).sum()), "gain08_14": int(((o["08"] == 0) & (o["14"] == 1)).sum()),
         "kept": int(np.asarray(st["keep"]).sum())}
    y, s = t["y"].to_numpy() - off, t["slot_14"].to_numpy()
    n = np.zeros((K, K), dtype=int)
    np.add.at(n, (y[s >= off], s[s >= off] - off), 1)                # novel class x novel slot; old slots count as wrong
    rows, cols = linear_sum_assignment(-n)                           # QD14: one-to-one by the number of images
    m["bal14_qd14"] = float((n[rows, cols] / np.bincount(y, minlength=K)[rows]).mean())
    for c, g in t.groupby("y"):
        m[f"right14_{names[c]}"] = int(g["14"].sum())
    for k, key in (("11", "stop_F"), ("13", "stop_S")):
        hit = ESTOP_0907.findall(logs[k])
        m[key] = int(hit[-1][0]) if hit else int(st["args"].n_epochs)
    return m


def tier2_e0907(run, k_first=0, k_last=10, m=1000):
    """E09.07 (notes/pl/experiments.vi.md, parts 1 and 2) on one snapshot run gc10_v2_ft_r<r>_s<N>: draws k_first..k_last
    of m single k-means inits each (variants.multi_init). Draw 0 uses the global RNG set to rng_in of step 09; draw
    k >= 1 uses RandomState(907000 + 1000 r + 100 N + k). In each draw the default branch is the solution kept after
    the first 10 inits (the pipeline's n_init 10) and the variant the one kept after all m; the solution after 100 is
    scored at tier 1 only. Every distinct partition (pool ids and test ids, clusters renumbered by first appearance)
    runs steps 10-14 once, in this process, from the state after step 08.
    Keys: d<k>_def_*, d<k>_var_* (_measures_0907), d<k>_changed, d<k>_J<q>, d<k>_t1_<q>_* (score09 of the solution
    kept after q inits), p<j>_* per partition run, ref_* (the snapshot's own chain), D_* and s_* (mean and SD, ddof 1,
    over the draws of variant minus default).
    Checks: C1 (only when draw 0 is run) the default branch of draw 0 gives the snapshot's ids and test ids, its inertia
    within 1e-6 (relative), and the measures of the snapshot's chain (integers equal, reals within 1e-6);
    C2 inertia of the variant <= inertia of the default in every draw; C3 _check_09 on every partition run;
    C4 no old test image goes from wrong at 08 to right at 14, and right at 14 = right at 08 - lost."""
    assert k_first >= 0 and k_last >= k_first and m >= 10, "draws k_first..k_last, m >= 10"
    st = load(run, "08", check=False)      # new_state first: imports torch, jax, main in the order of the v2 runs
    blobs = snapshots(run)
    _fresh(blobs, "14")                    # the snapshot's chain is the reference: it must match the code now
    import jax
    device = jax.default_backend()
    assert device == "gpu" or os.environ.get("REPLAY_ALLOW_CPU") == "1", f"JAX runs on {device}, not on the GPU (A1)"
    assert {b["meta"].get("device", "gpu") for b in blobs.values()} == {device}, "snapshots ran on another device"
    a = st["args"]
    assert a.dataset == "gc10", "E09.07 is pre-registered on GC10"
    tag = re.search(r"_r(\d+)_s(\d+)$", _run_dir(run).name)
    r, N = int(tag.group(1)), int(tag.group(2))
    assert N == a.seed, f"run name says seed {N}, the snapshot has seed {a.seed}"
    names = [c.split("_")[0] for c in json.load(open(ROOT / importlib.import_module(f"dataloaders.{a.dataset}").SPLIT))["classes"]]
    K, off, x, xt = int(a.increment), int(st["offset"]), st["pool_x"], st["test1_x"]
    yp = np.asarray(st["true_pool_y"]).astype(int)                   # scoring only (tier 1)
    marks = sorted({q for q in (10, 100, m) if q <= m})
    parts, index = [], {}

    def part(km, inertia):
        """Index of the partition of a fitted KMeans among the distinct ones seen so far."""
        cl = cluster_of(km, K, off)
        ids, tid = np.asarray(cl.predict_raw(x)), np.asarray(cl.predict_raw(xt))
        key = canon(np.concatenate([ids, tid])).tobytes()
        if key not in index:
            index[key] = len(parts)
            parts.append(dict(cl=cl, ids=ids, tid=tid, inertia=inertia, t1=score09(yp, ids, names, n_old=a.base)))
        return index[key]

    out = {"seed": N, "ft_seed": r, "k_first": k_first, "k_last": k_last, "m": m}
    draws, t0 = {}, time.time()
    for k in range(k_first, k_last + 1):
        if k == 0:
            np.random.set_state(blobs["09"]["rng_in"])
        kept, J = multi_init(x, K, m, None if k == 0 else np.random.RandomState(907000 + 1000 * r + 100 * N + k), marks)
        draws[k] = {q: part(kept[q][0], kept[q][2]) for q in marks}
        out[f"d{k}_Jmin"] = float(J.min())
        for q in marks:
            t1 = parts[draws[k][q]]["t1"]
            out[f"d{k}_J{q}"] = kept[q][2]
            out[f"d{k}_t1_{q}_bal"] = t1["bal_acc_hungarian"]
            out.update({f"d{k}_t1_{q}_owned_{c}": t1[f"owned_{c}"] for c in names[a.base:]})
    print(f"e0907: {len(draws)} draws of {m} inits in {time.time() - t0:.0f} s, {len(parts)} distinct partitions", flush=True)
    base, res = dict(st), {}
    for j in sorted({d[q] for d in draws.values() for q in (10, m)}):
        s2, p, t0 = dict(base), parts[j], time.time()
        s2.update(ids=p["ids"], cluster=p["cl"], inertia=p["inertia"])
        _check_09(s2)                                                # C3
        res[j] = _measures_0907(s2, names, _tail_0907(s2))
        res[j]["_keep"], res[j]["_acc1"] = np.asarray(s2["keep"]), dict(s2["acc1"])
        out.update({f"p{j}_inertia": p["inertia"], f"p{j}_bal14": res[j]["bal14"], f"p{j}_S1_Old": res[j]["S1_Old"]})
        print(f"e0907: partition {j} steps 10-14 in {time.time() - t0:.0f} s", flush=True)
    for k, d in draws.items():
        for q, name in ((10, "def"), (m, "var")):
            out.update({f"d{k}_{name}_{key}": v for key, v in res[d[q]].items() if not key.startswith("_")})
            out[f"d{k}_{name}_part"] = d[q]
        out[f"d{k}_changed"] = int(d[10] != d[m])
    out["n_partitions_run"], out["n_changed"] = len(res), sum(out[f"d{k}_changed"] for k in draws)
    for key in ("bal14", "S1_Old", "bal10", "bal14_qd14", "stop_S"):
        diff = np.array([out[f"d{k}_var_{key}"] - out[f"d{k}_def_{key}"] for k in draws], dtype=float)
        out[f"D_{key}"] = float(diff.mean())
        out[f"s_{key}"] = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    checks = {"C2_ok": int(all(out[f"d{k}_J{m}"] <= out[f"d{k}_J10"] for k in draws)), "C3_ok": 1,
              "C4_ok": int(all(v["gain08_14"] == 0 and v["old14"] == v["old08"] - v["lost08_14"] for v in res.values()))}
    if 0 in draws:
        ref_st, o9, j0 = dict(base), blobs["09"]["out"], draws[0][10]
        for kk in ("09", "10", "11", "12", "13", "14"):
            ref_st.update(blobs[kk]["out"])
        ref = _measures_0907(ref_st, names, {"11": blobs["11"]["log"], "13": blobs["13"]["log"]})
        out.update({f"ref_{key}": v for key, v in ref.items()})
        close = lambda u, v: u == v if isinstance(v, int) else abs(u - v) <= 1e-6 * max(1.0, abs(v))
        same = {"ids": np.array_equal(parts[j0]["ids"], np.asarray(o9["ids"])),
                "test_ids": np.array_equal(parts[j0]["tid"], np.asarray(o9["cluster"].predict_raw(xt))),
                "inertia": abs(parts[j0]["inertia"] - o9["inertia"]) <= 1e-6 * o9["inertia"],
                "keep": np.array_equal(res[j0]["_keep"], np.asarray(blobs["12"]["out"]["keep"])),
                "acc1": all(close(res[j0]["_acc1"][key], float(v)) for key, v in blobs["14"]["out"]["acc1"].items()),
                "measures": all(close(res[j0][key], v) for key, v in ref.items())}
        print("e0907: C1", {key: bool(v) for key, v in same.items()}, flush=True)
        checks["C1_ok"] = int(all(same.values()))
    out.update(checks)
    out["checks_ok"] = int(all(v == 1 for v in checks.values()))
    return out


def e0907_summary(T=0.0273, B=0.615):
    """Decision table of E09.07 from the 15 runs gc10_e09_07_r<r>_s<N> (and gc10_e09_07x_r<r>_s<N>, draws 11-20, when
    they exist): D_i per snapshot, Delta_r per FT seed, Delta, SE_MC, the rule of part 1 with T and B of part 2 (pooled
    SD; the defaults are the numbers fixed there) and, for the report, with the part-1 formula (mean of the SDs); T and
    B recomputed from the ref_* keys as a check; M1-M3; the k = 0 result; Spearman of inertia against bal14 over the
    partitions run in each snapshot. Refuses to read results unless every run has checks_ok = 1."""
    import pandas as pd
    from scipy import stats
    keys, rows = ("bal14", "S1_Old", "bal10", "bal14_qd14", "stop_S", "S1_New", "S1_All", "bal_all"), []
    for r in range(3):
        for N in range(5):
            ms = []
            for tag in (f"gc10_e09_07_r{r}_s{N}", f"gc10_e09_07x_r{r}_s{N}"):
                f = sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json"))
                assert len(f) == 1 or (len(f) == 0 and "x" in tag.split("_")[2]), f"{tag}: {len(f)} runs with metrics"
                ms += [json.loads(p.read_text()) for p in f]
            assert all(m_["checks_ok"] == 1 for m_ in ms), f"r{r} s{N}: checks failed, results are not read"
            assert "C1_ok" in ms[0] and ms[0]["k_first"] == 0, f"r{r} s{N}: the first run must hold draw 0 (C1)"
            ks = [(m_, k) for m_ in ms for k in range(m_["k_first"], m_["k_last"] + 1)]
            assert len({k for _, k in ks}) == len(ks), f"r{r} s{N}: a draw appears twice"
            row = {"snap": f"r{r}_s{N}", "ft": f"r{r}", "draws": len(ks), "changed": sum(m_[f"d{k}_changed"] for m_, k in ks)}
            for key in keys:
                d = np.array([m_[f"d{k}_var_{key}"] - m_[f"d{k}_def_{key}"] for m_, k in ks], dtype=float)
                row[f"D_{key}"], row[f"s_{key}"], row[f"k0_{key}"] = d.mean(), d.std(ddof=1), d[0]
            row["ref_bal14"], row["ref_S1_Old"] = ms[0]["ref_bal14"], ms[0]["ref_S1_Old"]
            for q in (10, 100, 1000):
                v = [m_[f"d{k}_t1_{q}_bal"] for m_, k in ks if f"d{k}_t1_{q}_bal" in m_]
                row[f"t1_{q}"] = float(np.mean(v)) if v else float("nan")
            pj = sorted({int(key[1:].split("_")[0]) for key in ms[0] if re.match(r"p\d+_inertia$", key)})
            J, b = [ms[0][f"p{j}_inertia"] for j in pj], [ms[0][f"p{j}_bal14"] for j in pj]
            row["n_parts"] = len(pj)
            row["spearman"] = float(stats.spearmanr(J, b)[0]) if len(set(J)) > 1 and len(set(b)) > 1 else float("nan")
            rows.append(row)
    df = pd.DataFrame(rows).set_index("snap")
    assert df["draws"].nunique() == 1, "every snapshot needs the same number of draws"
    n_draws, g = int(df["draws"].iloc[0]), df.groupby("ft")
    pd.set_option("display.width", 250)
    print(df[["draws", "changed", "n_parts", "D_bal14", "s_bal14", "D_S1_Old", "s_S1_Old", "D_bal10", "D_bal14_qd14",
              "D_stop_S", "k0_bal14", "k0_S1_Old", "t1_10", "t1_100", "t1_1000", "spearman"]].round(4).to_string())
    by = g[[f"D_{key}" for key in keys] + [f"k0_{key}" for key in keys] + ["t1_10", "t1_100", "t1_1000", "spearman"]].mean()
    print("\nmean by FT seed (Delta_r in the D_ columns):")
    print(by.round(4).to_string())
    delta, delta_old = float(by["D_bal14"].mean()), float(by["D_S1_Old"].mean())
    n_pos = int((by["D_bal14"] > 0).sum())
    se = {key: float(np.sqrt((df[f"s_{key}"] ** 2).sum() / n_draws) / len(df)) for key in ("bal14", "S1_Old")}
    thr = {}
    for name, col in (("T", "ref_bal14"), ("B", "ref_S1_Old")):
        s = g[col].std(ddof=1)
        thr[name] = (float(np.sqrt((s ** 2).mean())), float(s.mean()))          # pooled (equal n), mean of the SDs
    print(f"\nthresholds recomputed from the default chains: T pooled {thr['T'][0]:.5f}, part-1 formula {thr['T'][1]:.5f}; "
          f"B pooled {thr['B'][0]:.5f}, part-1 formula {thr['B'][1]:.5f}")
    print("pooled values round to the pre-registered T, B:", round(thr["T"][0], 4) == T, round(thr["B"][0], 3) == B)

    def rule(t_, b_):
        if delta <= 0 or delta_old < -b_:
            return "reject"
        return "accept" if delta >= t_ and n_pos >= 2 and delta_old >= -b_ else "not large enough"
    near = [name for name, v, edge, s in (("Delta vs T", delta, T, se["bal14"]), ("Delta vs 0", delta, 0.0, se["bal14"]),
                                          ("Delta_Old vs -B", delta_old, -B, se["S1_Old"])) if abs(v - edge) < 2 * s]
    print(f"\nDelta = {delta:.5f} (T = {T}), Delta_r > 0 in {n_pos}/3 FT seeds, Delta_Old = {delta_old:.4f} points (B = {B}), "
          f"SE_MC = {se['bal14']:.5f} (Delta), {se['S1_Old']:.4f} (Delta_Old), {n_draws} draws")
    print(f"decision (part 2 thresholds): {rule(T, B)} | with the part-1 formula ({thr['T'][1]:.4f}, {thr['B'][1]:.3f}): "
          f"{rule(thr['T'][1], thr['B'][1])}")
    print("within 2 SE_MC of a boundary:", near or "none",
          "-> 10 more draws (k = 11-20), once" if near and n_draws == 11 else "")
    print(f"M1 = {by['D_bal10'].mean():.5f} | M2 = {by['D_bal14_qd14'].mean():.5f} | M3 = {by['D_stop_S'].mean():.2f} steps | "
          f"k = 0 only: Delta {by['k0_bal14'].mean():.5f}, Delta_Old {by['k0_S1_Old'].mean():.4f} | "
          f"variant changed the partition in {int(df['changed'].sum())}/{n_draws * len(df)} draws")
    return df


def _gauss_logpdf64(z, means, covs):
    """(len(z), C) Gaussian log densities in float64, without the constant -d/2 log(2 pi) (same for every class)."""
    out = np.empty((len(z), len(means)))
    for k, (m, c) in enumerate(zip(means, covs)):
        L = np.linalg.cholesky(np.asarray(c, np.float64))
        out[:, k] = -np.log(np.diag(L)).sum() - 0.5 * (np.linalg.solve(L, (z - m).T) ** 2).sum(0)
    return out


def _branch_scores(name, pred, lp_true, yt, base):
    """acc, balanced accuracy, lowest recall over the base classes; mean test log density of the true class."""
    rec = np.array([(pred[yt == c] == c).mean() for c in range(base)])
    out = {f"acc_{name}": 100.0 * float((pred == yt).mean()), f"bal_{name}": 100.0 * float(rec.mean()),
           f"rmin_{name}": 100.0 * float(rec.min())}
    if lp_true is not None:
        out[f"logd_{name}"] = float(np.mean(lp_true))
    return out, rec


def tier2_e0604(dataset, run=None, features=None, ft=None, seed=None, checks_only=False):
    """E06.04 (notes/pl/experiments.vi.md, pre-registration 2026-11-11) on one chain: session 0 by four branches on the
    same PCA features and test set. A0 default (params0 of step 06, argmax over all num_classes slots as mngmm);
    A1 E06.03 (mngmm._fit_closed: sample mean + Ledoit-Wolf, as --estimator closed); A2 NIW, Gaussian plug-in of the
    posterior mean; A3 NIW, Student-t predictive (lab/niw.py). NIW prior: m0 = mean of S0, kappa0 = niw.K0,
    Psi0 = off * pooled within-class covariance of all base classes; off = nu0 - d - 1 chosen on the S0 rows by
    leave-one-out (niw.loo_curve, niw.choose). Test labels and pool labels are read for scoring only.
    gc10: state after step 08 of the snapshot run gc10_v2_ft_r<r>_s<N>. cub200: v04a_cub_t5, then the default steps
    05-08, seeded as main.py; the state after 08 is saved to <run dir>/state08.pkl (git ignores *.pkl).
    Keys: acc_, bal_, rmin_ (base classes), logd_ (mean log density of the true class) per branch A0..A3 and A0base
    (A0 with argmax over the base slots only); A0_unseen (A0 test images given to a slot not yet seen); auroc_<branch>
    (pool of session 1, old = positive, score = highest log density over the base classes); off, loo_<off>;
    off_lc, loolc_<off>, acc_A2lc, acc_A3lc (prior from the other classes); d23_* (A0 parameters against the sample
    statistics); C1_ok, C3_ok, C2_ok (C2 only in the chain r0 s0), checks_ok. Per-class recalls go to recall.json.
    Checks: C1 A0 equals acc0 of the snapshot (gc10) or S0_All of the run cub_base_ft_r<r>_s<N> (cub200), as numbers
    of correct images; C2 log evidence = sequential predictive for one class (relative 1e-8) and fast leave-one-out =
    direct recomputation over all S0 rows at the chosen off (relative 1e-8); C3 A1 labels equal an independent float64
    recomputation (closed_form_params, Cholesky log densities over all slots). checks_only: compute and print the
    checks only, return no result."""
    import niw
    from sklearn.metrics import roc_auc_score
    from replay import new_state, make_args, _git
    from variants import v04a_cub_t5, CUB_T5
    t0 = time.time()
    run_dir = Path(os.environ["EXP_RUN_DIR"]) if os.environ.get("EXP_RUN_DIR") else None
    if run_dir is not None and _git()["dirty"]:
        raise SystemExit("e0604: commit the code first (same dirty rule as scripts/exp.py)")
    if dataset == "gc10":
        assert run and features is None and ft is None and seed is None, "gc10: --from <snapshot run> only"
        st = load(run, "08", check=False)      # new_state first: imports torch, jax, main in the order of the v2 runs
        blobs = snapshots(run)
        _fresh(blobs, "08")
        tag = re.search(r"_r(\d+)_s(\d+)$", _run_dir(run).name)
        r, N = int(tag.group(1)), int(tag.group(2))
        assert N == st["args"].seed and st["args"].dataset == "gc10", "run name, seed and dataset must agree"
    else:
        assert features and ft is not None and seed is not None and not run, "cub200: --features, --ft and --seed"
        r, N = ft, seed
        st = new_state(make_args(features, seed, CUB_T5))
        import torch, jax
        torch.manual_seed(seed); np.random.seed(seed); jax.random.PRNGKey(seed)     # main.py order
        st.update(v04a_cub_t5(st))
        chain(st, None, start="05", upto="08")
    import jax
    import jax.numpy as jnp
    device = jax.default_backend()
    assert device == "gpu" or os.environ.get("REPLAY_ALLOW_CPU") == "1", f"JAX runs on {device}, not on the GPU (A1)"
    if run_dir is not None:
        m_ = re.search(r"_r(\d+)_s(\d+)$", run_dir.name)
        assert m_ and (int(m_.group(1)), int(m_.group(2))) == (r, N), f"run folder {run_dir.name} is not r{r} s{N}"
    a, clf = st["args"], st["clf"]
    base = a.base
    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    zt, yt = np.asarray(st["z_test0"], np.float64), np.asarray(st["true_test0_y"]).astype(int)
    zp, yp = np.asarray(st["z_pool"], np.float64), np.asarray(st["true_pool_y"]).astype(int)
    assert set(np.unique(y0)) == set(range(base)) and set(np.unique(yt)) <= set(range(base))
    old = (yp < base).astype(int)
    out, rec, chk = {"dataset": dataset, "ft": r, "seed": N}, {}, {}

    # A0: the pipeline's session-0 classifier
    p0 = _jnp(st["params0"])
    pred, lp = (np.asarray(v) for v in clf._predict(jnp.asarray(st["z_test0"]), p0))
    lpp = np.asarray(clf._predict(jnp.asarray(st["z_pool"]), p0)[1])
    n_ok = int((pred == yt).sum())
    if dataset == "gc10":
        ref = int(round(st["acc0"]["All"] * len(yt) / 100))
        ref_src = "acc0 of the snapshot"
    else:
        f = sorted((ROOT / "runs").glob(f"*__cub_base_ft_r{r}_s{N}/metrics.json"))
        assert len(f) == 1, f"cub_base_ft_r{r}_s{N}: {len(f)} runs"
        ref = int(round(json.loads(f[0].read_text())["S0_All"] * len(yt) / 100))
        ref_src = f"S0_All of {f[0].parent.name}"
    chk["C1_ok"] = int(n_ok == ref)
    print(f"e0604 C1: A0 {n_ok}/{len(yt)} correct, {ref_src} {ref}/{len(yt)} -> {'OK' if chk['C1_ok'] else 'FAIL'}")

    # A1: E06.03 through the pipeline's own code
    clf.global_params, clf.label_offset = None, 0
    with contextlib.redirect_stdout(io.StringIO()):
        p1 = clf._fit_closed(st["z_s0"], st["s0_y"], None, "e0604")
    pred1, lp1 = (np.asarray(v) for v in clf._predict(jnp.asarray(st["z_test0"]), p1))
    lpp1 = np.asarray(clf._predict(jnp.asarray(st["z_pool"]), p1)[1])
    from classifier.closed_form import closed_form_params
    q1, _ = closed_form_params(z0, y0, list(range(base)), prev=None, num_classes=a.num_classes)
    ind = _gauss_logpdf64(zt, q1["class_means"], q1["class_covs"]).argmax(1)
    chk["C3_mismatch"] = int((ind != pred1).sum())
    chk["C3_ok"] = int(chk["C3_mismatch"] == 0)
    print(f"e0604 C3: A1 against the float64 recomputation, {chk['C3_mismatch']} labels differ -> {'OK' if chk['C3_ok'] else 'FAIL'}")

    # A2, A3: NIW posterior, off by leave-one-out on the S0 rows
    curve = niw.loo_curve(z0, y0)
    off = niw.choose(curve)
    model = niw.fit(z0, y0, off)
    if r == 0 and N == 0:
        stc = niw.class_stats(z0, y0)
        x = z0[y0 == stc["classes"][0]]
        args_ev = (stc["m0"], niw.K0, stc["d"] + 1 + off, off * niw.prior_cov(stc, 0))
        ev, sq = niw.log_evidence(x, *args_ev), niw.log_sequential(x, *args_ev)
        direct, d, k0 = 0.0, stc["d"], niw.K0
        for i in range(len(z0)):                # each S0 row: statistics without it, posterior of its own class only
            keep = np.arange(len(z0)) != i
            s_i = niw.class_stats(z0[keep], y0[keep])
            k = int(np.where(s_i["classes"] == y0[i])[0][0])
            n, dm = s_i["n"][k], s_i["mean"][k] - s_i["m0"]
            psi = off * niw.prior_cov(s_i, k) + s_i["scatter"][k] + (k0 * n / (k0 + n)) * np.outer(dm, dm)
            L = np.linalg.cholesky(psi)
            q = (np.linalg.solve(L, z0[i] - (k0 * s_i["m0"] + n * s_i["mean"][k]) / (k0 + n)) ** 2).sum()
            direct += niw._t_logpdf(q, 2 * np.log(np.diag(L)).sum(), k0 + n, d + 1 + off + n, d)
        fast = niw.loo_curve(z0, y0, (off,))[0]
        chk["C2_ev_rel"], chk["C2_loo_rel"] = abs(ev - sq) / abs(ev), abs(fast - direct) / abs(direct)
        chk["C2_ok"] = int(chk["C2_ev_rel"] <= 1e-8 and chk["C2_loo_rel"] <= 1e-8)
        print(f"e0604 C2: evidence vs sequential {chk['C2_ev_rel']:.1e}, fast vs direct leave-one-out "
              f"{chk['C2_loo_rel']:.1e} (off {off}) -> {'OK' if chk['C2_ok'] else 'FAIL'}")
    chk["checks_ok"] = int(chk["C1_ok"] and chk["C3_ok"] and chk.get("C2_ok", 1))
    if checks_only:
        print(f"e0604: checks only, {'all OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
        return chk

    lg, lt = niw.log_density(model, zt, "gauss"), niw.log_density(model, zt, "t")
    lgp, ltp = niw.log_density(model, zp, "gauss"), niw.log_density(model, zp, "t")
    cls = model["classes"]
    idx = np.arange(len(yt))
    for name, pr, lpt, lpool in (("A0", pred, lp[idx, yt], lpp[:, :base]),
                                 ("A0base", lp[:, :base].argmax(1), lp[idx, yt], lpp[:, :base]),
                                 ("A1", pred1, lp1[idx, yt], lpp1[:, :base]),
                                 ("A2", cls[lg.argmax(1)], lg[idx, yt], lgp),
                                 ("A3", cls[lt.argmax(1)], lt[idx, yt], ltp)):
        s, rec[name] = _branch_scores(name, pr, lpt, yt, base)
        out.update(s)
        out[f"auroc_{name}"] = float(roc_auc_score(old, lpool.max(1)))
    out["A0_unseen"] = int((pred >= base).sum())
    out["off"] = off
    out.update({f"loo_{o}": float(v) for o, v in zip(niw.OFFS, curve)})
    curve_lc = niw.loo_curve(z0, y0, loo_class=True)
    out["off_lc"] = niw.choose(curve_lc)
    out.update({f"loolc_{o}": float(v) for o, v in zip(niw.OFFS, curve_lc)})
    m_lc = niw.fit(z0, y0, out["off_lc"], loo_class=True)
    for name, kind in (("A2lc", "gauss"), ("A3lc", "t")):
        out[f"acc_{name}"] = 100.0 * float((m_lc["classes"][niw.log_density(m_lc, zt, kind).argmax(1)] == yt).mean())
    means = np.asarray(st["params0"]["class_means"])[:base]
    covs = np.asarray(st["params0"]["class_covs"])[:base]
    xb = np.stack([z0[y0 == c].mean(0) for c in range(base)])
    ev0 = np.linalg.eigvalsh(covs.astype(np.float64))
    out["d23_rmu_med"] = float(np.median(np.linalg.norm(means, axis=1) / np.linalg.norm(xb, axis=1)))
    out["d23_eig_min"], out["d23_eig_max"] = float(ev0.min()), float(ev0.max())
    out.update(chk)
    print(f"e0604: {dataset} r{r} s{N} | off {off} | off_lc {out['off_lc']} | " +
          " | ".join(f"{b} {out[f'acc_{b}']:.2f}" for b in ("A0", "A1", "A2", "A3")) +
          f" | checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    if run_dir is not None:
        (run_dir / "recall.json").write_text(json.dumps({k: v.tolist() for k, v in rec.items()}) + "\n")
        if dataset == "cub200":
            import pickle
            keep = ("s0_y", "z_s0", "z_test0", "true_test0_y", "z_pool", "true_pool_y", "pca", "acc0")
            state = {k: st[k] for k in keep}
            state["params0"] = {k: np.asarray(v)[:base] for k, v in st["params0"].items()}
            state["meta"] = dict(_git(), args=vars(a), device=device, note="E06.04: state after step 08, base slots only")
            (run_dir / "state08.pkl").write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
    return out


def e0604_summary():
    """Decision table of E06.04 from the runs gc10_e06_04_r<r>_s<N> and cub_e06_04_r<r>_s<N>, r = 0-2, N = 0-4.
    D = acc A3 - acc A0 per chain, D_r = mean over the 5 seeds, D = mean of D_r; the rule of the pre-registration
    (accept: D >= 1 point and D_r > 0 in >= 2/3 FT seeds; reject: D <= 0; else not large enough), the same rule for
    A2 - A0, and the report numbers. Refuses to read results unless every run has checks_ok = 1 and C2 ran in r0 s0."""
    import pandas as pd
    import niw
    rule = lambda D, Dr: "accept" if D >= 1 and (Dr > 0).sum() >= 2 else "reject" if D <= 0 else "not large enough"
    branches = ("A0", "A0base", "A1", "A2", "A3")
    res = {}
    for ds, pre in (("gc10", "gc10"), ("cub200", "cub")):
        rows = []
        for r in range(3):
            for N in range(5):
                f = sorted((ROOT / "runs").glob(f"*__{pre}_e06_04_r{r}_s{N}/metrics.json"))
                assert len(f) == 1, f"{pre}_e06_04_r{r}_s{N}: {len(f)} runs with metrics"
                m_ = json.loads(f[0].read_text())
                assert m_.get("checks_ok") == 1 and "acc_A3" in m_, f"{pre} r{r} s{N}: checks failed or no result, not read"
                assert (r, N) != (0, 0) or m_.get("C2_ok") == 1, f"{pre} r0 s0: C2 did not run or failed"
                rows.append({**{k: v for k, v in m_.items() if not isinstance(v, str)}, "ft": f"r{r}", "seed": N})
        df = pd.DataFrame(rows)
        for b in ("A1", "A2", "A3"):
            df[f"D_{b}"] = df[f"acc_{b}"] - df["acc_A0"]
        df["D_A3_A2"] = df["acc_A3"] - df["acc_A2"]
        by = df.groupby("ft").mean(numeric_only=True)
        sd = df.groupby("ft")[[f"acc_{b}" for b in branches]].std(ddof=1)
        pd.set_option("display.width", 250)
        print(f"\n===== {ds}")
        print(df[["ft", "seed", "off", "off_lc", "acc_A0", "acc_A1", "acc_A2", "acc_A3", "D_A3", "A0_unseen"]]
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        cols = ["D_A3", "D_A2", "D_A1", "D_A3_A2"] + [f"{k}_{b}" for k in ("bal", "rmin") for b in ("A0", "A1", "A2", "A3")] + \
               ["logd_A2", "logd_A3"] + [f"auroc_{b}" for b in ("A0", "A1", "A2", "A3")] + \
               ["acc_A0base", "acc_A2lc", "acc_A3lc", "d23_rmu_med", "d23_eig_min", "d23_eig_max"]
        print("\nmean by FT seed:")
        print(by[cols].T.to_string(float_format=lambda v: f"{v:.4f}"))
        print("\nSD (ddof 1) over the 5 seeds, accuracy of each branch:")
        print(sd.T.to_string(float_format=lambda v: f"{v:.3f}"))
        pooled = float(np.sqrt((sd["acc_A0"] ** 2).mean()))
        D3, D2 = by["D_A3"], by["D_A2"]
        grid = niw.OFFS
        print(f"\npooled SD of acc A0 = {pooled:.3f} points | off at the upper edge of the grid in "
              f"{int((df['off'] == max(grid)).sum())}/15 chains, at the lower edge in {int((df['off'] == min(grid)).sum())}/15 | "
              f"logd A3 > logd A2 in {int((df['logd_A3'] > df['logd_A2']).sum())}/15 chains")
        print(f"D (A3 - A0) = {D3.mean():.3f} points, D_r = {' / '.join(f'{v:.3f}' for v in D3)}, "
              f"positive in {int((D3 > 0).sum())}/3 FT seeds -> {rule(D3.mean(), D3)}")
        print(f"A2 - A0 = {D2.mean():.3f} points, D_r = {' / '.join(f'{v:.3f}' for v in D2)} -> {rule(D2.mean(), D2)}")
        res[ds] = (rule(D3.mean(), D3), rule(D2.mean(), D2))
    (c3, c2), (g3, _) = res["cub200"], res["gc10"]
    if c3 == "accept" and g3 != "reject":
        verdict = "A3 is the representation of the old classes for E09.08"
    elif c3 == "reject":
        verdict = "CUB rejected: no NIW posterior as old classes on CUB; review the strategy before pre-registering E09.08"
    elif c2 == "accept":
        verdict = "prior accepted, Student-t not: E09.08 goes on (its E-step uses the quadratic form, not the Student-t)"
    else:
        verdict = "report only; the decision on E09.08 waits for the strategy review after E06.04 and E09.06"
    print(f"\nconsequence (pre-registration): {verdict}")
    return res


# ---------------------------------------------------------------- E09.06 (notes/pl/experiments.vi.md; pre-registered at 61b9eb9)
RULES_0906 = ("bf", "ward", "bhatt", "bhatt_mean")


def _ref_metrics(tag):
    """metrics.json of the one run with this tag."""
    f = sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json"))
    assert len(f) == 1, f"{tag}: {len(f)} runs with metrics.json"
    return json.loads(f[0].read_text())


def _purity_0906(y, sub, steps):
    """Share of the merge steps that join two groups with the same plurality class (old classes included; pool labels
    for scoring only)."""
    owner = lambda g: int(np.bincount(y[np.isin(sub, g)]).argmax())
    return float(np.mean([owner(a) == owner(b) for a, b, _ in steps]))


def tier2_e0906(run, k_first=0, k_last=10):
    """E09.06 on one snapshot run gc10_v2_ft_r<r>_s<N>: draws k_first..k_last. Draw 0 sets the global RNG to rng_in of
    step 09; draw k >= 1 to the state of RandomState(906000 + 1000 r + 100 N + k). In each draw the default branch
    (replay.s09_cluster: K = increment, n_init 10) and the variant (v09a_overcluster, K' = 20, then v09b_merge, rule
    "bf") start from the same state (CRN). The report-only rules (ward, bhatt, bhatt_mean) merge the same 20
    sub-clusters and are scored at tier 1 only. Every distinct partition of the two branches (pool ids and test ids,
    clusters renumbered by first appearance) runs steps 10-14 once, in this process, from the state after step 08.
    Keys: d<k>_def_*, d<k>_var_* (_measures_0907), d<k>_t1_<arm>_* (score09 of the arms def, k20, bf, ward, bhatt,
    bhatt_mean), d<k>_purity (bf), d<k>_J5, d<k>_J20, p<j>_* per partition run, ref_*, off, D_* and s_* (mean and SD,
    ddof 1, over the draws of variant minus default).
    Checks: C1 (draw 0) the default branch gives the snapshot's ids, test ids, inertia within 1e-6 and the measures of
    its chain; C2 (draw 0) the K' = 20 partition gives k20_inertia (1e-6), k20_owned_*, k20_best_* of
    gc10_e09_05_v2_r<r>_s<N>; C3 _check_09 on every partition run; C4 as E09.07; C5 every merged partition has
    increment non-empty clusters and is a coarsening of the 20 sub-clusters on pool and test, every bf step picks the
    pair with the largest log Bayes factor recomputed with niw.log_evidence over all current pairs (within 1e-6 nat),
    and off equals off_lc of gc10_e06_04_r<r>_s<N>."""
    assert 0 <= k_first <= k_last, "draws k_first..k_last"
    st = load(run, "08", check=False)      # new_state first: imports torch, jax, main in the order of the v2 runs
    blobs = snapshots(run)
    _fresh(blobs, "14")                    # the snapshot's chain is the reference: it must match the code now
    import jax
    device = jax.default_backend()
    assert device == "gpu" or os.environ.get("REPLAY_ALLOW_CPU") == "1", f"JAX runs on {device}, not on the GPU (A1)"
    assert {b["meta"].get("device", "gpu") for b in blobs.values()} == {device}, "snapshots ran on another device"
    a = st["args"]
    assert a.dataset == "gc10", "E09.06 is pre-registered on GC10"
    tag = re.search(r"_r(\d+)_s(\d+)$", _run_dir(run).name)
    r, N = int(tag.group(1)), int(tag.group(2))
    assert N == a.seed, f"run name says seed {N}, the snapshot has seed {a.seed}"
    names = [c.split("_")[0] for c in json.load(open(ROOT / importlib.import_module(f"dataloaders.{a.dataset}").SPLIT))["classes"]]
    nov = names[a.base:]
    K, xt = int(a.increment), st["test1_x"]
    yp = np.asarray(st["true_pool_y"]).astype(int)                   # scoring only (tier 1)
    view = {key: v for key, v in st.items() if not key.startswith("true_")}   # what 09, 09a, 09b get
    prior = merge_prior(view)
    parts, index, c5 = [], {}, []

    def part(ids, cl, inertia):
        """Index of a partition among the distinct ones seen so far."""
        ids, tid = np.asarray(ids), np.asarray(cl.predict_raw(xt))
        key = canon(np.concatenate([ids, tid])).tobytes()
        if key not in index:
            index[key] = len(parts)
            parts.append(dict(cl=cl, ids=ids, tid=tid, inertia=inertia))
        return index[key]

    out = {"seed": N, "ft_seed": r, "k_first": k_first, "k_last": k_last, "off": prior["off"]}
    draws, t0, k20_0 = {}, time.time(), None
    for k in range(k_first, k_last + 1):
        state = blobs["09"]["rng_in"] if k == 0 else np.random.RandomState(906000 + 1000 * r + 100 * N + k).get_state()
        np.random.set_state(state)
        d9 = s09_cluster(view)
        np.random.set_state(state)
        sub = v09a_overcluster(view, k=20)
        s2 = {**view, **sub}
        mer = {rule: v09b_merge(s2, rule, prior) for rule in RULES_0906}
        sid, stid = np.asarray(sub["sub_ids"]), np.asarray(sub["sub_cluster"].predict_raw(xt))
        # C5: structure (bf merge) and a brute-force recomputation of every bf step
        m = mer["bf"]
        ok = len(np.unique(m["ids"])) == K and np.array_equal(m["ids"], m["cluster"].mapping[sid]) \
            and np.array_equal(m["cluster"].predict_raw(xt), m["cluster"].mapping[stid])
        z = np.asarray(view["z_pool"], np.float64)
        ev = lambda g: log_evidence(z[np.isin(sid, g)], prior["m0"], prior["k0"], prior["nu0"], prior["psi0"])
        groups = [(i,) for i in range(20)]
        for ga, gb, c in m["merge_steps"]:
            bfs = {(p, q): ev(p + q) - ev(p) - ev(q) for i, p in enumerate(groups) for q in groups[i + 1:]}
            best = max(bfs, key=bfs.get)
            ok = ok and set(best) == {ga, gb} and abs(-c - bfs[best]) <= 1e-6
            groups = [g for g in groups if g not in (ga, gb)] + [tuple(sorted(ga + gb))]
        c5.append(bool(ok))
        # tier 1 of the six arms (pool labels for scoring only)
        arms = {"def": d9["ids"], "k20": sid, **{rule: mer[rule]["ids"] for rule in RULES_0906}}
        for arm, ids in arms.items():
            t1 = score09(yp, ids, names, n_old=a.base)
            out[f"d{k}_t1_{arm}_bal"] = t1["bal_acc_hungarian"]
            out.update({f"d{k}_t1_{arm}_{q}_{c}": t1[f"{q}_{c}"] for c in nov for q in ("owned", "best")})
            if k == 0 and arm == "k20":
                k20_0 = dict(t1, inertia=sub["sub_inertia"])
        out[f"d{k}_purity"] = _purity_0906(yp, sid, m["merge_steps"])
        out[f"d{k}_J5"], out[f"d{k}_J20"] = d9["inertia"], sub["sub_inertia"]
        draws[k] = (part(d9["ids"], d9["cluster"], d9["inertia"]), part(m["ids"], m["cluster"], float("nan")))
    print(f"e0906: {len(draws)} draws in {time.time() - t0:.0f} s, {len(parts)} distinct partitions, off {prior['off']}",
          flush=True)
    base, res = dict(st), {}
    for j in sorted({p for d in draws.values() for p in d}):
        s2, p, t0 = dict(base), parts[j], time.time()
        s2.update(ids=p["ids"], cluster=p["cl"], inertia=p["inertia"])
        _check_09(s2)                                                # C3
        res[j] = _measures_0907(s2, names, _tail_0907(s2))
        res[j]["_keep"], res[j]["_acc1"] = np.asarray(s2["keep"]), dict(s2["acc1"])
        out.update({f"p{j}_bal14": res[j]["bal14"], f"p{j}_S1_Old": res[j]["S1_Old"]})
        print(f"e0906: partition {j} steps 10-14 in {time.time() - t0:.0f} s", flush=True)
    for k, (jd, jv) in draws.items():
        for j, name in ((jd, "def"), (jv, "var")):
            out.update({f"d{k}_{name}_{key}": v for key, v in res[j].items() if not key.startswith("_")})
            out[f"d{k}_{name}_part"] = j
    out["n_partitions_run"] = len(res)
    for key in ("bal14", "S1_Old", "bal10", "stop_S", "S1_New", "S1_All", "bal_all"):
        diff = np.array([out[f"d{k}_var_{key}"] - out[f"d{k}_def_{key}"] for k in draws], dtype=float)
        out[f"D_{key}"] = float(diff.mean())
        out[f"s_{key}"] = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
    e604 = _ref_metrics(f"gc10_e06_04_r{r}_s{N}")
    checks = {"C3_ok": 1,
              "C4_ok": int(all(v["gain08_14"] == 0 and v["old14"] == v["old08"] - v["lost08_14"] for v in res.values())),
              "C5_ok": int(all(c5) and prior["off"] == e604["off_lc"])}
    print("e0906: C5", {"merges": f"{sum(c5)}/{len(c5)}", "off": prior["off"], "off_lc E06.04": e604["off_lc"]}, flush=True)
    if 0 in draws:
        ref_st, o9, j0 = dict(base), blobs["09"]["out"], draws[0][0]
        for kk in ("09", "10", "11", "12", "13", "14"):
            ref_st.update(blobs[kk]["out"])
        ref = _measures_0907(ref_st, names, {"11": blobs["11"]["log"], "13": blobs["13"]["log"]})
        out.update({f"ref_{key}": v for key, v in ref.items()})
        close = lambda u, v: u == v if isinstance(v, int) else abs(u - v) <= 1e-6 * max(1.0, abs(v))
        same = {"ids": np.array_equal(parts[j0]["ids"], np.asarray(o9["ids"])),
                "test_ids": np.array_equal(parts[j0]["tid"], np.asarray(o9["cluster"].predict_raw(xt))),
                "inertia": abs(parts[j0]["inertia"] - o9["inertia"]) <= 1e-6 * o9["inertia"],
                "keep": np.array_equal(res[j0]["_keep"], np.asarray(blobs["12"]["out"]["keep"])),
                "acc1": all(close(res[j0]["_acc1"][key], float(v)) for key, v in blobs["14"]["out"]["acc1"].items()),
                "measures": all(close(res[j0][key], v) for key, v in ref.items())}
        print("e0906: C1", {key: bool(v) for key, v in same.items()}, flush=True)
        checks["C1_ok"] = int(all(same.values()))
        e905 = _ref_metrics(f"gc10_e09_05_v2_r{r}_s{N}")
        c2 = {"inertia": abs(k20_0["inertia"] - e905["k20_inertia"]) <= 1e-6 * e905["k20_inertia"],
              **{f"{q}_{c}": k20_0[f"{q}_{c}"] == e905[f"k20_{q}_{c}"] for c in nov for q in ("owned", "best")}}
        print("e0906: C2", {key: bool(v) for key, v in c2.items()}, flush=True)
        checks["C2_ok"] = int(all(c2.values()))
    out.update(checks)
    out["checks_ok"] = int(all(v == 1 for v in checks.values()))
    return out


def e0906_summary(T=0.0273, B=0.615):
    """Decision table of E09.06 from the 15 runs gc10_e09_06_r<r>_s<N> (and gc10_e09_06x_r<r>_s<N>, draws 11-20, when
    they exist): D_i, Delta_r, Delta, SE_MC, the decision rule; T and B recomputed from ref_* as a check; the condition
    for dropping the discovery line (best >= 0.5 after the merge in at least half of the (seed, draw) pairs of an FT
    seed); M1, M3, k = 0; tier 1 of the six arms; merge purity; the inputs of the predictions. Refuses to read results
    unless every run has checks_ok = 1."""
    import pandas as pd
    keys = ("bal14", "S1_Old", "bal10", "stop_S", "S1_New", "S1_All", "bal_all")
    arms, rare = ("def", "k20") + RULES_0906, ("crease", "rolled")
    rows, pairs = [], []
    for r in range(3):
        for N in range(5):
            ms = []
            for tag in (f"gc10_e09_06_r{r}_s{N}", f"gc10_e09_06x_r{r}_s{N}"):
                f = sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json"))
                assert len(f) == 1 or (len(f) == 0 and "x" in tag.split("_")[2]), f"{tag}: {len(f)} runs with metrics"
                ms += [json.loads(p.read_text()) for p in f]
            assert all(m_["checks_ok"] == 1 for m_ in ms), f"r{r} s{N}: checks failed, results are not read"
            assert "C1_ok" in ms[0] and ms[0]["k_first"] == 0, f"r{r} s{N}: the first run must hold draw 0 (C1)"
            ks = [(m_, k) for m_ in ms for k in range(m_["k_first"], m_["k_last"] + 1)]
            assert len({k for _, k in ks}) == len(ks), f"r{r} s{N}: a draw appears twice"
            row = {"snap": f"r{r}_s{N}", "ft": f"r{r}", "draws": len(ks), "off": ms[0]["off"],
                   "n_parts": sum(m_["n_partitions_run"] for m_ in ms)}
            for key in keys:
                d = np.array([m_[f"d{k}_var_{key}"] - m_[f"d{k}_def_{key}"] for m_, k in ks], dtype=float)
                row[f"D_{key}"], row[f"s_{key}"], row[f"k0_{key}"] = d.mean(), d.std(ddof=1), d[0]
            row["ref_bal14"], row["ref_S1_Old"] = ms[0]["ref_bal14"], ms[0]["ref_S1_Old"]
            row["purity"] = float(np.mean([m_[f"d{k}_purity"] for m_, k in ks]))
            for arm in arms:
                row[f"t1_{arm}"] = float(np.mean([m_[f"d{k}_t1_{arm}_bal"] for m_, k in ks]))
            row["silk_owned_bf"] = float(np.mean([m_[f"d{k}_t1_bf_owned_silk"] for m_, k in ks]))
            for m_, k in ks:
                pairs.append({"ft": f"r{r}", "silk1": int(m_[f"d{k}_t1_bf_owned_silk"] == 1),
                              **{f"{arm}_{q}_{c}": m_[f"d{k}_t1_{arm}_{q}_{c}"] for arm in arms for c in rare
                                 for q in ("owned", "best")}})
            rows.append(row)
    df, pr = pd.DataFrame(rows).set_index("snap"), pd.DataFrame(pairs)
    assert df["draws"].nunique() == 1, "every snapshot needs the same number of draws"
    n_draws, g = int(df["draws"].iloc[0]), df.groupby("ft")
    pd.set_option("display.width", 250)
    print(df[["draws", "off", "n_parts", "D_bal14", "s_bal14", "D_S1_Old", "s_S1_Old", "D_bal10", "D_stop_S", "k0_bal14",
              "k0_S1_Old", "purity", "silk_owned_bf"] + [f"t1_{arm}" for arm in arms]].round(4).to_string())
    by = g[[f"D_{key}" for key in keys] + [f"k0_{key}" for key in keys] + [f"t1_{arm}" for arm in arms]
           + ["purity", "silk_owned_bf"]].mean()
    print("\nmean by FT seed (Delta_r in the D_ columns):")
    print(by.round(4).to_string())
    delta, delta_old = float(by["D_bal14"].mean()), float(by["D_S1_Old"].mean())
    n_pos = int((by["D_bal14"] > 0).sum())
    se = {key: float(np.sqrt((df[f"s_{key}"] ** 2).sum() / n_draws) / len(df)) for key in ("bal14", "S1_Old")}
    thr = {name: float(np.sqrt((g[col].std(ddof=1) ** 2).mean())) for name, col in (("T", "ref_bal14"), ("B", "ref_S1_Old"))}
    print(f"\nthresholds recomputed from the default chains (pooled SD): T {thr['T']:.5f}, B {thr['B']:.5f}; "
          f"round to the pre-registered values: {round(thr['T'], 4) == T} {round(thr['B'], 3) == B}")
    if delta <= 0 or delta_old < -B:
        decision = "reject"
    else:
        decision = "accept" if delta >= T and n_pos >= 2 else "not large enough"
    near = [name for name, v, edge, s in (("Delta vs T", delta, T, se["bal14"]), ("Delta vs 0", delta, 0.0, se["bal14"]),
                                          ("Delta_Old vs -B", delta_old, -B, se["S1_Old"])) if abs(v - edge) < 2 * s]
    print(f"\nDelta = {delta:.5f} (T = {T}), Delta_r > 0 in {n_pos}/3 FT seeds, Delta_Old = {delta_old:.4f} points (B = {B}), "
          f"SE_MC = {se['bal14']:.5f} (Delta), {se['S1_Old']:.4f} (Delta_Old), {n_draws} draws")
    print(f"decision: {decision}")
    print("within 2 SE_MC of a boundary:", near or "none",
          "-> 10 more draws (k = 11-20), once" if near and n_draws == 11 else "")
    print(f"M1 = {by['D_bal10'].mean():.5f} | M3 = {by['D_stop_S'].mean():.2f} steps | k = 0 only: Delta "
          f"{by['k0_bal14'].mean():.5f}, Delta_Old {by['k0_S1_Old'].mean():.4f}")
    print("M_f: acc0 is the same in both branches, so Delta M_f = -Delta S1_Old (no separate column)")
    # drop condition: own cluster (best >= 0.5) after the bf merge in at least half of the pairs of an FT seed
    half = int(np.ceil(5 * n_draws / 2))
    keep = {c: pr.groupby("ft")[f"bf_best_{c}"].apply(lambda s: int((s >= 0.5).sum())) for c in rare}
    held = {c: int((keep[c] >= half).sum()) for c in rare}
    print(f"\nown cluster after the merge (best >= 0.5), pairs per FT seed out of {5 * n_draws} (need {half}): "
          + "; ".join(f"{c} {keep[c].tolist()}" for c in rare))
    if decision != "accept":
        drop = all(held[c] < 2 for c in rare)
        print("drop condition:", "MET -> drop the discovery line (patch features Q03.03 or minimal labels)" if drop else
              "not met -> bottleneck after step 09 (E09.08)")
    print("\nshare of (seed, draw) pairs per arm (tier 1): own cluster (best >= 0.5) / owns >= 1 cluster")
    tab = {arm: {f"{c}_{w}": float(((pr[f"{arm}_best_{c}"] >= 0.5) if w == "own" else (pr[f"{arm}_owned_{c}"] >= 1)).mean())
                 for c in rare for w in ("own", "owns")} for arm in arms}
    print(pd.DataFrame(tab).T.round(3).to_string())
    for c in rare:
        had = pr[f"k20_best_{c}"] >= 0.5
        kept = float((pr.loc[had, f"bf_best_{c}"] >= 0.5).mean()) if had.any() else float("nan")
        print(f"prediction input, {c}: own cluster at K' = 20 in {int(had.sum())} pairs, kept after the bf merge in "
              f"{kept:.3f} of them")
    print(f"prediction input, silk owns exactly 1 cluster after the bf merge: {pr['silk1'].mean():.3f} of the pairs")
    return df


# ---- E09.09: stop the 09b merge at log BF <= 0 (pre-registered cff7c46, amended 594e34b) ----

def _cut_0909(sid, steps, k, stop=True):
    """Groups after the bf merge path `steps` of v09b_merge, cut at the first step whose log Bayes factor is <= 0
    (cost >= 0), never below k groups; stop=False keeps all steps (= bf5). Returns (pool ids, number of steps kept).
    The greedy path is deterministic, so the stopped merge is a prefix of the full one (E09.09, amendment)."""
    sid = np.asarray(sid).astype(int)
    groups, t = [(i,) for i in range(sid.max() + 1)], 0
    for ga, gb, c in steps:
        if (stop and c >= 0) or len(groups) <= k:
            break
        groups = [g for g in groups if g not in (ga, gb)] + [tuple(sorted(ga + gb))]
        t += 1
    return merged_ids(sid, groups)[sid], t


def _draw_0909(view, yp, names, n_old, k, prior, factors=(0.5, 2.0)):
    """One draw of E09.09 from the global RNG state already set: v09a (K' = 20) once, the bf merge path of v09b, and
    the arms k20, bf5, bf0 (+ bf0 under off x factors). Pool labels yp: scoring only. Returns (keys, check dict)."""
    nov = names[n_old:]
    sub = v09a_overcluster(view, k=20)
    s2 = {**view, **sub}
    sid = np.asarray(sub["sub_ids"]).astype(int)
    mer = v09b_merge(s2, "bf", prior)
    steps = mer["merge_steps"]
    ids5, t5 = _cut_0909(sid, steps, k, stop=False)
    ids0, t0 = _cut_0909(sid, steps, k)
    # G3: the full cut is bf5; the stop is the first step with log BF <= 0, recomputed from niw.log_evidence
    z = np.asarray(view["z_pool"], np.float64)
    ev = lambda g: log_evidence(z[np.isin(sid, g)], prior["m0"], prior["k0"], prior["nu0"], prior["psi0"])
    groups = [(i,) for i in range(20)]
    for ga, gb, _ in steps[:t0]:
        groups = [g for g in groups if g not in (ga, gb)] + [tuple(sorted(ga + gb))]
    best = max(ev(p + q) - ev(p) - ev(q) for i, p in enumerate(groups) for q in groups[i + 1:]) if len(groups) > k else None
    g3 = {"full_is_bf5": bool(t5 == len(steps) and np.array_equal(ids5, np.asarray(mer["ids"]))),
          "prefix_positive": bool(all(-c > 0 for _, _, c in steps[:t0])),
          "stop_nonpositive": bool(len(groups) == k or (best is not None and best <= 1e-6))}
    out = {"purity": _purity_0906(yp, sid, steps), "J20": sub["sub_inertia"], "K_bf0": int(len(np.unique(ids0))),
           "steps_bf0": t0}
    for arm, ids in (("k20", sid), ("bf5", ids5), ("bf0", ids0)):
        t1 = score09(yp, ids, names, n_old=n_old)
        out[f"t1_{arm}_bal"] = t1["bal_acc_hungarian"]
        out.update({f"t1_{arm}_{q}_{c}": t1[f"{q}_{c}"] for c in nov for q in ("owned", "best")})
    for f, name in zip(factors, ("lo", "hi")):
        off = prior["off"] * f
        pr = dict(prior, off=off, nu0=prior["d"] + 1 + off, psi0=prior["psi0"] * f)
        ids_f, _ = _cut_0909(sid, v09b_merge(s2, "bf", pr)["merge_steps"], k)
        t1 = score09(yp, ids_f, names, n_old=n_old)
        out.update({f"t1_bf0_{name}_bal": t1["bal_acc_hungarian"], f"K_bf0_{name}": int(len(np.unique(ids_f))),
                    f"t1_bf0_{name}_best_rolled": t1["best_rolled"], f"t1_bf0_{name}_best_crease": t1["best_crease"]})
    return out, g3


def tier1_e0909(run):
    """E09.09 on one snapshot run gc10_v2_ft_r<r>_s<N>: 21 draws with the RNG rule of E09.06 (draw 0: rng_in of step
    09; draw k >= 1: RandomState(906000 + 1000 r + 100 N + k)); tier 1 only, no SVI. Keys d<k>_* (see _draw_0909).
    Gates: G1 bf5 equals d<k>_t1_bf_bal, J20 (1e-6 relative) and purity of E09.06 at every draw; G2 off equals off_lc of
    gc10_e06_04_r<r>_s<N>; G3 the full cut equals bf5 and the stop of bf0 is the first step with log BF <= 0."""
    st = load(run, "08", check=False)
    blobs = snapshots(run)
    _fresh(blobs, "09")
    a = st["args"]
    assert a.dataset == "gc10", "E09.09 is pre-registered on GC10"
    tag = re.search(r"_r(\d+)_s(\d+)$", _run_dir(run).name)
    r, N = int(tag.group(1)), int(tag.group(2))
    assert N == a.seed, f"run name says seed {N}, the snapshot has seed {a.seed}"
    names = [c.split("_")[0] for c in json.load(open(ROOT / importlib.import_module(f"dataloaders.{a.dataset}").SPLIT))["classes"]]
    yp = np.asarray(st["true_pool_y"]).astype(int)                   # scoring only (tier 1)
    view = {key: v for key, v in st.items() if not key.startswith("true_")}
    prior = merge_prior(view)
    ref = {}
    for t in (f"gc10_e09_06_r{r}_s{N}", f"gc10_e09_06x_r{r}_s{N}"):
        m = _ref_metrics(t)
        ref.update({k: m for k in range(m["k_first"], m["k_last"] + 1)})
    assert sorted(ref) == list(range(21)), f"r{r} s{N}: E09.06 draws {sorted(ref)}"
    out, g1, g3, t_start = {"seed": N, "ft_seed": r, "off": prior["off"]}, [], [], time.time()
    for k in range(21):
        state = blobs["09"]["rng_in"] if k == 0 else np.random.RandomState(906000 + 1000 * r + 100 * N + k).get_state()
        np.random.set_state(state)
        with contextlib.redirect_stdout(io.StringIO()):
            d, c3 = _draw_0909(view, yp, names, a.base, int(a.increment), prior)
        m = ref[k]
        g1.append(abs(d["t1_bf5_bal"] - m[f"d{k}_t1_bf_bal"]) < 1e-12 and abs(d["purity"] - m[f"d{k}_purity"]) < 1e-12
                  and abs(d["J20"] - m[f"d{k}_J20"]) <= 1e-6 * m[f"d{k}_J20"])
        g3.append(all(c3.values()))
        out.update({f"d{k}_{key}": v for key, v in d.items()})
    e604 = _ref_metrics(f"gc10_e06_04_r{r}_s{N}")
    out.update(G1_ok=int(all(g1)), G2_ok=int(prior["off"] == e604["off_lc"]), G3_ok=int(all(g3)))
    out["checks_ok"] = int(out["G1_ok"] == out["G2_ok"] == out["G3_ok"] == 1)
    print(f"e0909: 21 draws in {time.time() - t_start:.0f} s | G1 {sum(g1)}/21 | G2 off {prior['off']} vs off_lc "
          f"{e604['off_lc']} | G3 {sum(g3)}/21", flush=True)
    return out


def e0909_summary(T=0.0273):
    """Decision table of E09.09 from the 25 runs gc10_e09_09_r<r>_s<N>: Delta_r (bf0 - bf5, tier 1, mean over 5
    snapshots x 21 draws), the retention of rolled_pit (own cluster, best >= 0.5, at bf0 among the draws where it has
    one at K' = 20), the pre-registered rule on r3, r4; r0-r2 report only (not clean). Refuses to read failed runs."""
    rows = {}
    for r in range(5):
        ms = []
        for N in range(5):
            f = sorted((ROOT / "runs").glob(f"*__gc10_e09_09_r{r}_s{N}/metrics.json"))
            if len(f) != 1:
                break
            m = json.loads(f[0].read_text())
            assert m["checks_ok"] == 1, f"r{r} s{N}: gates failed, results are not read"
            ms.append(m)
        if len(ms) < 5:
            print(f"r{r}: {len(ms)}/5 runs, skipped")
            continue
        g = lambda key: np.array([[m[f"d{k}_{key}"] for k in range(21)] for m in ms], dtype=float)   # 5 x 21
        own20, own0 = g("t1_k20_best_rolled") >= 0.5, g("t1_bf0_best_rolled") >= 0.5
        row = {"D": (g("t1_bf0_bal") - g("t1_bf5_bal")).mean(1).mean(),
               "D_lo": (g("t1_bf0_lo_bal") - g("t1_bf5_bal")).mean(), "D_hi": (g("t1_bf0_hi_bal") - g("t1_bf5_bal")).mean(),
               "bf5": g("t1_bf5_bal").mean(), "bf0": g("t1_bf0_bal").mean(), "k20": g("t1_k20_bal").mean(),
               "K_med": float(np.median(g("K_bf0"))), "K_min": int(g("K_bf0").min()), "K_max": int(g("K_bf0").max()),
               "K_lo_med": float(np.median(g("K_bf0_lo"))), "K_hi_med": float(np.median(g("K_bf0_hi"))),
               "rolled_k20": int(own20.sum()), "rolled_kept": int((own20 & own0).sum()),
               "rolled_bf5": int((g("t1_bf5_best_rolled") >= 0.5).sum()), "rolled_bf0": int(own0.sum()),
               "crease_bf5": int((g("t1_bf5_best_crease") >= 0.5).sum()), "crease_bf0": int((g("t1_bf0_best_crease") >= 0.5).sum())}
        for c in ("silk", "inclusion", "rolled", "crease", "waist"):
            row[f"best_{c}"] = (g(f"t1_bf5_best_{c}").mean(), g(f"t1_bf0_best_{c}").mean())
        rows[r] = row
        clean = "decision" if r in (3, 4) else "report only, not clean"
        print(f"r{r} ({clean}): Delta_r {row['D']:+.4f} (bf5 {row['bf5']:.4f}, bf0 {row['bf0']:.4f}, k20 {row['k20']:.4f}) | "
              f"K_bf0 median {row['K_med']} [{row['K_min']}, {row['K_max']}] | rolled: own at K'=20 {row['rolled_k20']}/105, "
              f"kept at bf0 {row['rolled_kept']}, own bf5 {row['rolled_bf5']} bf0 {row['rolled_bf0']} | crease own bf5 "
              f"{row['crease_bf5']} bf0 {row['crease_bf0']}")
        print("      best bf5 -> bf0: " + ", ".join(f"{c} {v[0]:.3f} -> {v[1]:.3f}" for c, v in
                                                  ((c, row[f"best_{c}"]) for c in ("silk", "inclusion", "rolled", "crease", "waist"))))
        print(f"      off x0.5: Delta {row['D_lo']:+.4f}, K median {row['K_lo_med']} | off x2: Delta {row['D_hi']:+.4f}, "
              f"K median {row['K_hi_med']}")
    if 3 in rows and 4 in rows:
        d3, d4 = rows[3]["D"], rows[4]["D"]
        den = rows[3]["rolled_k20"] + rows[4]["rolled_k20"]
        keep = (rows[3]["rolled_kept"] + rows[4]["rolled_kept"]) / den if den else float("nan")
        if (d3 + d4) / 2 <= 0:
            dec = "DROP (LOẠI)"
        elif den and d3 > 0 and d4 > 0 and keep >= 0.5:
            dec = "ALIVE (SỐNG)"
        else:
            dec = "UNCLEAR (KHÔNG RÕ)"
        print(f"\npre-registered rule (cff7c46, amended 594e34b): Delta_r3 {d3:+.4f}, Delta_r4 {d4:+.4f}, mean "
              f"{(d3 + d4) / 2:+.4f}; rolled_pit retention on r3, r4 {keep:.3f} ({rows[3]['rolled_kept'] + rows[4]['rolled_kept']}"
              f"/{den}; bf5 reference 0.015) -> {dec}")


# ---- E11.01: column 11 and the stop-step counterfactual on 09a + 09b (pre-registered 03e7f86) ----

def _measures_1101(st, names, logs):
    """_measures_0907 plus the keys of E11.01, from the loss ledger: bal11 (balanced novel at column 11, the classifier
    with the pass-F Gaussians), old11 (old test images right at column 11), L_F = bal10 - bal11, L_S = bal11 - bal14,
    filt_novel_right (pool: novel images kept by the filter whose cluster carries their class, column 12),
    filt_old_kept (pool: old images not dropped by the filter)."""
    m = _measures_0907(st, names, logs)
    led = ledger_state(st)
    t, o, p = led["test1"], led["test_old"], led["pool"]
    nov = p["novel"].to_numpy()
    m.update(bal11=float(t.groupby("y")["11"].mean().mean()), old11=int(o["11"].sum()),
             filt_novel_right=int(p.loc[nov, "12"].sum()), filt_old_kept=int((~nov).sum() - p.loc[~nov, "12"].sum()))
    m["L_F"], m["L_S"] = m["bal10"] - m["bal11"], m["bal11"] - m["bal14"]
    return m


def _steps_1101(st, a, b):
    """Run steps a..b of the default chain on st in place (no snapshots, no global RNG); return the log text."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        chain(st, start=a, upto=b)
    return buf.getvalue()


def _pass_s_1101(s12, step):
    """Pass S and step 14 on a copy of a state after step 12, with pass S stopped at loop step `step` by stop_at_step
    (lab/variants.py). The copy gets its own args (stop_at set) and stop component; s12 is not changed.
    Returns the state after step 14 (args and stop put back) and the log of step 13."""
    import copy
    s2 = dict(s12)
    s2["args"] = copy.copy(s12["args"])
    s2["args"].stop_at = int(step)
    s2["stop"] = stop_at_step.__name__
    log = _steps_1101(s2, "13", "13")
    s2["args"], s2["stop"] = s12["args"], s12["stop"]
    _steps_1101(s2, "14", "14")
    return s2, log


def tier2_e1101(run, k_first=0, k_last=3):
    """E11.01 on one snapshot run gc10_v2_ft_r<r>_s<N>: draws k_first..k_last (0-3; 4-7 only as extension (1)), with
    the RNG rule of E09.06 (draw 0: rng_in of step 09; draw k >= 1: RandomState(906000 + 1000 r + 100 N + k)).
    Branches per draw: def (s09_cluster), var (v09a_overcluster K' = 20, then v09b_merge rule "bf"), both through the
    default steps 10-14; vd = the partition and steps 10-12 of var, pass S stopped at the pass-S stop step of def
    (stop_at_step), then step 14; no vd when that step is <= n_epochs / 3 (counted in n_vd_missing).
    Keys: d<k>_<branch>_* (_measures_1101) for branch def, var, vd; n_vd_missing; G1_ok ... G4_ok, checks_ok.
    Gates: G1 def and var equal the keys d<k>_def_*, d<k>_var_* of gc10_e09_06_r<r>_s<N> (bal10, bal14, stop_F,
    stop_S, S1_Old, old14; floats within 1e-6); G2 (draw 0 only) var with pass S stopped at its own natural stop step
    gives params1 equal to the bit; G3 stop_S of vd equals stop_S of def; G4 _check_09 on every partition (C3) and the
    old-image flows of E09.06 (C4)."""
    assert (k_first, k_last) in ((0, 3), (4, 7)), "E11.01: draws 0-3, or 4-7 for extension (1)"
    st = load(run, "08", check=False)      # new_state first: imports torch, jax, main in the order of the v2 runs
    blobs = snapshots(run)
    _fresh(blobs, "14")
    import jax
    device = jax.default_backend()
    assert device == "gpu" or os.environ.get("REPLAY_ALLOW_CPU") == "1", f"JAX runs on {device}, not on the GPU (A1)"
    assert {b["meta"].get("device", "gpu") for b in blobs.values()} == {device}, "snapshots ran on another device"
    a = st["args"]
    assert a.dataset == "gc10", "E11.01 is pre-registered on GC10"
    tag = re.search(r"_r(\d+)_s(\d+)$", _run_dir(run).name)
    r, N = int(tag.group(1)), int(tag.group(2))
    assert N == a.seed, f"run name says seed {N}, the snapshot has seed {a.seed}"
    names = [c.split("_")[0] for c in json.load(open(ROOT / importlib.import_module(f"dataloaders.{a.dataset}").SPLIT))["classes"]]
    view = {key: v for key, v in st.items() if not key.startswith("true_")}   # what 09, 09a, 09b get
    prior = merge_prior(view)
    e906 = _ref_metrics(f"gc10_e09_06_r{r}_s{N}")
    n_ep = int(a.n_epochs)
    close = lambda u, v: u == v if isinstance(v, int) else abs(u - v) <= 1e-6 * max(1.0, abs(v))
    g1_keys = ("bal10", "bal14", "stop_F", "stop_S", "S1_Old", "old14")
    out = {"seed": N, "ft_seed": r, "k_first": k_first, "k_last": k_last, "off": prior["off"]}
    g1, g3, g4, g2, n_missing, t0 = [], [], [], None, 0, time.time()
    for k in range(k_first, k_last + 1):
        state = blobs["09"]["rng_in"] if k == 0 else np.random.RandomState(906000 + 1000 * r + 100 * N + k).get_state()
        np.random.set_state(state)
        d9 = s09_cluster(view)
        np.random.set_state(state)
        sub = v09a_overcluster(view, k=20)
        mer = v09b_merge({**view, **sub}, "bf", prior)
        res, s12_var = {}, None
        for name, ids, cl, inertia in (("def", d9["ids"], d9["cluster"], d9["inertia"]),
                                       ("var", mer["ids"], mer["cluster"], float("nan"))):
            s2 = dict(st)
            s2.update(ids=np.asarray(ids), cluster=cl, inertia=inertia)
            _check_09(s2)                                            # G4, C3
            logs = {"11": _steps_1101(s2, "10", "11")}
            _steps_1101(s2, "12", "12")
            if name == "var":
                s12_var = dict(s2)
            logs["13"] = _steps_1101(s2, "13", "13")
            _steps_1101(s2, "14", "14")
            res[name] = _measures_1101(s2, names, logs)
            res[name]["_p1"] = s2["params1"]
            g1.append(all(close(res[name][key], e906[f"d{k}_{name}_{key}"]) for key in g1_keys))
        stop_def = res["def"]["stop_S"]
        if stop_def == n_ep or stop_def > n_ep / 3:
            s_vd, log13 = _pass_s_1101(s12_var, stop_def)
            res["vd"] = _measures_1101(s_vd, names, {"11": logs["11"], "13": log13})
            g3.append(res["vd"]["stop_S"] == stop_def)
        else:
            n_missing += 1
        if k == 0:                                                   # G2
            s_own, _ = _pass_s_1101(s12_var, res["var"]["stop_S"])
            g2 = all(np.array_equal(np.asarray(s_own["params1"][key]), np.asarray(res["var"]["_p1"][key]))
                     for key in res["var"]["_p1"])
        for name, m in res.items():
            g4.append(m["gain08_14"] == 0 and m["old14"] == m["old08"] - m["lost08_14"])   # G4, C4
            out.update({f"d{k}_{name}_{key}": v for key, v in m.items() if not key.startswith("_")})
        print(f"e1101: draw {k} done at {time.time() - t0:.0f} s; stop_S def {stop_def}, var {res['var']['stop_S']}",
              flush=True)
    out["n_vd_missing"] = n_missing
    checks = {"G1_ok": int(all(g1)), "G3_ok": int(all(g3)), "G4_ok": int(all(g4))}
    if g2 is not None:
        checks["G2_ok"] = int(g2)
    print("e1101: gates", {key: v for key, v in checks.items()}, f"G1 {sum(g1)}/{len(g1)}, G3 {sum(g3)}/{len(g3)}",
          flush=True)
    out.update(checks)
    out["checks_ok"] = int(all(v == 1 for v in checks.values()))
    return out


def e1101_summary(B=0.615):
    """Decision table of E11.01 from the 25 runs gc10_e11_01_r<r>_s<N> (and gc10_e11_01x_r<r>_s<N>, draws 4-7, when
    they exist): per FT seed the means over 5 snapshots x draws of L_F, L_S (def, var), old14 and bal14 of the three
    branches; the rules of the pre-registration (Q11.01 on var: L_F >= L_S in >= 4/5 FT seeds; Q07.04: |D_old| < B
    in images). Refuses to read results unless every run has checks_ok = 1 and the first run holds draw 0 (G2)."""
    import pandas as pd
    n_old = 241                                                      # old test images of GC10 t1 (00-data-gc10)
    b_img = B / 100 * n_old
    rows = []
    for r in range(5):
        for N in range(5):
            ms = []
            for tag in (f"gc10_e11_01_r{r}_s{N}", f"gc10_e11_01x_r{r}_s{N}"):
                f = sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json"))
                assert len(f) == 1 or (len(f) == 0 and tag.startswith("gc10_e11_01x")), f"{tag}: {len(f)} runs with metrics"
                ms += [json.loads(q.read_text()) for q in f]
            assert all(m_["checks_ok"] == 1 for m_ in ms), f"r{r} s{N}: checks failed, results are not read"
            assert "G2_ok" in ms[0] and ms[0]["k_first"] == 0, f"r{r} s{N}: the first run must hold draw 0 (G2)"
            for m_ in ms:
                for k in range(m_["k_first"], m_["k_last"] + 1):
                    row = {"ft": f"r{r}", "snap": f"r{r}_s{N}", "k": k, "vd": f"d{k}_vd_old14" in m_}
                    for br in ("def", "var", "vd"):
                        for key in ("L_F", "L_S", "bal10", "bal11", "bal14", "old11", "old14", "stop_S",
                                    "filt_novel_right", "filt_old_kept"):
                            row[f"{br}_{key}"] = m_.get(f"d{k}_{br}_{key}", float("nan"))
                    rows.append(row)
    df = pd.DataFrame(rows)
    per = df.groupby("snap")["k"].nunique()
    assert per.nunique() == 1, "every snapshot needs the same draws"
    n_draws = int(per.iloc[0])
    df["D_old"] = df["vd_old14"] - df["def_old14"]
    df["D_old_nat"] = df["var_old14"] - df["def_old14"]
    df["cost_new"] = df["var_bal14"] - df["vd_bal14"]
    cols = [f"{br}_{key}" for br in ("def", "var") for key in ("L_F", "L_S")] + \
           ["def_old14", "var_old14", "vd_old14", "D_old", "D_old_nat", "cost_new", "def_stop_S", "var_stop_S",
            "var_filt_novel_right", "var_filt_old_kept", "def_filt_novel_right", "def_filt_old_kept"]
    by = df.groupby("ft")[cols].mean()
    pd.set_option("display.width", 250)
    print(f"{n_draws} draws per snapshot; draws without vd: {int((~df['vd']).sum())} of {len(df)}")
    print(by.round(4).to_string())
    n_f = {br: int((by[f"{br}_L_F"] >= by[f"{br}_L_S"]).sum()) for br in ("def", "var")}
    q11 = "PASS F" if n_f["var"] >= 4 else "FILTER AND PASS S" if n_f["var"] <= 1 else "BOTH"
    d_old = float(by["D_old"].mean())
    q704 = "STOP STEP EXPLAINS" if abs(d_old) < b_img else "PARTITION HAS A PART"
    print(f"\nQ11.01 (var): L_F >= L_S in {n_f['var']}/5 FT seeds -> {q11}; def: {n_f['def']}/5")
    print(f"var - def: L_F {float((by['var_L_F'] - by['def_L_F']).mean()):+.4f}, "
          f"L_S {float((by['var_L_S'] - by['def_L_S']).mean()):+.4f}")
    print(f"Q07.04: D_old = {d_old:+.3f} images (B = {b_img:.2f} images) -> {q704}")
    print(f"reported: old14 var - def at natural stops {float(by['D_old_nat'].mean()):+.3f} images; "
          f"cost of moving the stop step for the novel classes (bal14 var - vd) {float(by['cost_new'].mean()):+.4f}")
    if q11 == "BOTH" and n_draws == 4:
        print("-> extension (1): draws 4-7, once (tag gc10_e11_01x_r<r>_s<N>, --k-first 4 --k-last 7)")
    if q11 == "FILTER AND PASS S":
        print("-> extension (2): oracle-filter branch (needs its own code)")
    return df


# ---- CUB t5: T, B of QD28 from the default chains of lab/sessions.py, on both feature sets ----
def cub_tb(prefixes=("cub_chain_ft", "cub_chain_ft_l25"), official="cub_chain_ft_l25"):
    """T = pooled SD of M_d, B = pooled SD of S5_S0test, between the 5 chain seeds inside each FT seed (QD28; as
    14-eval section 3.3 for GC10), for each feature set: cub_chain_ft (fine-tuned on 2996 images, D9) and
    cub_chain_ft_l25 (2500 labeled images). Reads runs/*/config.json and metrics.json only. T and B measure the
    chain-seed noise on fixed features: the yardstick for paired variants on one snapshot, not for the difference of
    the two feature sets (that one is the spread of the FT-seed means, n = 3; no test is made)."""
    keys, out = ("M_d", "S5_S0test", "S5_All", "S5_Old", "M_f"), {}
    for prefix in prefixes:
        rows, tab = {}, {}
        for f in sorted((ROOT / "runs").glob(f"*__{prefix}_r[0-9]_s[0-9]/config.json")):
            cfg = json.loads(f.read_text())
            if cfg.get("exit_code") != 0:
                continue
            m = re.fullmatch(rf"{re.escape(prefix)}_r(\d)_s(\d)", cfg["tag"])
            key = (int(m.group(1)), int(m.group(2)))
            assert key not in rows, f"{cfg['tag']}: more than one run with exit 0"
            assert cfg["git"]["dirty"] is False, f"{cfg['run_id']}: dirty tree"
            assert len(cfg["inputs"]) == 1, f"{cfg['run_id']}: expected one input"
            rows[key] = (next(iter(cfg["inputs"].values())), json.loads((f.parent / "metrics.json").read_text()))
        assert sorted(rows) == [(r, k) for r in range(3) for k in range(5)], f"{prefix}: need 3 FT seeds x 5 seeds, found {sorted(rows)}"
        hashes = [{rows[r, k][0] for k in range(5)} for r in range(3)]
        assert all(len(h) == 1 for h in hashes) and len(set.union(*hashes)) == 3, f"{prefix}: input hashes {hashes}"
        for key in keys:
            v = np.array([[rows[r, k][1][key] for k in range(5)] for r in range(3)], dtype=float)
            tab[key] = (v.mean(1), v.std(1, ddof=1))
        print(f"\n{prefix}: 15 runs, inputs {' '.join(sorted(h.pop()[:6] for h in hashes))}")
        print("  FT seed  " + "  ".join(f"{key:>17}" for key in keys) + "     (mean, SD between the 5 seeds, ddof 1)")
        for r in range(3):
            print(f"  r{r}       " + "  ".join(f"{tab[key][0][r]:9.3f} {tab[key][1][r]:7.3f}" for key in keys))
        t, b = (float(np.sqrt((tab[key][1] ** 2).mean())) for key in ("M_d", "S5_S0test"))
        out[prefix] = {"T": t, "B": b, "mean": {key: float(tab[key][0].mean()) for key in keys},
                       "sd_ft": {key: float(tab[key][0].std(ddof=1)) for key in keys}}
        print(f"  T (pooled SD of M_d) = {t:.5f}   B (pooled SD of S5_S0test) = {b:.5f}   mean of the SDs: {tab['M_d'][1].mean():.5f}, {tab['S5_S0test'][1].mean():.5f}")
        print("  mean over the FT seeds: " + "  ".join(f"{key} {out[prefix]['mean'][key]:.3f}" for key in keys))
        print("  SD of the 3 FT-seed means: " + "  ".join(f"{key} {out[prefix]['sd_ft'][key]:.3f}" for key in keys))
    if len(prefixes) == 2:
        a, b = prefixes
        print(f"\n{b} minus {a}: difference of two feature sets (2996 vs 2500 fine-tune images), not the leak size; "
              "FT seeds are not paired, no test")
        for key in keys:
            print(f"  {key:>10} {out[b]['mean'][key] - out[a]['mean'][key]:+.3f}   (SD of the FT-seed means: {out[a]['sd_ft'][key]:.3f} and {out[b]['sd_ft'][key]:.3f})")
    if official in out:
        print(f"\nofficial for CUB (QD28): T = {out[official]['T']:.5f}, B = {out[official]['B']:.5f}, from {official}")
    return out


# ---- E14.01: sensitivity of the scores to the slot-class matching rule on CUB t5 (pre-registered at 5c6956f) ----
RULES_1401 = ("P0", "P1", "P2", "P3")


def _best_map_1401(pred, y, classes, slots):
    """{class: slot}: the one-to-one map between `classes` and `slots` (equal sizes) with the most images of class c
    predicted in the slot of c (Hungarian on the confusion block; images outside the block do not count)."""
    classes, slots = np.asarray(classes), np.asarray(slots)
    assert len(classes) == len(slots)
    ci, si = {int(c): j for j, c in enumerate(classes)}, {int(s): j for j, s in enumerate(slots)}
    cm = np.zeros((len(classes), len(slots)), dtype=np.int64)
    for c, s in zip(y, pred):
        if int(c) in ci and int(s) in si:
            cm[ci[int(c)], si[int(s)]] += 1
    rows, cols = linear_sum_assignment(-cm)
    return {int(classes[r]): int(slots[c]) for r, c in zip(rows, cols)}


def _rules_1401(preds, ys, luts0, base, inc):
    """The four rules of E14.01 on the same predictions. preds[i-1], ys[i-1]: predicted slot and true class of the
    test images of session i (old and novel); luts0[i-1]: the cumulative lut of the chain after session i (P0, QD15).
    A lut maps a true class to its slot; an image is correct when pred == lut[class].
    P1: block of session i matched on the predictions, blocks of earlier sessions frozen (P1 maps of those sessions).
    P2: slots 0..base-1 keep their class, every novel slot seen so far is matched again at every session (SPE).
    P3: one Hungarian over every seen slot and class (global).
    Returns ({S<i>_<rule>_<All|Old|New|Base|Disc>: accuracy in %}, {G2_new, G2_all: bool}, images predicted in slots
    not seen yet)."""
    n_slots = len(luts0[0])
    out, lut1, g2_new, g2_all, unseen = {}, np.arange(n_slots), True, True, 0
    for i, (pred, y, lut0) in enumerate(zip(preds, ys, luts0), 1):
        pred, y = np.asarray(pred).astype(int), np.asarray(y).astype(int)
        off, seen = base + (i - 1) * inc, base + i * inc
        assert y.max() < seen and (y >= off).any() and (y < off).any(), f"session {i}: unexpected test classes"
        unseen += int((pred >= seen).sum())
        lut1 = lut1.copy()
        for c, s in _best_map_1401(pred, y, np.arange(off, seen), np.arange(off, seen)).items():
            lut1[c] = s
        lut2, lut3 = np.arange(n_slots), np.arange(n_slots)
        for c, s in _best_map_1401(pred, y, np.arange(base, seen), np.arange(base, seen)).items():
            lut2[c] = s
        for c, s in _best_map_1401(pred, y, np.arange(seen), np.arange(seen)).items():
            lut3[c] = s
        sets = {"All": np.ones(len(y), bool), "Old": y < off, "New": y >= off, "Base": y < base,
                "Disc": (y >= base) & (y < off)}
        ok = {}
        for rule, lut in zip(RULES_1401, (lut0, lut1, lut2, lut3)):
            assert sorted(lut[:seen]) == list(range(seen)), f"session {i} {rule}: lut is not one-to-one on the seen classes"
            ok[rule] = pred == np.asarray(lut)[y]
            for name, sel in sets.items():
                if sel.any():
                    out[f"S{i}_{rule}_{name}"] = 100.0 * float(ok[rule][sel].mean())
        g2_new &= bool(ok["P1"][sets["New"]].sum() >= ok["P0"][sets["New"]].sum())
        g2_all &= bool(ok["P3"].sum() >= ok["P2"].sum() >= max(ok["P1"].sum(), ok["P0"].sum()))
    for rule in RULES_1401:
        out[f"Md_{rule}"] = float(np.mean([out[f"S{i}_{rule}_New"] for i in range(1, len(preds) + 1)]))
    return out, {"G2_new": g2_new, "G2_all": g2_all}, unseen


def tier2_e1401(prefix, checks_only=False):
    """E14.01 on the 15 default chains <prefix>_r<r>_s<N> (prefix cub_chain_ft or cub_chain_ft_l25): at every session
    the classifier of snapshot S<i>_13 predicts the test set of the session (as replay._confusion), and the
    predictions are scored by the four rules (_rules_1401). No SVI, no RNG, no pool labels.
    Keys: r<r>_s<N>_S<i>_<rule>_<set>, r<r>_s<N>_Md_<rule>; G1_ok, G2_ok, n_unseen, checks_ok.
    G1: P0 equals acc1 of snapshot S<i>_14 (All, Old, Novel; 1e-6) and S<i>_All, S<i>_Old, S<i>_New of metrics.json
    (0.006). G2: the orders that follow from the definitions (_rules_1401). checks_only: chain r0 s0 only, no keys."""
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.01 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1401: commit the code first (same dirty rule as scripts/exp.py)")
    out, g1, g2, unseen = {}, True, True, 0
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag = f"{prefix}_r{r}_s{N}"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a, ref = st["args"], _ref_metrics(tag)
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        preds, ys, luts0 = [], [], []
        for i, v in enumerate(views, 1):
            clf, off = _clf(v), v["offset"]
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, off + a.increment
            pr, _ = clf._predict(jnp.array(clf.pca.transform(v["test_all_x"])), _jnp(v["params1"]))
            y = np.asarray(v["true_test_all_y"]).astype(int)
            assert (y < off).sum() == len(v["test_old_x"]) and (y >= off).sum() == len(v["test1_x"]), f"{tag} S{i}: test_all is not old + novel"
            preds.append(np.asarray(pr))
            ys.append(y)
            luts0.append(np.asarray(v["lut"]))
        res, gates, n_un = _rules_1401(preds, ys, luts0, a.base, a.increment)
        ok1 = True
        for i, v in enumerate(views, 1):
            for name, snap_key, key in (("All", "All", "All"), ("Old", "Old", "Old"), ("New", "Novel", "New")):
                p0 = res[f"S{i}_P0_{name}"]
                ok1 &= abs(p0 - float(v["acc1"][snap_key])) <= 1e-6 and abs(p0 - float(ref[f"S{i}_{key}"])) <= 0.006
        g1, g2, unseen = g1 and ok1, g2 and gates["G2_new"] and gates["G2_all"], unseen + n_un
        out.update({f"r{r}_s{N}_{k}": round(val, 4) for k, val in res.items()})
        print(f"e1401 {tag}: G1 {'OK' if ok1 else 'DIFF'} | G2 new {'OK' if gates['G2_new'] else 'DIFF'}, all "
              f"{'OK' if gates['G2_all'] else 'DIFF'} | M_d " + " ".join(f"{x} {res[f'Md_{x}']:.2f}" for x in RULES_1401)
              + f" | S5_All " + " ".join(f"{x} {res[f'S5_{x}_All']:.2f}" for x in RULES_1401)
              + f" | unseen-slot predictions {n_un} ({time.time() - t0:.0f} s)", flush=True)
    chk = {"G1_ok": int(g1), "G2_ok": int(g2), "n_unseen": unseen}
    chk["checks_ok"] = int(g1 and g2)
    print(f"e1401 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1401_summary(T=0.935, B=0.376):
    """Table of E14.01 from the runs cub_e14_01_ft and cub_e14_01_ft_l25: per feature set and FT seed (mean of the 5
    chains) M_d and the session-5 scores under the four rules, and the differences Pk - P0 next to T and B of QD28.
    Descriptive: no pass/fail rule. Refuses to read a run without checks_ok = 1."""
    cols = [("M_d", "Md_{}"), ("S5_All", "S5_{}_All"), ("S5_Old", "S5_{}_Old"), ("S5_New", "S5_{}_New"),
            ("S5_Base", "S5_{}_Base"), ("S5_Disc", "S5_{}_Disc")]
    for tag in ("cub_e14_01_ft", "cub_e14_01_ft_l25"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        v = {(name, rule): np.array([[m[f"r{r}_s{N}_{pat.format(rule)}"] for N in range(5)] for r in range(3)]).mean(1)
             for name, pat in cols for rule in RULES_1401}
        print(f"\n{tag}: G1, G2 OK; predictions in slots not seen yet: {m['n_unseen']}")
        print("  " + "".join(f"{x:>10}" for x in ("", "rule", "r0", "r1", "r2", "mean", "Pk - P0")))
        for name, _ in cols:
            for rule in RULES_1401:
                x, d = v[name, rule], v[name, rule].mean() - v[name, "P0"].mean()
                print("  " + f"{name:>10}{rule:>10}" + "".join(f"{val:10.3f}" for val in x) + f"{x.mean():10.3f}"
                      + (f"{d:+10.3f}" if rule != "P0" else f"{'':>10}"))
        d1 = v["M_d", "P1"] - v["M_d", "P0"]
        print(f"  M_d(P1) - M_d(P0) by FT seed: " + " ".join(f"{x:+.3f}" for x in d1) + f" (T = {T}); "
              f"S5_Old(P3) - S5_Old(P0): {(v['S5_Old', 'P3'] - v['S5_Old', 'P0']).mean():+.3f} (B = {B})")
    print("\ndescriptive (E14.01): the main quantity of QD28 stays M_d under P0; 09e is named by P1 and compared under P1")


# ---- E14.02: CUB t5, forgetting per class cohort, loss ledger per session, error flows (pre-registered at 8e90cc8) ----
def _hung_correct_1402(cluster_ids, class_ids, k):
    """Largest number of images whose cluster is matched to their class, over the one-to-one maps of k clusters to k
    classes (both ids in 0..k-1)."""
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(class_ids).astype(int), np.asarray(cluster_ids).astype(int)), 1)
    rows, cols = linear_sum_assignment(-cm)
    return int(cm[rows, cols].sum())


def _tables_1402(sess, base, inc):
    """The three tables of E14.02 for one chain. sess[i-1] of session i: pred, pred_f (slot predicted by params1 and
    by params_f for the test images of the session), y (their true class), lut (cumulative, P0), km_test and y_test1
    (k-means cluster and true class of the novel test images), pi (pi_test: cluster -> class - offset), pool_ids and
    y_pool (k-means cluster and true class of the pool images).
    Keys (accuracies in %, flows as fractions): coh<c>_S<i>; S<i>_c09, _c10, _c11, _c14; S<i>_new_ok, _new_sess,
    _new_disc, _new_base, _new_unseen; S<i>_old_ok, _old_sess, _old_other, _old_unseen.
    Returns (keys, {G2, G3: bool})."""
    out, g2, g3 = {}, True, True
    for i, d in enumerate(sess, 1):
        pred, pred_f, y, lut = (np.asarray(d[x]).astype(int) for x in ("pred", "pred_f", "y", "lut"))
        off, seen = base + (i - 1) * inc, base + i * inc
        ok, new = pred == lut[y], y >= off
        cohort = np.where(y < base, 0, (y - base) // inc + 1)
        n_coh = 0
        for c in range(i + 1):
            sel = cohort == c
            assert sel.any(), f"session {i}: no test image of cohort {c}"
            out[f"coh{c}_S{i}"] = 100.0 * float(ok[sel].mean())
            n_coh += int(sel.sum())
        g3 &= n_coh == len(y)
        # ledger of the novel classes of the session
        yp, ids = np.asarray(d["y_pool"]).astype(int), np.asarray(d["pool_ids"]).astype(int)
        pn = yp >= off
        out[f"S{i}_c09"] = 100.0 * _hung_correct_1402(ids[pn], yp[pn] - off, inc) / int(pn.sum())
        km, y1 = np.asarray(d["km_test"]).astype(int), np.asarray(d["y_test1"]).astype(int)
        assert len(y1) == int(new.sum()) and (y1 >= off).all(), f"session {i}: novel test set"
        named = off + np.array([d["pi"][int(c)] for c in km])
        n10 = int((named == y1).sum())
        g2 &= n10 == _hung_correct_1402(km, y1 - off, inc)
        out[f"S{i}_c10"] = 100.0 * n10 / len(y1)
        out[f"S{i}_c11"] = 100.0 * float((pred_f == lut[y])[new].mean())
        out[f"S{i}_c14"] = 100.0 * float(ok[new].mean())
        # error flows of the final classifier
        flows = {"new_ok": new & ok, "new_sess": new & ~ok & (pred >= off) & (pred < seen),
                 "new_disc": new & (pred >= base) & (pred < off), "new_base": new & (pred < base),
                 "new_unseen": new & (pred >= seen),
                 "old_ok": ~new & ok, "old_sess": ~new & (pred >= off) & (pred < seen),
                 "old_other": ~new & ~ok & (pred < off), "old_unseen": ~new & (pred >= seen)}
        for side, n in (("new", int(new.sum())), ("old", int((~new).sum()))):
            tot = 0
            for name, sel in flows.items():
                if name.startswith(side):
                    out[f"S{i}_{name}"] = float(sel.sum()) / n
                    tot += int(sel.sum())
            g3 &= tot == n
    return out, {"G2": g2, "G3": g3}


def tier2_e1402(prefix, checks_only=False):
    """E14.02 on the 15 default chains <prefix>_r<r>_s<N>: per session the predictions of params1 and params_f on the
    test set (as tier2_e1401), the k-means clusters of the novel test images and of the pool, then _tables_1402.
    Reads the true pool labels, for scoring c09 only. No SVI, no RNG.
    Keys: r<r>_s<N>_<key of _tables_1402>, r<r>_s<N>_coh0_S0 (S0_All of metrics.json); G1_ok, G2_ok, G3_ok, checks_ok.
    G1: All, Old, New recomputed from the cohorts and c14 equal acc1 of snapshot S<i>_14 (1e-6) and metrics.json
    (0.006); coh0_S<i> equals S<i>_S0test. checks_only: chain r0 s0 only, no keys."""
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.02 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1402: commit the code first (same dirty rule as scripts/exp.py)")
    out, gates = {}, {"G1": True, "G2": True, "G3": True}
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag = f"{prefix}_r{r}_s{N}"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a, ref = st["args"], _ref_metrics(tag)
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        sess = []
        for i, v in enumerate(views, 1):
            clf, off = _clf(v), v["offset"]
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, off + a.increment
            z = jnp.array(clf.pca.transform(v["test_all_x"]))
            pr, _ = clf._predict(z, _jnp(v["params1"]))
            pf, _ = clf._predict(z, _jnp(v["params_f"]))
            sess.append(dict(pred=np.asarray(pr), pred_f=np.asarray(pf), y=np.asarray(v["true_test_all_y"]).astype(int),
                             lut=np.asarray(v["lut"]), km_test=np.asarray(v["cluster"].predict_raw(v["test1_x"])),
                             y_test1=np.asarray(v["true_test1_y"]).astype(int), pi=v["pi_test"],
                             pool_ids=np.asarray(v["ids"]), y_pool=np.asarray(v["true_pool_y"]).astype(int)))
        res, g = _tables_1402(sess, a.base, a.increment)
        ok1 = True
        for i, (v, d) in enumerate(zip(views, sess), 1):
            y, off = d["y"], v["offset"]
            hit = d["pred"] == d["lut"][y]
            mine = {"All": 100.0 * hit.mean(), "Old": 100.0 * hit[y < off].mean(), "Novel": res[f"S{i}_c14"], "S0": res[f"coh0_S{i}"]}
            for snap_key, key in (("All", "All"), ("Old", "Old"), ("Novel", "New"), ("S0", "S0test")):
                ok1 &= abs(mine[snap_key] - float(v["acc1"][snap_key])) <= 1e-6 and abs(mine[snap_key] - float(ref[f"S{i}_{key}"])) <= 0.006
        res["coh0_S0"] = float(ref["S0_All"])
        gates = {"G1": gates["G1"] and ok1, "G2": gates["G2"] and g["G2"], "G3": gates["G3"] and g["G3"]}
        out.update({f"r{r}_s{N}_{k}": round(val, 4) for k, val in res.items()})
        forget = np.mean([res[f"coh{c}_S{c}"] - res[f"coh{c}_S5"] for c in range(1, 5)])
        print(f"e1402 {tag}: G1 {'OK' if ok1 else 'DIFF'} | G2 {'OK' if g['G2'] else 'DIFF'} | G3 {'OK' if g['G3'] else 'DIFF'}"
              f" | mean over sessions: c09 {np.mean([res[f'S{i}_c09'] for i in range(1, 6)]):.2f}"
              f" c10 {np.mean([res[f'S{i}_c10'] for i in range(1, 6)]):.2f} c11 {np.mean([res[f'S{i}_c11'] for i in range(1, 6)]):.2f}"
              f" c14 {np.mean([res[f'S{i}_c14'] for i in range(1, 6)]):.2f} | forgetting of cohorts 1-4 {forget:+.2f}"
              f" ({time.time() - t0:.0f} s)", flush=True)
    chk = {f"{k}_ok": int(v) for k, v in gates.items()}
    chk["checks_ok"] = int(all(gates.values()))
    print(f"e1402 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1402_summary(T=0.935, B=0.376):
    """Tables of E14.02 from the runs cub_e14_02_ft and cub_e14_02_ft_l25. Every number is first the mean of the 5
    chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). Descriptive: no pass/fail rule.
    Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e14_02_ft_l25", "cub_e14_02_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        ft = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)]).mean(1)
        line = lambda name, x: f"  {name:<22}" + "".join(f"{val:9.3f}" for val in x) + f"{x.mean():9.3f}{x.std(ddof=1):9.3f}"
        head = f"  {'':<22}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        print(f"\n{tag}: G1, G2, G3 OK")
        print("  table 1: accuracy of cohort c (rows) after session i (columns), mean over the FT seeds")
        print(f"  {'':<10}" + "".join(f"{'S' + str(i):>9}" for i in range(6)))
        for c in range(6):
            print(f"  {'cohort ' + str(c):<10}" + "".join(f"{ft(f'coh{c}_S{i}').mean():9.3f}" if i >= c else f"{'':>9}" for i in range(6)))
        print("  forgetting = cell (c, c) - cell (c, 5)\n" + head)
        fg = [ft(f"coh{c}_S{c}") - ft(f"coh{c}_S5") for c in range(5)]
        for c in range(5):
            print(line(f"cohort {c}", fg[c]))
        f14 = np.mean(fg[1:], axis=0)
        print(line("cohorts 1-4 (mean)", f14))
        print("  table 2: ledger of the novel classes, mean over the FT seeds")
        print(f"  {'':<10}" + "".join(f"{x:>9}" for x in ("c09", "c10", "c11", "c14", "L_10", "L_F", "L_S")))
        col = {k: [ft(f"S{i}_{k}") for i in range(1, 6)] for k in ("c09", "c10", "c11", "c14")}
        for i in range(5):
            c9, c10, c11, c14 = (col[k][i].mean() for k in ("c09", "c10", "c11", "c14"))
            print(f"  {'S' + str(i + 1):<10}" + "".join(f"{x:9.3f}" for x in (c9, c10, c11, c14, c10 - c14, c10 - c11, c11 - c14)))
        avg = {k: np.mean(col[k], axis=0) for k in col}
        print("  mean over the 5 sessions\n" + head)
        for name, x in (("c09", avg["c09"]), ("c10", avg["c10"]), ("c11", avg["c11"]), ("c14 (= M_d)", avg["c14"]),
                        ("L_10 = c10 - c14", avg["c10"] - avg["c14"]), ("L_F = c10 - c11", avg["c10"] - avg["c11"]),
                        ("L_S = c11 - c14", avg["c11"] - avg["c14"]), ("c09 - c10", avg["c09"] - avg["c10"])):
            print(line(name, x))
        print("  table 3: error flows of the final classifier, fraction of the set, mean over the 5 sessions\n" + head)
        for k in ("new_ok", "new_sess", "new_disc", "new_base", "new_unseen", "old_ok", "old_sess", "old_other", "old_unseen"):
            print(line(k, np.mean([ft(f"S{i}_{k}") for i in range(1, 6)], axis=0)))
        l10 = avg["c10"] - avg["c14"]
        print(f"  readings fixed in advance: forgetting of cohorts 1-4 = {f14.mean():+.3f} (B = {B}): "
              f"{'above B' if f14.mean() > B else 'below B'}; L_10 = {l10.mean():+.3f} (T = {T}): "
              f"{'above T' if l10.mean() > T else 'not above T'}; c10 = {avg['c10'].mean():.3f}")
    print("\ndescriptive (E14.02): tables under P0 (QD15); c10 knows which test images are novel; c11 is a counterfactual")


# ---- E14.02, amendment of 2026-11-16: who takes the forgotten images (tables 4 and 5) ----
def _tables_1402b(sess, base, inc, s0_all):
    """Tables 4 and 5 of E14.02 for one chain. sess[i-1] of session i: pred, pred_prev (slot predicted for the test
    images of the session by the classifier after session i and after session i-1; i = 1: the S0 classifier), y, lut
    (cumulative, P0), pool_ids, y_pool, keep (filter of step 12, per pool image). s0_all: S0 test accuracy in %.
    Table 4, old test images of cohort c at session i (% of the cohort): tr_c<c>_S<i>_lost_new (right before, now in
    a slot of session i), _lost_other (right before, now wrong elsewhere), _gained (wrong before, right now).
    Table 5, slots of session i ranked by the share of old images in their k-means cluster on the pool (S<i>_cap_top<k>)
    or among the kept images of the cluster (S<i>_capk_top<k>): share of the old test images predicted in the slots
    of the session that falls in the k highest-ranked slots, k = 1, 2, 5, 10. Context: S<i>_cap_n (old test images in
    the slots of the session, % of the old test images), S<i>_pool_old, S<i>_keep_old (old share of the pool and of
    the kept images), S<i>_f_old_kept, S<i>_f_new_kept (share of the old and of the novel pool images kept).
    Returns (keys, {G4: bool, pure: bool}); pure: no _lost_other and no _gained anywhere."""
    out, g4, pure, before = {}, True, True, {0: s0_all}
    for i, d in enumerate(sess, 1):
        pred, prev, y, lut = (np.asarray(d[x]).astype(int) for x in ("pred", "pred_prev", "y", "lut"))
        off, seen = base + (i - 1) * inc, base + i * inc
        ok, ok_prev, old = pred == lut[y], prev == lut[y], y < off
        cohort = np.where(y < base, 0, (y - base) // inc + 1)
        for c in range(i):
            sel = cohort == c
            g4 &= abs(100.0 * float(ok_prev[sel].mean()) - before[c]) <= 1e-6
            lost = sel & ok_prev & ~ok
            new_slot = (pred >= off) & (pred < seen)
            vals = {"lost_new": lost & new_slot, "lost_other": lost & ~new_slot, "gained": sel & ~ok_prev & ok}
            for name, m in vals.items():
                out[f"tr_c{c}_S{i}_{name}"] = 100.0 * float(m.sum()) / int(sel.sum())
            pure &= int(vals["lost_other"].sum()) == 0 and int(vals["gained"].sum()) == 0
        before = {c: 100.0 * float(ok[cohort == c].mean()) for c in range(i + 1)}
        # table 5
        yp, ids, keep = np.asarray(d["y_pool"]).astype(int), np.asarray(d["pool_ids"]).astype(int), np.asarray(d["keep"]).astype(bool)
        p_old = yp < off
        cap = np.bincount(pred[old & (pred >= off) & (pred < seen)] - off, minlength=inc).astype(float)
        assert cap.sum() > 0 and len(keep) == len(yp), f"session {i}: no old test image in the new slots, or keep of another length"
        share = np.array([p_old[ids == c].mean() if (ids == c).any() else 0.0 for c in range(inc)])
        share_k = np.array([p_old[(ids == c) & keep].mean() if ((ids == c) & keep).any() else 0.0 for c in range(inc)])
        for name, sh in (("cap", share), ("capk", share_k)):
            order = np.argsort(-sh, kind="stable")
            for k in (1, 2, 5, 10):
                out[f"S{i}_{name}_top{k}"] = float(cap[order[:k]].sum() / cap.sum())
        out[f"S{i}_cap_n"] = 100.0 * float(cap.sum()) / int(old.sum())
        out[f"S{i}_pool_old"], out[f"S{i}_keep_old"] = float(p_old.mean()), float(p_old[keep].mean())
        out[f"S{i}_f_old_kept"], out[f"S{i}_f_new_kept"] = float(keep[p_old].mean()), float(keep[~p_old].mean())
    return out, {"G4": g4, "pure": pure}


def tier2_e1402b(prefix, checks_only=False):
    """Amendment of E14.02 on the 15 default chains <prefix>_r<r>_s<N>: at session i the classifier after session i
    and the classifier after session i-1 (i = 1: params0) predict the same test images, then _tables_1402b. Reads the
    true pool labels, for table 5 only. No SVI, no RNG.
    Keys: r<r>_s<N>_<key of _tables_1402b>; G1_ok, G4_ok, pure, checks_ok (= G1 and G4; pure is a result, not a gate).
    G1: All of the classifier after session i equals acc1 of snapshot S<i>_14 (1e-6). G4: the previous classifier
    scores on the images of session i what it scored on its own test set. checks_only: chain r0 s0 only, no keys."""
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.02 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1402b: commit the code first (same dirty rule as scripts/exp.py)")
    out, g1, g4, pure = {}, True, True, True
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag = f"{prefix}_r{r}_s{N}"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a = st["args"]
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        sess, ok1 = [], True
        for i, v in enumerate(views, 1):
            clf, off = _clf(v), v["offset"]
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, off + a.increment
            z = jnp.array(clf.pca.transform(v["test_all_x"]))
            pr, _ = clf._predict(z, _jnp(v["params1"]))
            pp, _ = clf._predict(z, _jnp(st["params0"] if i == 1 else views[i - 2]["params1"]))
            d = dict(pred=np.asarray(pr), pred_prev=np.asarray(pp), y=np.asarray(v["true_test_all_y"]).astype(int),
                     lut=np.asarray(v["lut"]), pool_ids=np.asarray(v["ids"]), y_pool=np.asarray(v["true_pool_y"]).astype(int),
                     keep=np.asarray(v["keep"]))
            ok1 &= abs(100.0 * float((d["pred"] == d["lut"][d["y"]]).mean()) - float(v["acc1"]["All"])) <= 1e-6
            sess.append(d)
        res, g = _tables_1402b(sess, a.base, a.increment, float(st["acc0"]["All"]))
        g1, g4, pure = g1 and ok1, g4 and g["G4"], pure and g["pure"]
        out.update({f"r{r}_s{N}_{k}": round(val, 4) for k, val in res.items()})
        print(f"e1402b {tag}: G1 {'OK' if ok1 else 'DIFF'} | G4 {'OK' if g['G4'] else 'DIFF'} | pure {'yes' if g['pure'] else 'NO'}"
              f" | lost to the new slots, cohort 0: " + " ".join(f"{res[f'tr_c0_S{i}_lost_new']:.2f}" for i in range(1, 6))
              + f" | top-5 share " + " ".join(f"{res[f'S{i}_cap_top5']:.2f}" for i in range(1, 6)) + f" ({time.time() - t0:.0f} s)", flush=True)
    chk = {"G1_ok": int(g1), "G4_ok": int(g4), "pure": int(pure), "checks_ok": int(g1 and g4)}
    print(f"e1402b {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'}; pure {'yes' if pure else 'NO'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1402b_summary():
    """Tables 4 and 5 of E14.02 from the runs cub_e14_02b_ft_l25 and cub_e14_02b_ft. Every number is first the mean of
    the 5 chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). Descriptive.
    Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e14_02b_ft_l25", "cub_e14_02b_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        ft = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)]).mean(1)
        line = lambda name, x: f"  {name:<26}" + "".join(f"{val:9.3f}" for val in x) + f"{x.mean():9.3f}{x.std(ddof=1):9.3f}"
        head = f"  {'':<26}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        pairs = [(c, i) for i in range(1, 6) for c in range(i)]
        other = sum(ft(f"tr_c{c}_S{i}_lost_other").sum() + ft(f"tr_c{c}_S{i}_gained").sum() for c, i in pairs)
        print(f"\n{tag}: G1, G4 OK; pure = {m['pure']} (sum of every lost_other and gained cell: {other:.4f})")
        print("  table 4: % of the test images of cohort c (rows) that were right after session i-1 and are in a slot of session i after session i (columns)")
        print(f"  {'':<10}" + "".join(f"{'S' + str(i):>9}" for i in range(1, 6)))
        for c in range(5):
            print(f"  {'cohort ' + str(c):<10}" + "".join(f"{ft(f'tr_c{c}_S{i}_lost_new').mean():9.3f}" if i > c else f"{'':>9}" for i in range(1, 6)))
        print(head)
        print(line("first session after (c >= 1)", np.mean([ft(f"tr_c{c}_S{c + 1}_lost_new") for c in range(1, 5)], axis=0)))
        print(line("later sessions (c >= 1)", np.mean([ft(f"tr_c{c}_S{i}_lost_new") for c, i in pairs if c >= 1 and i > c + 1], axis=0)))
        print(line("cohort 0, per session", np.mean([ft(f"tr_c0_S{i}_lost_new") for i in range(1, 6)], axis=0)))
        print("  table 5: share of the old test images taken by the slots of the session that falls in the k slots whose pool cluster")
        print("  has the largest share of old images (cap) or of old images among the kept ones (capk); 20 slots, uniform = k / 20")
        print(f"  {'':<10}" + "".join(f"{x:>9}" for x in ("top1", "top2", "top5", "top10", "k_top5", "taken %", "pool_old", "keep_old", "old_kept", "new_kept")))
        keys = ("cap_top1", "cap_top2", "cap_top5", "cap_top10", "capk_top5", "cap_n", "pool_old", "keep_old", "f_old_kept", "f_new_kept")
        for i in range(1, 6):
            print(f"  {'S' + str(i):<10}" + "".join(f"{ft(f'S{i}_{k}').mean():9.3f}" for k in keys))
        print("  mean over the 5 sessions\n" + head)
        for k in keys:
            print(line(k, np.mean([ft(f"S{i}_{k}") for i in range(1, 6)], axis=0)))
    print("\ndescriptive (E14.02, amendment): table 5 is a correlation; the intervention is the oracle ladder (step D)")


Q_1403 = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98)


def _parts_1403(L, pred, y, lut, off, inc, base):
    """Part A and D of E14.03 for one session. L: log-densities (images x slots, float64) of params1 on the test set of
    the session; pred: the slot _predict chose; y: true class; lut: cumulative lut (P0). Returns (keys, G3 ok)."""
    new, tgt = y >= off, lut[y]
    out = {}
    g3 = bool(np.array_equal(L.argmax(1), pred))                     # mask open on every slot = the pipeline
    masks = {"c14": None, "nobase": (0, base), "nodisc": (base, off), "g20": None}
    for name, cut in masks.items():
        M = L.copy()
        if name == "g20":
            keep = np.zeros(L.shape[1], bool)
            keep[off:off + inc] = True
            M[:, ~keep] = -np.inf
        elif cut is not None:
            M[:, cut[0]:cut[1]] = -np.inf
        out[name] = 100.0 * float((M.argmax(1) == tgt)[new].mean())
    m = L[:, off:off + inc].max(1) - L[:, :off].max(1)              # margin new block - old slots, no labels
    bs = [0.0] + [float(-np.quantile(m, q)) for q in Q_1403]
    nb = np.zeros(L.shape[1])
    nb[off:off + inc] = 1.0
    for j, b in enumerate(bs):
        pr = (L + b * nb).argmax(1)
        out[f"D{j}_b"] = b
        out[f"D{j}_New"] = 100.0 * float((pr == tgt)[new].mean())
        out[f"D{j}_Old"] = 100.0 * float((pr == tgt)[~new].mean())
    return out, g3


def _gain_1403(d, n_pts, drop):
    """New-class gain at an old-class drop of `drop` points along b >= 0 (New, Old monotone in b), linear between grid
    points. Returns (gain, reached): reached False when the largest b on the grid drops Old by less than `drop`."""
    pts = sorted((d[f"D{j}_b"], d[f"D{j}_New"], d[f"D{j}_Old"]) for j in range(n_pts) if d[f"D{j}_b"] >= 0)
    n0, o0 = d["D0_New"], d["D0_Old"]
    prev = (0.0, 0.0)
    for _, n, o in pts:
        dn, do = n - n0, o0 - o
        if do >= drop:
            if do == prev[1]:
                return dn, True
            return prev[0] + (dn - prev[0]) * (drop - prev[1]) / (do - prev[1]), True
        prev = (dn, do)
    return prev[0], False


def _mech_1403(z, L, pred, y, lut, off, params, cache):
    """Part B of E14.03: novel test images that the pipeline puts in an old slot o; n = their true slot.
    D_det = -1/2 (logdet S_o - logdet S_n), D_maha = -1/2 (d2_o - d2_n), numpy float64 (Cholesky); G4: their sum equals
    L[o] - L[n] within 1e-3 x max(1, |L[o]|). cache: slot -> (mean, cholesky, logdet). Returns (rows, G4 ok)."""
    def slot(k):
        if k not in cache:
            mu = np.asarray(params["class_means"][k], np.float64)
            C = np.linalg.cholesky(np.asarray(params["class_covs"][k], np.float64))
            cache[k] = (mu, C, 2.0 * float(np.log(np.diag(C)).sum()))
        return cache[k]

    def d2(x, k):
        mu, C, _ = slot(k)
        w = np.linalg.solve(C, x - mu)
        return float(w @ w)
    rows, g4 = [], True
    for idx in np.flatnonzero((y >= off) & (pred < off)):
        o, n, x = int(pred[idx]), int(lut[y[idx]]), z[idx]
        dd = -0.5 * (slot(o)[2] - slot(n)[2])
        dm = -0.5 * (d2(x, o) - d2(x, n))
        g4 &= abs(dd + dm - (L[idx, o] - L[idx, n])) <= 1e-3 * max(1.0, abs(L[idx, o]))
        rows.append((o, dd, dm))
    return rows, g4


def tier2_e1403(prefix, checks_only=False):
    """E14.03 (pre-registration 3e7e2b3) on the 15 default chains <prefix>_r<r>_s<N>. Per session: log-densities of
    params1 (S<i>_13) on the test set; part A (argmax over slot sets), B (log det vs Mahalanobis on the novel images
    taken by old slots), C (nearest-centroid pipeline on the raw features), D (bias b on the 20 slots of the session).
    No SVI, no RNG; true labels for scoring and for the S0 class means only. Gates G1-G5 of the pre-registration.
    G2 compares with new_disc + new_base of the E14.02 run, stored rounded to 4 decimals: tolerance 1e-6 + 100 x 1e-4.
    Keys: r<r>_s<N>_<key>; G1_ok ... G5_ok, checks_ok. checks_only: chain r0 s0 only, no keys."""
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.03 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1403: commit the code first (same dirty rule as scripts/exp.py)")
    e02 = _ref_metrics("cub_e14_02_" + prefix[len("cub_chain_"):])
    assert e02.get("checks_ok") == 1, "E14.02 run of this feature set: checks failed"
    out, gates = {}, {f"G{k}": True for k in range(1, 6)}
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag, pre = f"{prefix}_r{r}_s{N}", f"r{r}_s{N}_"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a, ref = st["args"], _ref_metrics(tag)
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        base, inc, res, g = a.base, a.increment, {}, {f"G{k}": True for k in range(1, 6)}
        s0y = np.asarray(st["s0_y"]).astype(int)
        g["G5"] &= bool(np.array_equal(np.bincount(s0y, minlength=base)[:base], np.full(base, 25)) and s0y.max() < base)
        cent = np.stack([np.asarray(st["s0_x"], np.float64)[s0y == k].mean(0) for k in range(base)])
        y0 = np.asarray(st["true_test0_y"]).astype(int)
        x0 = np.asarray(st["test0_x"], np.float64)
        near = lambda x, c: ((x ** 2).sum(1)[:, None] - 2.0 * x @ c.T + (c ** 2).sum(1)[None, :]).argmin(1)   # Euclid
        res["C_S0_S0test"] = 100.0 * float((near(x0, cent) == y0).mean())
        B = {"base": [], "disc": []}
        for i, v in enumerate(views, 1):
            off = v["offset"]
            clf = _clf(v)
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, off + inc
            z = np.asarray(clf.pca.transform(v["test_all_x"]), np.float64)
            pr, lp = clf._predict(jnp.array(clf.pca.transform(v["test_all_x"])), _jnp(v["params1"]))
            pr, L = np.asarray(pr).astype(int), np.asarray(lp, np.float64)
            y, lut = np.asarray(v["true_test_all_y"]).astype(int), np.asarray(v["lut"])
            new = y >= off
            # part A, D
            ad, ok3 = _parts_1403(L, pr, y, lut, off, inc, base)
            g["G3"] &= ok3
            km, y1 = np.asarray(v["cluster"].predict_raw(v["test1_x"])).astype(int), np.asarray(v["true_test1_y"]).astype(int)
            c10 = 100.0 * float((off + np.array([v["pi_test"][int(c)] for c in km]) == y1).mean())
            old_acc = 100.0 * float((pr == lut[y])[~new].mean())
            g["G1"] &= abs(ad["c14"] - float(v["acc1"]["Novel"])) <= 1e-6 and abs(ad["c14"] - float(ref[f"S{i}_New"])) <= 0.006 \
                and abs(old_acc - float(v["acc1"]["Old"])) <= 1e-6 and abs(old_acc - float(ref[f"S{i}_Old"])) <= 0.006 \
                and abs(c10 - float(e02[f"{pre}S{i}_c10"])) <= 1e-4
            bound = 100.0 * (float(e02[f"{pre}S{i}_new_disc"]) + float(e02[f"{pre}S{i}_new_base"]))
            g["G2"] &= ad["g20"] - ad["c14"] <= bound + 1e-6 + 100 * 1e-4
            res.update({f"S{i}_{k}": val for k, val in ad.items()})
            res[f"S{i}_c10"] = c10
            gb, rb = _gain_1403(ad, len(Q_1403) + 1, 0.376)
            g1p, r1p = _gain_1403(ad, len(Q_1403) + 1, 1.0)
            res.update({f"S{i}_D_gainB": gb, f"S{i}_D_reachB": int(rb), f"S{i}_D_gain1": g1p, f"S{i}_D_reach1": int(r1p)})
            # part B
            rows, ok4 = _mech_1403(z, L, pr, y, lut, off, v["params1"], {})
            g["G4"] &= ok4
            for o, dd, dm in rows:
                B["base" if o < base else "disc"].append((dd, dm))
            ld = lambda lo, hi: float(np.mean([np.linalg.slogdet(np.asarray(v["params1"]["class_covs"][k], np.float64))[1] for k in range(lo, hi)]))
            res[f"S{i}_ld_base"], res[f"S{i}_ld_sess"] = ld(0, base), ld(off, off + inc)
            if off > base:
                res[f"S{i}_ld_disc"] = ld(base, off)
            # part C: nearest centroid over the slots seen so far (raw features)
            cent = np.concatenate([cent, np.asarray(v["cluster"].model.cluster_centers_, np.float64)])
            assert len(cent) == off + inc, f"{tag} S{i}: {len(cent)} centroids"
            x1 = np.asarray(v["test1_x"], np.float64)
            g["G5"] &= bool(np.array_equal(near(x1, cent[off:]), km))
            pc = near(np.asarray(v["test_all_x"], np.float64), cent)
            okc = pc == lut[y]
            cohort = np.where(y < base, 0, (y - base) // inc + 1)
            res.update({f"S{i}_C_All": 100.0 * float(okc.mean()), f"S{i}_C_Old": 100.0 * float(okc[~new].mean()),
                        f"S{i}_C_New": 100.0 * float(okc[new].mean()), f"S{i}_C_S0test": 100.0 * float(okc[y < base].mean())})
            for c in range(i + 1):
                res[f"S{i}_C_coh{c}"] = 100.0 * float(okc[cohort == c].mean())
        for grp, rows in B.items():
            res[f"B_{grp}_n"] = len(rows)
            dd, dm = np.array(rows).T if rows else (np.array([np.nan]), np.array([np.nan]))
            res.update({f"B_{grp}_med_det": float(np.median(dd)), f"B_{grp}_med_maha": float(np.median(dm)),
                        f"B_{grp}_pos_det": float((dd > 0).mean()) if rows else float("nan"),
                        f"B_{grp}_det_dom": float((np.abs(dd) > np.abs(dm)).mean()) if rows else float("nan")})
        for k in gates:
            gates[k] = gates[k] and g[k]
        out.update({pre + k: round(val, 4) if isinstance(val, float) else val for k, val in res.items()})
        mean = lambda k: np.mean([res[f"S{i}_{k}"] for i in range(1, 6)])
        print(f"e1403 {tag}: " + " ".join(f"{k} {'OK' if g[k] else 'DIFF'}" for k in g)
              + f" | c10 {mean('c10'):.2f} g20 {mean('g20'):.2f} nobase {mean('nobase'):.2f} nodisc {mean('nodisc'):.2f}"
              f" c14 {mean('c14'):.2f} | L_comp {mean('g20') - mean('c14'):+.2f} L_10 {mean('c10') - mean('c14'):+.2f}"
              f" | B base n {res['B_base_n']} disc n {res['B_disc_n']} | C M_d {mean('C_New'):.2f}"
              f" | D gain at B {mean('D_gainB'):+.2f} ({time.time() - t0:.0f} s)", flush=True)
    chk = {f"{k}_ok": int(v) for k, v in gates.items()}
    chk["checks_ok"] = int(all(gates.values()))
    print(f"e1403 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1403_summary(T=0.935, B=0.376):
    """Tables of E14.03 from the runs cub_e14_03_ft_l25 and cub_e14_03_ft. Every number is first the mean of the 5
    chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). Part A: reading rule of the
    pre-registration; B, C, D: report only. Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e14_03_ft_l25", "cub_e14_03_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        per = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)])
        ses = lambda k: np.mean([per(f"S{i}_{k}") for i in range(1, 6)], axis=0)        # chains x mean over sessions
        line = lambda name, x: f"  {name:<24}" + "".join(f"{val:9.3f}" for val in x.mean(1)) + f"{x.mean(1).mean():9.3f}{x.mean(1).std(ddof=1):9.3f}"
        head = f"  {'':<24}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        print(f"\n{tag}: G1-G5 OK\n  part A, novel test images, mean over the 5 sessions\n{head}")
        for k in ("c10", "g20", "nobase", "nodisc", "c14"):
            print(line(k, ses(k)))
        L10, Lc, Lb, Ld = ses("c10") - ses("c14"), ses("g20") - ses("c14"), ses("nobase") - ses("c14"), ses("nodisc") - ses("c14")
        for name, x in (("L_10 = c10 - c14", L10), ("L_comp = g20 - c14", Lc), ("L_base", Lb), ("L_disc", Ld)):
            print(line(name, x))
        print("  L_comp per chain (r0 s0..s4 | r1 | r2): " + " | ".join(" ".join(f"{v:.2f}" for v in row) for row in Lc))
        lc, l10 = Lc.mean(1).mean(), L10.mean(1).mean()
        verdict = ("old-slot competition at prediction is the main target of E09.08" if lc > T and lc >= 0.5 * l10 else
                   "estimation at steps 11-13 is the target (as E14.02)" if lc <= T else "both")
        print(f"  reading rule: L_comp {lc:.3f} vs T {T}, 1/2 L_10 {0.5 * l10:.3f} -> {verdict}")
        print("  part B, novel test images taken by old slots, pooled over the 5 sessions of a chain")
        for grp in ("base", "disc"):
            print(head)
            print(line(f"{grp}: n", per(f"B_{grp}_n").astype(float)))
            for k in ("med_det", "med_maha", "pos_det", "det_dom"):
                print(line(f"{grp}: {k}", per(f"B_{grp}_{k}")))
        print("  mean log det of the slot covariances, per session (base / discovered / session)")
        for i in range(1, 6):
            print(f"  S{i}: {per(f'S{i}_ld_base').mean():9.3f} / " + (f"{per(f'S{i}_ld_disc').mean():9.3f}" if i > 1 else f"{'-':>9}")
                  + f" / {per(f'S{i}_ld_sess').mean():9.3f}")
        print("  part C, nearest-centroid pipeline (raw features); confounded: space, estimator and training data differ")
        print(head)
        print(line("C M_d (mean of New)", ses("C_New")))
        print(line("pipeline M_d (c14)", ses("c14")))
        for k in ("All", "Old", "S0test"):
            print(line(f"C S5_{k}", per(f"S5_C_{k}")))
        print(line("C M_f proxy", per("C_S0_S0test") - per("S5_C_S0test")))
        forget = np.mean([per(f"S{c}_C_coh{c}") - per(f"S5_C_coh{c}") for c in range(1, 5)], axis=0)
        print(line("C forgetting, cohorts 1-4", forget))
        print("  part D, bias b on the 20 slots of the session (b >= 0), mean over the 5 sessions; not a method")
        print(head)
        print(line("New gain at Old drop B", ses("D_gainB")))
        print(line("New gain at Old drop 1", ses("D_gain1")))
        print(f"  sessions where the grid does not reach the drop: B {int((1 - ses('D_reachB')).sum() * 5)}, 1 point {int((1 - ses('D_reach1')).sum() * 5)} (of 75)")
    print("\nE14.03: part A has the pre-registered reading rule; B, C, D are report only")


LADDER_1405 = ("M0", "Mk", "Mc", "M1", "M2", "M3")


def _ladder_1405(v, off, inc, base, lut):
    """Means of the 20 slots of session i for the rungs of E14.05 (float64, slot order off..off+inc-1), the shared
    covariance, and the gate parts. Mk: kept pool images by cluster id (the set of step 13); Mc: Mk without old
    images (pool labels); M1: all pool images of the class named by the slot (lut, P0). Empty cluster: the M0 mean."""
    p1 = v["params1"]
    m0 = np.asarray(p1["class_means"], np.float64)[off:off + inc]
    zp = np.asarray(v["z_pool"], np.float64)
    ids, k = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)
    y = np.asarray(v["true_pool_y"]).astype(int)
    cls = np.arange(off, off + inc)
    inv = {int(lut[c]): int(c) for c in cls}                  # slot -> true class
    g2 = bool(sorted(inv) == list(range(off, off + inc)) and all((y == c).sum() == 25 for c in cls)
              and int(sum((k & (ids == c)).sum() for c in range(inc))) == int(k.sum()))
    mk, mc, m1, g3, n_min, empty_k, empty_c = m0.copy(), m0.copy(), m0.copy(), True, 10 ** 9, 0, 0
    for c in range(inc):
        sk = k & (ids == c)
        sc = sk & (y >= off)
        g3 &= bool(np.all(sk[sc]))
        n_min = min(n_min, int(sk.sum()))
        if sk.any():
            mk[c] = zp[sk].mean(0)
        else:
            empty_k += 1
        if sc.any():
            mc[c] = zp[sc].mean(0)
        else:
            empty_c += 1
        if sk.any() and sc.sum() == sk.sum():
            g3 &= bool(np.array_equal(mc[c], mk[c]))
        m1[c] = zp[y == inv[off + c]].mean(0)
    shared = np.asarray(p1["class_covs"], np.float64)[:base].mean(0)
    try:
        np.linalg.cholesky(shared)
        g4 = True
    except np.linalg.LinAlgError:
        g4 = False
    return {"M0": m0, "Mk": mk, "Mc": mc, "M1": m1}, shared, g2, g3, g4, n_min, empty_k, empty_c


def _dist_1405(p1, off, inc, m0, mk):
    """Per new slot, with the covariance of that slot: Mahalanobis distance from the M0 mean to the Mk mean, and from
    the Mk mean to the nearest old-slot mean. Returns the two medians over the 20 slots."""
    means = np.asarray(p1["class_means"], np.float64)
    covs = np.asarray(p1["class_covs"], np.float64)
    d0, dn = [], []
    for c in range(inc):
        C = np.linalg.cholesky(covs[off + c])
        w = np.linalg.solve(C, (mk[c] - m0[c])[:, None])
        d0.append(float(np.sqrt((w ** 2).sum())))
        W = np.linalg.solve(C, (means[:off] - mk[c]).T)
        dn.append(float(np.sqrt((W ** 2).sum(0)).min()))
    return float(np.median(d0)), float(np.median(dn))


def tier2_e1405(prefix, checks_only=False):
    """E14.05 (pre-registration 45f7ffd) on the 15 default chains <prefix>_r<r>_s<N>. Per session: params1 (S<i>_13)
    with the means of the 20 slots of the session replaced (M0 itself, Mk, Mc, M1; M2 = M0 means and M3 = M1 means
    with one shared covariance for every slot), log-densities of _predict on the test set; New, Old, g20 per rung.
    The parameters are cut to the slots seen so far (off + 20): argmax over seen slots; at M0 this is the pipeline,
    since no test image goes to an unseen slot (E14.02), and G1 checks it. No SVI, no RNG; pool labels in Mc, M1, M3
    only; test labels for scoring only. Gates G1-G4 of the pre-registration; the progress line prints the gates only.
    Keys: r<r>_s<N>_<key>; G1_ok ... G4_ok, checks_ok. checks_only: chain r0 s0 only, no keys."""
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.05 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1405: commit the code first (same dirty rule as scripts/exp.py)")
    e03 = _ref_metrics("cub_e14_03_" + prefix[len("cub_chain_"):])
    assert e03.get("checks_ok") == 1, "E14.03 run of this feature set: checks failed"
    out, gates = {}, {f"G{k}": True for k in range(1, 5)}
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag, pre = f"{prefix}_r{r}_s{N}", f"r{r}_s{N}_"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a = st["args"]
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        base, inc, res, g = a.base, a.increment, {}, {f"G{k}": True for k in range(1, 5)}
        for i, v in enumerate(views, 1):
            off = v["offset"]
            n = off + inc
            clf = _clf(v)
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, n
            zt = jnp.array(clf.pca.transform(v["test_all_x"]))
            y, lut = np.asarray(v["true_test_all_y"]).astype(int), np.asarray(v["lut"])
            new, tgt = y >= off, lut[y]
            means, shared, g2, g3, g4, n_min, ek, ec = _ladder_1405(v, off, inc, base, lut)
            g["G2"] &= g2
            g["G3"] &= g3
            g["G4"] &= g4
            p1 = v["params1"]
            dt_m, dt_c = np.asarray(p1["class_means"]).dtype, np.asarray(p1["class_covs"]).dtype
            for rung in LADDER_1405:
                mu = np.asarray(p1["class_means"], np.float64)[:n].copy()
                mu[off:n] = means["M1" if rung == "M3" else "M0" if rung == "M2" else rung]
                cov = np.asarray(p1["class_covs"])[:n]
                if rung in ("M2", "M3"):
                    cov = np.broadcast_to(shared, cov.shape)
                params = {"class_means": mu.astype(dt_m), "class_covs": np.ascontiguousarray(cov, dtype=dt_c)}
                _, lp = clf._predict(zt, _jnp(params))
                L = np.asarray(lp, np.float64)
                pr = L.argmax(1)
                res[f"S{i}_{rung}_New"] = 100.0 * float((pr == tgt)[new].mean())
                res[f"S{i}_{rung}_Old"] = 100.0 * float((pr == tgt)[~new].mean())
                res[f"S{i}_{rung}_g20"] = 100.0 * float((off + L[:, off:n].argmax(1) == tgt)[new].mean())
            g["G1"] &= abs(res[f"S{i}_M0_New"] - float(v["acc1"]["Novel"])) <= 1e-6 \
                and abs(res[f"S{i}_M0_Old"] - float(v["acc1"]["Old"])) <= 1e-6 \
                and abs(res[f"S{i}_M0_New"] - float(e03[f"{pre}S{i}_c14"])) <= 1e-4 \
                and abs(res[f"S{i}_M0_g20"] - float(e03[f"{pre}S{i}_g20"])) <= 1e-4
            res[f"S{i}_d_k"], res[f"S{i}_d_old"] = _dist_1405(p1, off, inc, means["M0"], means["Mk"])
            res.update({f"S{i}_nmin_k": n_min, f"S{i}_empty_k": ek, f"S{i}_empty_c": ec})
        for k in gates:
            gates[k] = gates[k] and g[k]
        out.update({pre + k: round(val, 4) if isinstance(val, float) else val for k, val in res.items()})
        print(f"e1405 {tag}: " + " ".join(f"{k} {'OK' if g[k] else 'DIFF'}" for k in g) + f" ({time.time() - t0:.0f} s)", flush=True)
    chk = {f"{k}_ok": int(v) for k, v in gates.items()}
    chk["checks_ok"] = int(all(gates.values()))
    print(f"e1405 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1405_summary(B=0.376):
    """Tables of E14.05 from the runs cub_e14_05_ft_l25 and cub_e14_05_ft. Every number is first the mean of the 5
    sessions and 5 chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). R(M) = [New(M) -
    New(M0)] / L_comp(M0) per FT seed, then the mean. Reading rule of the pre-registration on the leak-free set.
    Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e14_05_ft_l25", "cub_e14_05_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        per = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)], float)
        ft = lambda k: np.mean([per(f"S{i}_{k}") for i in range(1, 6)], axis=0).mean(1)      # one value per FT seed
        line = lambda name, x: f"  {name:<26}" + "".join(f"{val:9.3f}" for val in x) + f"{x.mean():9.3f}{x.std(ddof=1):9.3f}"
        head = f"  {'':<26}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        print(f"\n{tag}: G1-G4 OK\n  per rung, mean over 5 sessions and 5 chains\n{head}")
        for rung in LADDER_1405:
            for q in ("New", "Old", "g20"):
                print(line(f"{rung} {q}", ft(f"{rung}_{q}")))
            print(line(f"{rung} L_comp", ft(f"{rung}_g20") - ft(f"{rung}_New")))
        lc0 = ft("M0_g20") - ft("M0_New")
        R = {}
        print(f"  R(M) and dOld(M) = Old(M) - Old(M0)\n{head}")
        for rung in LADDER_1405[1:]:
            R[rung] = (ft(f"{rung}_New") - ft("M0_New")) / lc0
            print(line(f"R({rung})", R[rung]))
            print(line(f"dOld({rung})", ft(f"{rung}_Old") - ft("M0_Old")))
        print(f"  distances (median over the 20 slots, then mean)\n{head}")
        print(line("d(M0 mean, Mk mean)", ft("d_k")))
        print(line("d(Mk mean, nearest old)", ft("d_old")))
        print(line("ratio", ft("d_k") / ft("d_old")))
        print(f"  smallest kept cluster: {int(min(per(f'S{i}_nmin_k').min() for i in range(1, 6)))} images; "
              f"empty slots Mk {int(sum(per(f'S{i}_empty_k').sum() for i in range(1, 6)))}, "
              f"Mc {int(sum(per(f'S{i}_empty_c').sum() for i in range(1, 6)))} (of 1500)")
        rk, rc, r1 = R["Mk"].mean(), R["Mc"].mean(), R["M1"].mean()
        verdict = ("the estimator of step 13 is the main source (Mk is label-free: confirm on other FT seeds)" if rk >= 0.5 else
                   "the data fed to step 13 is the main source (old images in clusters, wrong clusters)" if r1 >= 0.5 else
                   "new and old classes overlap in the feature space")
        print(f"  reading rule: R(Mk) {rk:.3f}, R(Mc) {rc:.3f}, R(M1) {r1:.3f} -> {verdict}")
        print(f"  dOld at the deciding rung vs B {B}: Mk {(ft('Mk_Old') - ft('M0_Old')).mean():+.3f}, M1 {(ft('M1_Old') - ft('M0_Old')).mean():+.3f}")
    print("\nE14.05: the reading rule is read on cub_e14_05_ft_l25; cub_e14_05_ft is reported alongside")


def _post_1406(x, prior):
    """NIW posterior of one slot from rows x (lab/niw.py formulas; prior = (m0, k0, nu0, psi0)). No rows: the prior.
    Returns (mn, kn, vn, chol of Psi_n, log det Psi_n)."""
    m0, k0, nu0, psi0 = prior
    n = len(x)
    if n == 0:
        mn, kn, vn, psi = m0, k0, nu0, psi0
    else:
        xb = x.mean(0)
        S = (x - xb).T @ (x - xb)
        kn, vn = k0 + n, nu0 + n
        mn = (k0 * m0 + n * xb) / kn
        psi = psi0 + S + (k0 * n / kn) * np.outer(xb - m0, xb - m0)
    L = np.linalg.cholesky(psi)
    return mn, kn, vn, L, 2.0 * float(np.log(np.diag(L)).sum())


def _logd_1406(slots, x, d):
    """(len(x), len(slots)) log densities, Student-t (A3) and Gaussian plug-in (A2), as niw.log_density."""
    import niw
    t, g = np.empty((len(x), len(slots))), np.empty((len(x), len(slots)))
    for k, (mn, kn, vn, L, ld) in enumerate(slots):
        q = (np.linalg.solve(L, (x - mn).T) ** 2).sum(0)
        t[:, k] = niw._t_logpdf(q, ld, kn, vn, d)
        c = vn - d - 1
        g[:, k] = -0.5 * (ld - d * np.log(c)) - 0.5 * q * c - d / 2 * np.log(2 * np.pi)
    return t, g


def _score_1406(L, y, tgt, off, inc, base):
    """New, Old, All, g20, nobase, nodisc (percent) of one log-density matrix over the seen slots."""
    new, n = y >= off, off + inc
    pr = L.argmax(1)
    ok = pr == tgt
    M = L.copy()
    M[:, :base] = -np.inf
    nob = (M.argmax(1) == tgt)[new]
    M = L.copy()
    M[:, base:off] = -np.inf
    nod = (M.argmax(1) == tgt)[new]
    g20 = (off + L[:, off:n].argmax(1) == tgt)[new]
    pc = lambda a: 100.0 * float(np.mean(a))
    return {"New": pc(ok[new]), "Old": pc(ok[~new]), "All": pc(ok), "g20": pc(g20), "nobase": pc(nob), "nodisc": pc(nod)}


def tier2_e1406(prefix, checks_only=False):
    """E14.06 (pre-registration ac26f0e) on the 15 default chains <prefix>_r<r>_s<N>. Model N: NIW posterior for every
    slot with one prior (m0 = mean of S0, kappa0 = niw.K0, Psi0 = off x pooled S0 covariance, off by leave-one-out on
    S0): base slots from the labeled S0 rows (= A3 of E06.04), slot off_j + c of session j <= i from the kept pool
    images of session j with cluster id c (the rows of step 13); empty cluster = the prior. Prediction: argmax of the
    Student-t log density over the seen slots. Reported with it: A2 (Gaussian plug-in of the same posterior) and Nc
    (oracle: session slots from kept images whose true class is a novel class of that session). The pipeline columns
    (c14, Old, All) come from params1 through _predict; c10 as E14.03. No SVI, no RNG. Gates G1-G4; the progress line
    prints the gates only. Keys: r<r>_s<N>_<key>; G1_ok ... G4_ok, checks_ok. checks_only: chain r0 s0 only, no keys."""
    import niw
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E14.06 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1406: commit the code first (same dirty rule as scripts/exp.py)")
    e03 = _ref_metrics("cub_e14_03_" + prefix[len("cub_chain_"):])
    assert e03.get("checks_ok") == 1, "E14.03 run of this feature set: checks failed"
    out, gates = {}, {f"G{k}": True for k in range(1, 5)}
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag, pre = f"{prefix}_r{r}_s{N}", f"r{r}_s{N}_"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a = st["args"]
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        base, inc, res, g = a.base, a.increment, {}, {f"G{k}": True for k in range(1, 5)}
        z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
        d = z0.shape[1]
        off0 = niw.choose(niw.loo_curve(z0, y0))
        sts = niw.class_stats(z0, y0)
        prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
        base_slots = [_post_1406(z0[y0 == k], prior) for k in range(base)]
        ref = niw.fit(z0, y0, off0)
        rel = lambda u, w: float(np.max(np.abs(u - w)) / max(1e-300, float(np.max(np.abs(w)))))
        g["G3"] &= rel(np.stack([s[0] for s in base_slots]), ref["mn"]) <= 1e-8 \
            and rel(np.array([s[4] for s in base_slots]), ref["logdet"]) <= 1e-8
        zt0, yt0 = np.asarray(st["z_test0"], np.float64), np.asarray(st["true_test0_y"]).astype(int)
        lt0, _ = _logd_1406(base_slots, zt0, d)
        g["G3"] &= rel(lt0[:50], niw.log_density(ref, zt0[:50], "t")) <= 1e-8
        acc_s0 = 100.0 * float((lt0.argmax(1) == yt0).mean())
        res["off"], res["S0_A3"] = off0, acc_s0
        if prefix == "cub_chain_ft":
            e64 = _ref_metrics(f"cub_e06_04_r{r}_s{N}")
            g["G2"] &= int(e64["off"]) == int(off0) and abs(float(e64["acc_A3"]) - acc_s0) <= 1e-6
        disc = {"N": [], "Nc": []}                       # posteriors of the slots of earlier sessions
        for i, v in enumerate(views, 1):
            off = v["offset"]
            n = off + inc
            zp = np.asarray(v["z_pool"], np.float64)
            ids, k = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)
            yp = np.asarray(v["true_pool_y"]).astype(int)
            g["G4"] &= int(sum((k & (ids == c)).sum() for c in range(inc))) == int(k.sum())
            sess = {"N": [], "Nc": []}
            empty = {"N": 0, "Nc": 0}
            for c in range(inc):
                sk = k & (ids == c)
                sc = sk & (yp >= off)
                g["G4"] &= bool(np.all(sk[sc]))
                for name, sel in (("N", sk), ("Nc", sc)):
                    empty[name] += int(not sel.any())
                    sess[name].append(_post_1406(zp[sel], prior))
            clf = _clf(v)
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, n
            xt = clf.pca.transform(v["test_all_x"])
            zt = np.asarray(xt, np.float64)
            y, lut = np.asarray(v["true_test_all_y"]).astype(int), np.asarray(v["lut"])
            tgt, new = lut[y], y >= off
            pr = np.asarray(clf._predict(jnp.array(xt), _jnp(v["params1"]))[0]).astype(int)
            okp = pr == tgt
            res.update({f"S{i}_P_New": 100.0 * float(okp[new].mean()), f"S{i}_P_Old": 100.0 * float(okp[~new].mean()),
                        f"S{i}_P_All": 100.0 * float(okp.mean())})
            g["G1"] &= abs(res[f"S{i}_P_New"] - float(v["acc1"]["Novel"])) <= 1e-6 and abs(res[f"S{i}_P_Old"] - float(v["acc1"]["Old"])) <= 1e-6
            km, y1 = np.asarray(v["cluster"].predict_raw(v["test1_x"])).astype(int), np.asarray(v["true_test1_y"]).astype(int)
            c10 = 100.0 * float((off + np.array([v["pi_test"][int(c)] for c in km]) == y1).mean())
            g["G1"] &= abs(c10 - float(e03[f"{pre}S{i}_c10"])) <= 1e-4
            res[f"S{i}_c10"] = c10
            lb_t, lb_g = _logd_1406(base_slots, zt, d)
            for name in ("N", "Nc"):
                lt, lg = _logd_1406(disc[name] + sess[name], zt, d)
                Lt, Lg = np.concatenate([lb_t, lt], 1), np.concatenate([lb_g, lg], 1)
                assert Lt.shape[1] == n, f"{tag} S{i}: {Lt.shape[1]} slots"
                res.update({f"S{i}_{name}_{q}": val for q, val in _score_1406(Lt, y, tgt, off, inc, base).items()})
                if name == "N":
                    res.update({f"S{i}_A2_{q}": val for q, val in _score_1406(Lg, y, tgt, off, inc, base).items()})
                res[f"S{i}_{name}_empty"] = empty[name]
                disc[name] = disc[name] + sess[name]
        for kk in gates:
            gates[kk] = gates[kk] and g[kk]
        out.update({pre + kk: round(val, 4) if isinstance(val, float) else val for kk, val in res.items()})
        print(f"e1406 {tag}: " + " ".join(f"{kk} {'OK' if g[kk] else 'DIFF'}" for kk in g) + f" ({time.time() - t0:.0f} s)", flush=True)
    chk = {f"{kk}_ok": int(v) for kk, v in gates.items()}
    chk["checks_ok"] = int(all(gates.values()))
    print(f"e1406 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1406_summary(T=0.935, B=0.376):
    """Tables of E14.06 from the runs cub_e14_06_ft_l25 and cub_e14_06_ft. Every number is first the mean of the 5
    sessions and 5 chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). Main and secondary
    reading rules of the pre-registration on the leak-free set. Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e14_06_ft_l25", "cub_e14_06_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        per = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)], float)
        ft = lambda k: np.mean([per(f"S{i}_{k}") for i in range(1, 6)], axis=0).mean(1)
        line = lambda name, x: f"  {name:<26}" + "".join(f"{val:9.3f}" for val in x) + f"{x.mean():9.3f}{x.std(ddof=1):9.3f}"
        head = f"  {'':<26}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        print(f"\n{tag}: G1-G4 OK\n  mean over 5 sessions and 5 chains\n{head}")
        print(line("c10", ft("c10")))
        for q in ("New", "Old", "All"):
            print(line(f"pipeline {q}", ft(f"P_{q}")))
        for v in ("N", "A2", "Nc"):
            for q in ("New", "Old", "All", "g20", "nobase", "nodisc"):
                print(line(f"{v} {q}", ft(f"{v}_{q}")))
            print(line(f"{v} L_comp = g20 - New", ft(f"{v}_g20") - ft(f"{v}_New")))
            print(line(f"{v} L_10 = c10 - New", ft("c10") - ft(f"{v}_New")))
            for q in ("New", "Old", "All"):
                print(line(f"{v} d{q} vs pipeline", ft(f"{v}_{q}") - ft(f"P_{q}")))
        print(line("S0 A3 (base test)", per("S0_A3").mean(1)))
        offs = sorted(set(int(x) for x in per("off").ravel()))
        print(f"  off chosen: {offs}; empty slots N {int(sum(per(f'S{i}_N_empty').sum() for i in range(1, 6)))}, "
              f"Nc {int(sum(per(f'S{i}_Nc_empty').sum() for i in range(1, 6)))} (of 1500)")
        lc = (ft("N_g20") - ft("N_New")).mean()
        l10 = (ft("c10") - ft("N_New")).mean()
        verdict = ("old/new competition stays the main target of E09.08 under a consistent estimator" if lc > T and lc >= 0.5 * l10 else
                   "the competition of E14.03 came from the near-zero means; E09.08 targets the rest of L_10_N" if lc <= T else "both")
        print(f"  main rule: L_comp_N {lc:.3f} vs T {T}, 1/2 L_10_N {0.5 * l10:.3f} -> {verdict}")
        dn, do = (ft("N_New") - ft("P_New")).mean(), (ft("N_Old") - ft("P_Old")).mean()
        better = dn > T and do > -B
        print(f"  secondary rule: dNew {dn:+.3f} (> T {T}?), dOld {do:+.3f} (> -B {-B}?) -> "
              + ("N is better than the pipeline: starting rung of E09.08" if better else "N is not better than the pipeline"))
    print("\nE14.06: the reading rules are read on cub_e14_06_ft_l25; cub_e14_06_ft is reported alongside")


# ---- E12.02: unlabeled old/new filter from the old-class model (pre-registered d566669, amended 571660b) ----
CANDS_1202 = ("A1", "A3_25", "A3_100", "A3_250", "A3_1000")
ALPHAS_1202 = {"NFa02": 0.02, "NFa05": 0.05, "NF": 0.10, "NFa20": 0.20, "NFa30": 0.30, "NFa50": 0.50}
BRANCHES_1202 = ("N", "Nc", "NF", "NFand", "NFa02", "NFa05", "NFa20", "NFa30", "NFa50", "NFcnt")


def _loo_t_1202(z, y, off):
    """Leave-one-out Student-t log density of every labeled row under the NIW posterior of its own class fitted
    without that row, m0 and the pooled covariance also recomputed without it: the body of niw.loo_curve
    (loo_class=False) for one off, returned per row (row order of z) instead of summed."""
    import niw
    z, y = np.asarray(z, np.float64), np.asarray(y).astype(int)
    st = niw.class_stats(z, y)
    d, N, C, k0 = st["d"], st["N"], len(st["n"]), niw.K0
    St = st["scatter"].sum(0)
    out = np.empty(N)
    for k, cls in enumerate(st["classes"]):
        rows = np.flatnonzero(y == cls)
        n, xb, x = st["n"][k], st["mean"][k], z[rows]
        A = off * St / (N - 1 - C) + st["scatter"][k]
        a = (off / (N - 1 - C) + 1) * n / (n - 1)
        u = x - xb
        m0 = st["m0"] - (x - st["m0"]) / (N - 1)
        w = xb - u / (n - 1) - m0
        kn, vn, c = k0 + n - 1, d + 1 + off + n - 1, k0 * (n - 1) / (k0 + n - 1)
        r = x - (k0 * m0 + (n - 1) * (xb - u / (n - 1))) / kn
        L = np.linalg.cholesky(A)
        lu, lw, lr = (np.linalg.solve(L, v.T) for v in (u, w, r))
        m11, m22, m12 = -1 / a + (lu * lu).sum(0), 1 / c + (lw * lw).sum(0), (lu * lw).sum(0)
        b1, b2 = (lu * lr).sum(0), (lw * lr).sum(0)
        det = m11 * m22 - m12 ** 2
        assert (det < 0).all(), "leave-one-out: Psi without the row is not positive definite"
        q = (lr * lr).sum(0) - (m22 * b1 ** 2 - 2 * m12 * b1 * b2 + m11 * b2 ** 2) / det
        logdet = 2 * np.log(np.diag(L)).sum() + np.log(a * c) + np.log(-det)
        out[rows] = niw._t_logpdf(q, logdet, kn, vn, d)
    return out


def _gslot_1202(x):
    """A1 slot from rows x (n >= 3; at n = 2 Ledoit-Wolf gives shrinkage 0 and a singular covariance): sample mean and
    Ledoit-Wolf covariance (closed_form_params, branch n >= 2). Calls sklearn.covariance.ledoit_wolf as
    LedoitWolf.fit does, but skips the precision matrix (a full eigendecomposition per call; G4 checks equality).
    Returns (mean, cholesky, log det)."""
    from sklearn.covariance import ledoit_wolf
    x = np.asarray(x, np.float64)
    m = x.mean(0)
    cov, _ = ledoit_wolf(x - m, assume_centered=True)        # = LedoitWolf().fit, without its precision (pinvh)
    L = np.linalg.cholesky(cov)
    return m, L, 2.0 * float(np.log(np.diag(L)).sum())


def _logd_1202(kind, slots, x):
    """(len(x), len(slots)) log densities: A1 Gaussian with (mean, chol, logdet) slots; A3 Student-t predictive with
    _post_1406 slots (as niw.log_density). Float64, triangular solves."""
    import niw
    from scipy.linalg import solve_triangular
    x = np.asarray(x, np.float64)
    d, out = x.shape[1], np.empty((len(x), len(slots)))
    for k, s in enumerate(slots):
        if kind == "A1":
            m, L, ld = s
            q = (solve_triangular(L, (x - m).T, lower=True) ** 2).sum(0)
            out[:, k] = -0.5 * ld - 0.5 * q - d / 2 * np.log(2 * np.pi)
        else:
            mn, kn, vn, L, ld = s
            q = (solve_triangular(L, (x - mn).T, lower=True) ** 2).sum(0)
            out[:, k] = niw._t_logpdf(q, ld, kn, vn, d)
    return out


def _cand_1202(name, z0, y0, prior_n):
    """One candidate of the score model on S0: (kind, prior, base slots, L0 = log densities of the S0 rows under the
    full-S0 slots, own = leave-one-out log density of each row under its own class). prior_n gives m0 and P."""
    base = int(y0.max()) + 1
    if name == "A1":
        slots = [_gslot_1202(z0[y0 == k]) for k in range(base)]
        own = np.empty(len(y0))
        for k in range(base):
            rows = np.flatnonzero(y0 == k)
            for j in rows:
                s = _gslot_1202(z0[rows[rows != j]])
                own[j] = _logd_1202("A1", [s], z0[j:j + 1])[0, 0]
        return "A1", None, slots, _logd_1202("A1", slots, z0), own
    off = int(name.split("_")[1])
    m0, k0, P = prior_n[0], prior_n[1], prior_n[4]
    prior = (m0, k0, z0.shape[1] + 1 + off, off * P)
    slots = [_post_1406(z0[y0 == k], prior) for k in range(base)]
    return "A3", prior, slots, _logd_1202("A3", slots, z0), _loo_t_1202(z0, y0, off)


def _select_1202(z0, y0, prior_n, off_n):
    """Choice of the score model by a leave-one-class-out simulation on S0 only (pre-registration of E12.02).
    Fake old score of a row = max(own leave-one-out density, densities under the other classes); fake new score of a
    row of class k = max over the classes j != k (full-S0 slots). AUROC with fake old as the positive class; the
    largest wins, ties go to the A3 candidate whose off is nearest off_n (A1 loses ties).
    Returns (name, {name: auroc}, {name: candidate}, {name: sorted calibration scores})."""
    from sklearn.metrics import roc_auc_score
    res, au, cal = {}, {}, {}
    idx = np.arange(len(y0))
    for name in CANDS_1202:
        c = _cand_1202(name, z0, y0, prior_n)
        L0 = c[3].copy()
        L0[idx, y0] = -np.inf
        new = L0.max(1)
        old = np.maximum(c[4], new)
        au[name] = float(roc_auc_score(np.r_[np.ones(len(old)), np.zeros(len(new))], np.r_[old, new]))
        res[name], cal[name] = c, np.sort(old)
    rank = lambda nm: (au[nm], nm != "A1", -abs(int(nm.split("_")[1]) - off_n) if nm != "A1" else 0)
    return max(CANDS_1202, key=rank), au, res, cal


def _pval_1202(s, cal):
    """Conformal p-value (Bates et al. 2023) of scores s against sorted calibration scores: (1 + #{s_j <= s}) / (n + 1)."""
    return (1.0 + np.searchsorted(cal, s, side="right")) / (len(cal) + 1.0)


def _branch_1202(ctx, rule):
    """One branch of E12.02 over the 5 sessions. rule: ("fixed", [keep per session]) for N, Nc; ("override", [keep])
    = the filter path with the keep replaced (G2); ("alpha", a); ("and", a) = filter and pipeline keep; ("count",)
    = the number of novel pool images, lowest scores (oracle). The keep rules see the scores only (count: one number).
    Slots of earlier sessions (N posterior and score-model slot) come from this branch's own keep.
    Returns ([per-session dict], [keep per session], [score per session or None])."""
    kind, base, inc = ctx["kind"], ctx["base"], ctx["inc"]
    discN, discF, out, keeps, scores = [], [], [], [], []
    for i, sv in enumerate(ctx["sess"], 1):
        off = base + (i - 1) * inc
        zp, ids, yp = sv["zp"], sv["ids"], sv["yp"]
        s = None
        if rule[0] != "fixed":
            s = ctx["fpool"][i - 1].max(1)
            if discF:
                s = np.maximum(s, _logd_1202(kind, discF, zp).max(1))
        if rule[0] in ("fixed", "override"):
            keep = np.asarray(rule[1][i - 1]).astype(bool)
        elif rule[0] == "alpha":
            keep = _pval_1202(s, ctx["cal"]) <= rule[1]
        elif rule[0] == "and":
            keep = (_pval_1202(s, ctx["cal"]) <= rule[1]) & sv["keep_pipe"]
        elif rule[0] == "count":
            keep = np.zeros(len(zp), bool)
            keep[np.argsort(s, kind="stable")[:int((yp >= off).sum())]] = True
        else:
            raise ValueError(rule)
        sessN, sessF, empty = [], [], 0
        for c in range(inc):
            sel = keep & (ids == c)
            empty += int(not sel.any())
            sessN.append(_post_1406(zp[sel], ctx["priorN"]))
            if sel.sum() >= (3 if kind == "A1" else 2):              # Ledoit-Wolf at n = 2: shrinkage 0, singular
                sessF.append(_gslot_1202(zp[sel]) if kind == "A1" else _post_1406(zp[sel], ctx["priorF"]))
        lt = _logd_1202("A3", discN + sessN, sv["zt"])          # N slots: Student-t as _logd_1406, triangular solves
        r = _score_1406(np.concatenate([sv["lbN"], lt], 1), sv["y"], sv["tgt"], off, inc, base)
        old_b, old_d, nov = yp < base, (yp >= base) & (yp < off), yp >= off
        r.update(empty=empty, oldbase_kept=float(keep[old_b].mean()), new_dropped=float(1.0 - keep[nov].mean()))
        if old_d.any():
            r["olddisc_kept"] = float(keep[old_d].mean())
        if s is not None:
            from sklearn.metrics import roc_auc_score
            r["auroc_pool"] = float(roc_auc_score((yp < off).astype(int), s))
        out.append(r)
        keeps.append(keep)
        scores.append(s)
        discN, discF = discN + sessN, discF + sessF
    return out, keeps, scores


def tier2_e1202(prefix, checks_only=False):
    """E12.02 (pre-registration d566669, amendment 571660b) on the 15 default chains <prefix>_r<r>_s<N>.
    Score model chosen per chain on S0 (_select_1202); filter = conformal p-value <= alpha on the pool of each
    session; model N of E14.06 fitted on the kept images (branches BRANCHES_1202, _branch_1202). The rules that keep
    images read the scores only; pool labels serve the oracle branches (Nc, NFcnt) and the scoring of the filter; test
    labels serve scoring. No SVI, no RNG. Gates:
    G1 pipeline columns = acc1 (1e-6); N, Nc, c10 = the E14.06 run of this set (New, Old, All, g20, c10; 1e-4).
    G2 the filter path with the pipeline keep gives N (1e-6).
    G3 (amended) session 1: keeps nested in alpha, NFand within NF; every session: the alphas on the NF score vector
    give nested keeps.
    G4 sum of _loo_t_1202 = niw.loo_curve for every A3 candidate (1e-8); class 0 of S0: _loo_t_1202 = refit by
    niw.fit without the row (off of N, 1e-8); base slots of N = niw.fit (1e-8); A1 base labels on the S0 test set =
    mngmm._fit_closed + _predict (base slots).
    G5 calibration p-values (own row left out) <= alpha at a rate within 1/(n + 1) + ties of alpha.
    The progress line prints the gates and the time only. Keys: r<r>_s<N>_<key>; G1_ok ... G5_ok, checks_ok."""
    import niw
    from replay import _clf, _git
    from sessions import load as load_sessions, _device
    assert prefix in ("cub_chain_ft", "cub_chain_ft_l25"), "E12.02 is pre-registered on the two sets of CUB chains"
    t0 = time.time()
    if os.environ.get("EXP_RUN_DIR") and _git()["dirty"]:
        raise SystemExit("e1202: commit the code first (same dirty rule as scripts/exp.py)")
    e06 = _ref_metrics("cub_e14_06_" + prefix[len("cub_chain_"):])
    assert e06.get("checks_ok") == 1, "E14.06 run of this feature set: checks failed"
    out, gates = {}, {f"G{k}": True for k in range(1, 6)}
    for r, N in ((0, 0),) if checks_only else [(r, N) for r in range(3) for N in range(5)]:
        tag, pre = f"{prefix}_r{r}_s{N}", f"r{r}_s{N}_"
        st, views = load_sessions(tag)                  # the workbench first: torch, main, then jax, as the chains
        import jax.numpy as jnp
        _device()
        a = st["args"]
        assert a.dataset == "cub200" and a.load_mode == "t5" and len(views) == 5 and a.seed == N, f"{tag}: not a CUB t5 chain of seed {N}"
        base, inc, res, g = a.base, a.increment, {}, {f"G{k}": True for k in range(1, 6)}
        z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
        assert np.array_equal(np.unique(y0), np.arange(base)), f"{tag}: S0 classes"
        d = z0.shape[1]
        # model N (E14.06)
        off_n = niw.choose(niw.loo_curve(z0, y0))
        sts = niw.class_stats(z0, y0)
        P = niw.prior_cov(sts, 0)
        prior_n = (sts["m0"], niw.K0, d + 1 + off_n, off_n * P, P)
        priorN = prior_n[:4]
        baseN = [_post_1406(z0[y0 == k], priorN) for k in range(base)]
        ref = niw.fit(z0, y0, off_n)
        rel = lambda u, w: float(np.max(np.abs(np.asarray(u) - np.asarray(w))) / max(1e-300, float(np.max(np.abs(w)))))
        g["G4"] &= rel(np.stack([s[0] for s in baseN]), ref["mn"]) <= 1e-8 and rel(np.array([s[4] for s in baseN]), ref["logdet"]) <= 1e-8
        # score model, chosen on S0
        sel, au, cands, cals = _select_1202(z0, y0, prior_n, off_n)
        for name in CANDS_1202:
            if name != "A1":
                o = int(name.split("_")[1])
                g["G4"] &= rel(cands[name][4].sum(), niw.loo_curve(z0, y0, (o,))[0]) <= 1e-8
        rows0 = np.flatnonzero(y0 == 0)
        own_n = _loo_t_1202(z0, y0, off_n)
        for j in rows0:
            keepj = np.arange(len(y0)) != j
            m = niw.fit(z0[keepj], y0[keepj], off_n)
            g["G4"] &= rel(own_n[j], niw.log_density(m, z0[j:j + 1], "t")[0, int(np.flatnonzero(m["classes"] == 0)[0])]) <= 1e-8
        clf = _clf(views[0])
        clf.global_params, clf.label_offset = None, 0
        with contextlib.redirect_stdout(io.StringIO()):
            p1 = clf._fit_closed(st["z_s0"], st["s0_y"], None, "e1202")
        zt0 = np.asarray(st["z_test0"], np.float64)
        lab_pipe = np.asarray(clf._predict(jnp.asarray(st["z_test0"]), p1)[1])[:, :base].argmax(1)
        g["G4"] &= bool(np.array_equal(_logd_1202("A1", cands["A1"][2], zt0).argmax(1), lab_pipe))
        from sklearn.covariance import LedoitWolf
        lw0 = LedoitWolf().fit(z0[y0 == 0])
        g["G4"] &= bool(np.array_equal(cands["A1"][2][0][0], lw0.location_)
                        and np.array_equal(cands["A1"][2][0][1], np.linalg.cholesky(lw0.covariance_)))
        kind, priorF, slotsF = cands[sel][0], cands[sel][1], cands[sel][2]
        cal = cals[sel]
        n_cal = len(cal)
        p_self = (1.0 + np.searchsorted(cal, cal, side="right") - 1.0) / (n_cal + 1.0)
        ties = 1.0 - len(np.unique(cal)) / n_cal
        g["G5"] &= all(abs(float((p_self <= al).mean()) - al) <= 1.0 / (n_cal + 1) + ties for al in ALPHAS_1202.values())
        res.update({"sel": CANDS_1202.index(sel), "off": int(off_n)})
        res.update({f"auroc_{nm}": au[nm] for nm in CANDS_1202})
        # per-session inputs and the pipeline columns
        sess = []
        for i, v in enumerate(views, 1):
            off = v["offset"]
            clf = _clf(v)
            _component(v["predict"])(clf)
            clf.label_offset, clf.n_seen = off, off + inc
            xt = clf.pca.transform(v["test_all_x"])
            zt = np.asarray(xt, np.float64)
            y, lut = np.asarray(v["true_test_all_y"]).astype(int), np.asarray(v["lut"])
            tgt, new = lut[y], y >= off
            pr = np.asarray(clf._predict(jnp.array(xt), _jnp(v["params1"]))[0]).astype(int)
            okp = pr == tgt
            res.update({f"S{i}_P_New": 100.0 * float(okp[new].mean()), f"S{i}_P_Old": 100.0 * float(okp[~new].mean()),
                        f"S{i}_P_All": 100.0 * float(okp.mean())})
            g["G1"] &= abs(res[f"S{i}_P_New"] - float(v["acc1"]["Novel"])) <= 1e-6 and abs(res[f"S{i}_P_Old"] - float(v["acc1"]["Old"])) <= 1e-6
            km, y1 = np.asarray(v["cluster"].predict_raw(v["test1_x"])).astype(int), np.asarray(v["true_test1_y"]).astype(int)
            res[f"S{i}_c10"] = 100.0 * float((off + np.array([v["pi_test"][int(c)] for c in km]) == y1).mean())
            g["G1"] &= abs(res[f"S{i}_c10"] - float(e06[f"{pre}S{i}_c10"])) <= 1e-4
            zp = np.asarray(v["z_pool"], np.float64)
            yp = np.asarray(v["true_pool_y"]).astype(int)
            sess.append(dict(zp=zp, ids=np.asarray(v["ids"]).astype(int), yp=yp, keep_pipe=np.asarray(v["keep"]).astype(bool),
                             zt=zt, y=y, tgt=tgt, lbN=_logd_1202("A3", baseN, zt)))
        ctx = dict(kind=kind, base=base, inc=inc, d=d, priorN=priorN, priorF=priorF, cal=cal, sess=sess,
                   fpool=[_logd_1202(kind, slotsF, sv["zp"]) for sv in sess])
        pipe = [sv["keep_pipe"] for sv in sess]
        oracle = [sv["keep_pipe"] & (sv["yp"] >= base + i * inc) for i, sv in enumerate(sess)]
        rules = {"N": ("fixed", pipe), "Nc": ("fixed", oracle), "NF": ("alpha", ALPHAS_1202["NF"]),
                 "NFand": ("and", ALPHAS_1202["NF"]), "NFcnt": ("count",),
                 **{b: ("alpha", al) for b, al in ALPHAS_1202.items() if b != "NF"}}
        br = {b: _branch_1202(ctx, rules[b]) for b in BRANCHES_1202}
        g2, _, _ = _branch_1202(ctx, ("override", pipe))
        qs = ("New", "Old", "All", "g20")
        for i in range(1, 6):
            for b in ("N", "Nc"):
                g["G1"] &= all(abs(br[b][0][i - 1][q] - float(e06[f"{pre}S{i}_{b}_{q}"])) <= 1e-4 for q in qs)
            g["G2"] &= all(abs(g2[i - 1][q] - br["N"][0][i - 1][q]) <= 1e-6 for q in qs + ("nobase", "nodisc"))
        order = ("NFa02", "NFa05", "NF", "NFa20", "NFa30", "NFa50")
        k1 = {b: br[b][1][0] for b in order + ("NFand",)}
        g["G3"] &= all(not np.any(k1[x] & ~k1[y]) for x, y in zip(order, order[1:])) and not np.any(k1["NFand"] & ~k1["NF"])
        for s in br["NF"][2]:
            ks = [_pval_1202(s, cal) <= ALPHAS_1202[b] for b in order]
            g["G3"] &= all(not np.any(x & ~y) for x, y in zip(ks, ks[1:]))
        for b in BRANCHES_1202:
            for i, rr in enumerate(br[b][0], 1):
                res.update({f"S{i}_{b}_{q}": val for q, val in rr.items()})
        for kk in gates:
            gates[kk] = gates[kk] and g[kk]
        out.update({pre + kk: round(val, 4) if isinstance(val, float) else val for kk, val in res.items()})
        print(f"e1202 {tag}: " + " ".join(f"{kk} {'OK' if g[kk] else 'DIFF'}" for kk in g) + f" ({time.time() - t0:.0f} s)", flush=True)
    chk = {f"{kk}_ok": int(v) for kk, v in gates.items()}
    chk["checks_ok"] = int(all(gates.values()))
    print(f"e1202 {prefix}: checks {'OK' if chk['checks_ok'] else 'FAILED'} ({time.time() - t0:.0f} s)")
    return chk if checks_only else {**chk, **out}


def e1202_summary(T=0.935, B=0.376):
    """Tables of E12.02 from the runs cub_e12_02_ft_l25 and cub_e12_02_ft. Every number is first the mean of the 5
    sessions and 5 chains of an FT seed; "mean" and "sd" are over the 3 FT seeds (sd with ddof 1). rho(branch) =
    (Old - Old_N) / (Old_Nc - Old_N) per FT seed, then the mean. Reading rule of the pre-registration on NF, leak-free
    set. Refuses to read a run without checks_ok = 1."""
    for tag in ("cub_e12_02_ft_l25", "cub_e12_02_ft"):
        if not sorted((ROOT / "runs").glob(f"*__{tag}/metrics.json")):
            print(f"\n{tag}: no run yet")
            continue
        m = _ref_metrics(tag)
        assert m.get("checks_ok") == 1, f"{tag}: checks failed, results are not read"
        per = lambda key: np.array([[m[f"r{r}_s{N}_{key}"] for N in range(5)] for r in range(3)], float)
        ft = lambda k, ss=range(1, 6): np.mean([per(f"S{i}_{k}") for i in ss], axis=0).mean(1)
        line = lambda name, x: f"  {name:<28}" + "".join(f"{val:9.3f}" for val in x) + f"{x.mean():9.3f}{x.std(ddof=1):9.3f}"
        head = f"  {'':<28}" + "".join(f"{x:>9}" for x in ("r0", "r1", "r2", "mean", "sd"))
        print(f"\n{tag}: G1-G5 OK\n  score model chosen on S0 (chains): "
              + ", ".join(f"{nm} {int((per('sel') == j).sum())}" for j, nm in enumerate(CANDS_1202)))
        print(head)
        for nm in CANDS_1202:
            print(line(f"S0 leave-one-class-out AUROC {nm}", per(f"auroc_{nm}").mean(1)))
        print(f"  mean over 5 sessions and 5 chains\n{head}")
        for q in ("New", "Old", "All"):
            print(line(f"pipeline {q}", ft(f"P_{q}")))
        oN, oC = ft("N_Old"), ft("Nc_Old")
        for b in BRANCHES_1202:
            for q in ("New", "Old", "All", "g20"):
                print(line(f"{b} {q}", ft(f"{b}_{q}")))
            print(line(f"{b} L_comp = g20 - New", ft(f"{b}_g20") - ft(f"{b}_New")))
            for q in ("New", "Old"):
                print(line(f"{b} d{q} vs pipeline", ft(f"{b}_{q}") - ft(f"P_{q}")))
                print(line(f"{b} d{q} vs N", ft(f"{b}_{q}") - ft(f"N_{q}")))
            print(line(f"{b} rho", (ft(f"{b}_Old") - oN) / (oC - oN)))
            print(line(f"{b} old base kept", ft(f"{b}_oldbase_kept")))
            print(line(f"{b} old discovered kept (S2-S5)", ft(f"{b}_olddisc_kept", range(2, 6))))
            print(line(f"{b} novel dropped", ft(f"{b}_new_dropped")))
            if b not in ("N", "Nc"):
                print(line(f"{b} pool AUROC of s", ft(f"{b}_auroc_pool")))
            print(f"  {b} empty N slots: {int(sum(per(f'S{i}_{b}_empty').sum() for i in range(1, 6)))} (of 1500)")
        print(f"  NF per session (old base kept / old discovered kept / novel dropped / pool AUROC):")
        for i in range(1, 6):
            od = f"{per(f'S{i}_NF_olddisc_kept').mean():.3f}" if i > 1 else "-"
            print(f"  S{i}: {per(f'S{i}_NF_oldbase_kept').mean():.3f} / {od} / {per(f'S{i}_NF_new_dropped').mean():.3f} / "
                  f"{per(f'S{i}_NF_auroc_pool').mean():.3f}")
        rho = ((ft("NF_Old") - oN) / (oC - oN)).mean()
        dnew = (ft("NF_New") - ft("P_New")).mean()
        verdict = ("NF is better than the pipeline (secondary rule of E14.06): starting rung of E09.08; confirm on r3, r4 or "
                   "the server" if rho >= 0.67 and dnew > T else
                   "the hard filter recovers only part: soft responsibilities stay the main mechanism of E09.08" if rho >= 1 / 3 else
                   "the old-class model alone does not separate enough: see NFcnt and the pool AUROC")
        print(f"  reading rule: rho(NF) {rho:.3f}, dNew(NF) vs pipeline {dnew:+.3f} (T {T}) -> {verdict}")
        print(f"  L_comp(NF) {(ft('NF_g20') - ft('NF_New')).mean():.3f}: the filter does not target the competition at prediction")
    print("\nE12.02: the reading rule is read on cub_e12_02_ft_l25; cub_e12_02_ft is reported alongside")


def main(argv):
    p = argparse.ArgumentParser(prog="score.py", description="tier-1 scores of one step of a replay v2 run")
    p.add_argument("step", choices=["03", "09", "e0907", "e0907-summary", "e0604", "e0604-summary", "e0906", "e0906-summary", "e0909", "e0909-summary", "e1101", "e1101-summary", "cub-tb", "e1401", "e1401-summary", "e1402", "e1402-summary", "e1402b", "e1402b-summary", "e1403", "e1403-summary", "e1405", "e1405-summary", "e1406", "e1406-summary", "e1202", "e1202-summary"])
    p.add_argument("--from", dest="run", help="09: run id or tag of a v2 run (runs/<run>/snap)")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30], help="numbers of clusters K'")
    p.add_argument("--k-first", type=int, default=0, help="e0907, e0906: first draw")
    p.add_argument("--k-last", type=int, default=10, help="e0907, e0906: last draw")
    p.add_argument("--m", type=int, default=1000, help="e0907: single inits per draw (10 = check C1 only)")
    p.add_argument("--dataset", default="gc10", choices=["gc10", "cub200"], help="e0604")
    p.add_argument("--features", help="e0604 cub200: features/<run_id>/<model>")
    p.add_argument("--ft", type=int, help="e0604 cub200: FT seed of the features")
    p.add_argument("--seed", type=int, help="e0604 cub200: seed of the chain")
    p.add_argument("--checks-only", action="store_true", help="e0604: print the checks C1-C3 only, no result; e1401: chain r0 s0 only")
    p.add_argument("--prefix", choices=["cub_chain_ft", "cub_chain_ft_l25"], help="e1401, e1402: tag prefix of the 15 chains")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: features/, splits/)"
    assert o.step in ("03", "e0907-summary", "e0604-summary", "e0906-summary", "e0909-summary", "e1101-summary", "cub-tb", "e1401", "e1401-summary", "e1402", "e1402-summary", "e1402b", "e1402b-summary", "e1403", "e1403-summary", "e1405", "e1405-summary", "e1406", "e1406-summary", "e1202", "e1202-summary") or o.run or (o.step == "e0604" and o.dataset == "cub200"), \
        "steps 09, e0907, e0906 and e0604 (gc10) need --from <run>"
    if o.step == "e0907-summary":
        e0907_summary()
        return
    if o.step == "e0604-summary":
        e0604_summary()
        return
    if o.step == "e0906-summary":
        e0906_summary()
        return
    if o.step == "e0909-summary":
        e0909_summary()
        return
    if o.step == "e1101-summary":
        e1101_summary()
        return
    if o.step == "cub-tb":
        cub_tb()
        return
    if o.step == "e1401-summary":
        e1401_summary()
        return
    if o.step == "e1402-summary":
        e1402_summary()
        return
    if o.step == "e1402b-summary":
        e1402b_summary()
        return
    if o.step == "e1403-summary":
        e1403_summary()
        return
    if o.step == "e1405-summary":
        e1405_summary()
        return
    if o.step == "e1406-summary":
        e1406_summary()
        return
    if o.step == "e1202-summary":
        e1202_summary()
        return
    if o.step in ("e1401", "e1402", "e1402b", "e1403", "e1405", "e1406", "e1202"):
        assert o.prefix, f"{o.step} needs --prefix"
        out = {"e1401": tier2_e1401, "e1402": tier2_e1402, "e1402b": tier2_e1402b, "e1403": tier2_e1403, "e1405": tier2_e1405, "e1406": tier2_e1406, "e1202": tier2_e1202}[o.step](o.prefix, o.checks_only)
        if os.environ.get("EXP_RUN_DIR") and not o.checks_only:
            (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
        return
    if o.step == "e0604":
        out = tier2_e0604(o.dataset, o.run, o.features, o.ft, o.seed, o.checks_only)
        if os.environ.get("EXP_RUN_DIR") and not o.checks_only:
            (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
        return
    out = tier1_03() if o.step == "03" else tier1_09(o.run, o.ks) if o.step == "09" else \
        tier2_e0906(o.run, o.k_first, o.k_last) if o.step == "e0906" else tier1_e0909(o.run) if o.step == "e0909" else \
        tier2_e1101(o.run, o.k_first, o.k_last) if o.step == "e1101" else tier2_e0907(o.run, o.k_first, o.k_last, o.m)
    for key, v in out.items():
        if o.step not in ("e0907", "e0906", "e0909", "e1101") or not re.match(r"[dp]\d+_", key):   # per-draw keys: metrics.json only
            print(f"{key:30s} {v}")
    if os.environ.get("EXP_RUN_DIR"):
        (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
