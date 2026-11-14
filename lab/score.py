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


def main(argv):
    p = argparse.ArgumentParser(prog="score.py", description="tier-1 scores of one step of a replay v2 run")
    p.add_argument("step", choices=["03", "09", "e0907", "e0907-summary", "e0604", "e0604-summary", "e0906", "e0906-summary", "e0909", "e0909-summary", "e1101", "e1101-summary"])
    p.add_argument("--from", dest="run", help="09: run id or tag of a v2 run (runs/<run>/snap)")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30], help="numbers of clusters K'")
    p.add_argument("--k-first", type=int, default=0, help="e0907, e0906: first draw")
    p.add_argument("--k-last", type=int, default=10, help="e0907, e0906: last draw")
    p.add_argument("--m", type=int, default=1000, help="e0907: single inits per draw (10 = check C1 only)")
    p.add_argument("--dataset", default="gc10", choices=["gc10", "cub200"], help="e0604")
    p.add_argument("--features", help="e0604 cub200: features/<run_id>/<model>")
    p.add_argument("--ft", type=int, help="e0604 cub200: FT seed of the features")
    p.add_argument("--seed", type=int, help="e0604 cub200: seed of the chain")
    p.add_argument("--checks-only", action="store_true", help="e0604: print the checks C1-C3 only, no result")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: features/, splits/)"
    assert o.step in ("03", "e0907-summary", "e0604-summary", "e0906-summary", "e0909-summary", "e1101-summary") or o.run or (o.step == "e0604" and o.dataset == "cub200"), \
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
