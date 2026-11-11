"""Tier-1 evaluation (QD18; lab/README.vi.md sections 5 and 9): the output of one step scored with the true labels,
for scoring only, to screen methods fast; a ranking at tier 1 must be confirmed at tier 2 (full pipeline, exp.py runs).
Apart from the check commands of replay.py, this is the only place in lab/ that reads the true labels of the pool.

  python lab/score.py 09 --from <v2 run> [--ks 5 10 15 20 30]     through scripts/exp.py: writes metrics.json
  python lab/score.py 03                                             E03.02: linear probes on the GC10 features
  python lab/score.py e0907 --from <v2 run> [--k-first 0 --k-last 10 --m 1000]   E09.07, one snapshot (exp.py)
  python lab/score.py e0907-summary                                  E09.07: decision table from the 15 runs
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
from replay import load, snapshots, _fresh, _component, _jnp, chain, _check_09, _run_dir   # noqa: E402
from variants import v09a_overcluster, multi_init, cluster_of              # noqa: E402
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


def main(argv):
    p = argparse.ArgumentParser(prog="score.py", description="tier-1 scores of one step of a replay v2 run")
    p.add_argument("step", choices=["03", "09", "e0907", "e0907-summary"])
    p.add_argument("--from", dest="run", help="09: run id or tag of a v2 run (runs/<run>/snap)")
    p.add_argument("--ks", type=int, nargs="+", default=[5, 10, 15, 20, 30], help="numbers of clusters K'")
    p.add_argument("--k-first", type=int, default=0, help="e0907: first draw")
    p.add_argument("--k-last", type=int, default=10, help="e0907: last draw")
    p.add_argument("--m", type=int, default=1000, help="e0907: single inits per draw (10 = check C1 only)")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: features/, splits/)"
    assert o.step in ("03", "e0907-summary") or o.run, "steps 09 and e0907 need --from <run>"
    if o.step == "e0907-summary":
        e0907_summary()
        return
    out = tier1_03() if o.step == "03" else tier1_09(o.run, o.ks) if o.step == "09" else \
        tier2_e0907(o.run, o.k_first, o.k_last, o.m)
    for key, v in out.items():
        if o.step != "e0907" or not re.match(r"[dp]\d+_", key):       # e0907: the per-draw keys go to metrics.json only
            print(f"{key:30s} {v}")
    if os.environ.get("EXP_RUN_DIR"):
        (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
