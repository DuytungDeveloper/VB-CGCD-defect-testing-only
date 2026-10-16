#!/usr/bin/env python
"""Replay main.py stage by stage for one seed (t1 protocol, --hungarian_on eval, labeled S0) and keep the
intermediate objects, for the notebooks in lab/ and offline experiments of the PL line.

Same calls in the same order as main.py, so the global numpy RNG (loader permutations, the S0 PCA with the
randomized solver, k-means) is consumed identically. main.py itself is not changed.
  r = replay(seed)                   # full: S0 + S1 classifier, about 2 min on the laptop
  r = replay(seed, stop="cluster")   # stop right after the S1 clustering (the S0 SVI still runs)
  r["stages"][i]: train, test, test_old, test_all (DataPoint ._x ._y); for i >= 1 also cluster, raw (cluster ids),
                  inertia, pi_test (cluster -> novel offset), lut, offset, acc (full mode only)
  r["svi"]: (log_prefix, params) of every SVI pass (S0, then F and S of each session); r["clf"]: the classifier
Check against runs of the pipeline (default: the E1 runs gc10_frozen_km_evhung_d64_head_s*):
  python lab/replay.py 0 1 2 3 4      # stop="cluster": contingency table equal, inertia rel < 1e-6
  python lab/replay.py --full 0       # full: S1 accuracies must equal the run's metrics.json
"""
import functools, json, os, re, sys, tempfile, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]      # repo root: main.py, clustering/, dataloaders/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GC10 = dict(dataset="gc10", base=5, increment=5, num_classes=10, num_dim=64)
NEU = dict(dataset="neudet", base=3, increment=3, num_classes=6, num_dim=128)
DEFAULTS = dict(load_mode="t1", clustering_alg="kmeans", hungarian_on="eval", s0_labels="true",
                classifier_alg="mngmm", kmeans_n_init=10, gmm_n_init=1, with_early_stop=True,
                use_correct_scaling_factor=True, n_epochs=1000, lr=4e-6, batch_size=128, scaling_factor=1.2,
                early_stop_ratio=0, estimator="svi", cov_prior_kappa=0.0, cov_prior_target="pooled",
                mask_unseen=False, oracle_filter=False, print_confusion=False, print_cluster_xtab=False,
                trail_name="replay")


def make_args(features, seed, preset=GC10, **over):
    f = Path(features)                              # features/<run_id>/<model>, as in runs/inputs.txt
    return types.SimpleNamespace(**{**DEFAULTS, **preset, "data_dir": str(f.parent),
                                    "pretrained_model_name": f.name, "seed": seed, **over})


def replay(seed=0, features=None, preset=GC10, stop=None, **over):
    assert Path("main.py").exists(), "run from the repo root (relative paths: runs/, features/, splits/)"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # memory only, results unchanged
    import torch, jax                             # same import order as main.py
    import main                                     # the pipeline's own functions and classes
    from clustering.clustering_base import hungarian_map
    args = make_args(features or Path("runs/inputs.txt").read_text().strip(), seed, preset, **over)
    assert (args.load_mode, args.hungarian_on, args.s0_labels) == ("t1", "eval", "true"), "only the protocol in use"
    torch.manual_seed(args.seed); np.random.seed(args.seed); jax.random.PRNGKey(args.seed)   # main.py order
    Clustering = functools.partial(main.Clustering_alg(args.clustering_alg),
                                   n_init=args.kmeans_n_init if args.clustering_alg == "kmeans" else args.gmm_n_init)
    loaders = main.Dataloader(args)
    tmp = tempfile.mkdtemp(prefix="replay_"); Path(f"{tmp}/saved_models").mkdir()
    clf = main.Classifier_alg(args.classifier_alg)(num_classes=args.num_classes, num_dim=args.num_dim,
                                                   with_early_stop=args.with_early_stop)
    for k in ("cov_prior_kappa", "cov_prior_target", "estimator", "mask_unseen", "oracle_filter"):
        setattr(clf, k, getattr(args, k))
    clf.init_parameters(n_epochs=args.n_epochs, lr=args.lr, log_dir=f"{tmp}/log/stage0",
                        save_dir=f"{tmp}/saved_models/stage0", batch_size=args.batch_size, increment=args.increment,
                        base=args.base, scaling_factor=args.scaling_factor,
                        use_correct_scaling_factor=args.use_correct_scaling_factor, early_stop_ratio=args.early_stop_ratio)
    svi, inner = [], clf.run_inference              # record every SVI pass; the numbers are not touched
    def rec(*a, **k):
        p = inner(*a, **k); svi.append((k.get("log_prefix"), {n: np.asarray(v) for n, v in p.items()})); return p
    clf.run_inference = rec
    out, to_cluster, known = {"args": args, "stages": [], "svi": svi, "clf": clf}, {}, None
    for i, (tr, te, te_old, te_all) in enumerate(zip(*loaders)):
        st = dict(train=tr, test=te, test_old=te_old, test_all=te_all)
        out["stages"].append(st)
        if i == 0:
            clf.run(tr._x, tr._y, te._x, te._y, current_stage=0,
                    testing_set={"test_old": te, "test_all": te, "known_test": te})
            known = te
            continue
        off = args.base + (i - 1) * args.increment
        cl = Clustering(num_classes=args.increment, label_offset=off)
        cl.fit(tr._x)
        raw = cl.predict_raw(tr._x)
        pi = hungarian_map(cl.predict_raw(te._x), np.asarray(te._y).astype(int) - off, args.increment)
        to_cluster.update({off + t: off + c for c, t in pi.items()})
        lut = np.arange(args.num_classes)
        for t, c in to_cluster.items():
            lut[t] = c
        st.update(cluster=cl, raw=raw, pi_test=pi, inertia=getattr(cl.model, "inertia_", None), lut=lut, offset=off)
        if stop == "cluster":
            break
        rel = lambda d: types.SimpleNamespace(_x=d._x, _y=lut[np.asarray(d._y).astype(int)])
        clf._set_label_offset(off)
        clf.update_dir_infos(log_dir=f"{tmp}/log/stage{i}", save_dir=f"{tmp}/saved_models/stage{i}")
        clf.oracle_novel = np.asarray(tr._y).astype(int) >= off
        tn = rel(te)
        clf.run(tr._x, raw + off, tn._x, tn._y, current_stage=i,
                testing_set={"test_old": rel(te_old), "test_all": rel(te_all), "known_test": known})
        st["acc"] = {k: clf.test(d._x, d._y)[2] for k, d in
                     (("All", rel(te_all)), ("Old", rel(te_old)), ("New", tn), ("S0test", known))}
    return out


def _reference(seed, tag="gc10_frozen_km_evhung_d64_head"):
    import ast
    rd = sorted(Path("runs").glob(f"*__{tag}_s{seed}"))
    assert len(rd) == 1, (tag, seed, rd)
    log = (rd[0] / "console.log").read_text(errors="replace")
    m = re.search(r"xtab stage 1 objective: inertia=(\S+)", log)
    x = re.search(r"xtab stage 1 train: rows=\[[^\]]*\] cols=\[[^\]]*\] (\[\[.*\]\])", log)
    return (float(m.group(1)) if m else None), (ast.literal_eval(x.group(1)) if x else None), \
        json.loads((rd[0] / "metrics.json").read_text())


# ====================================================================================================== v2
# Replay harness v2 (PL-QD6, notes/pl/README.vi.md). The default chain of main.py (t1, --hungarian_on eval,
# labeled S0) as one function per step 04-14 (lab/README.vi.md section 3). A step sNN(st) reads the state dict
# st and returns only the keys it adds (OWNS). True labels of the pool and of the test sets live in true_* keys.

STEPS = ("04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14")
DEFAULT = {"04": "s04_loader", "05": "s05_pca", "06": "s06_gauss_svi", "07": "s07_early_stop",
           "08": "s08_classifier", "09": "s09_cluster", "10": "s10_naming", "11": "s11_gauss_f",
           "12": "s12_filter", "13": "s13_gauss_s", "14": "s14_eval"}
OWNS = {"04": ("s0_x", "s0_y", "pool_x", "test0_x", "test1_x", "test_old_x", "test_all_x", "offset",
               "true_pool_y", "true_test0_y", "true_test1_y", "true_test_old_y", "true_test_all_y"),
        "05": ("pca", "z_s0", "z_pool", "z_test0", "z_test1"), "06": ("params0", "fit"), "07": ("stop",),
        "08": ("predict", "acc0"), "09": ("ids", "cluster", "inertia"), "10": ("pi_test", "lut"),
        "11": ("params_f",), "12": ("keep",), "13": ("params1",), "14": ("acc1",)}
PROTOCOL = dict(load_mode="t1", hungarian_on="eval", s0_labels="true", clustering_alg="kmeans", classifier_alg="mngmm",
                estimator="svi", cov_prior_kappa=0.0, mask_unseen=False, oracle_filter=False)
DETERMINISTIC = "--xla_gpu_deterministic_ops=true"    # without it the GPU scatter-add is not bit-reproducible (A6)


def _xla_flags():
    """XLA_FLAGS with the deterministic GPU mode added; XLA reads it when jax starts."""
    flags = os.environ.get("XLA_FLAGS", "")
    return flags if DETERMINISTIC in flags else f"{flags} {DETERMINISTIC}".strip()


def new_state(args):
    """Fresh state: args and the classifier object, set up as main.py does, used by the steps as a workbench.
    The GPU runs in deterministic mode, so two runs on the same input agree to the bit; main.py on the GPU
    does not (notes/reproducibility-issues.md A6)."""
    bad = {k: getattr(args, k) for k, v in PROTOCOL.items() if getattr(args, k) != v}
    assert not bad, f"replay v2 runs the default protocol only, not {bad}"
    assert Path("main.py").exists(), "run from the repo root (relative paths: splits/, features/)"
    if os.environ.get("XLA_FLAGS") != _xla_flags():
        assert "jax" not in sys.modules, f"jax was imported before replay.new_state: start Python with XLA_FLAGS={DETERMINISTIC}"
        os.environ["XLA_FLAGS"] = _xla_flags()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # memory only, results unchanged
    import torch, jax                                                  # noqa: F401  (import order of main.py)
    import main
    clf = main.Classifier_alg(args.classifier_alg)(num_classes=args.num_classes, num_dim=args.num_dim,
                                                   with_early_stop=args.with_early_stop)
    for k in ("cov_prior_kappa", "cov_prior_target", "estimator", "mask_unseen", "oracle_filter"):
        setattr(clf, k, getattr(args, k))
    tmp = tempfile.mkdtemp(prefix="replay_")
    clf.init_parameters(n_epochs=args.n_epochs, lr=args.lr, log_dir=f"{tmp}/log", save_dir=f"{tmp}/",
                        batch_size=args.batch_size, increment=args.increment, base=args.base,
                        scaling_factor=args.scaling_factor, use_correct_scaling_factor=args.use_correct_scaling_factor,
                        early_stop_ratio=args.early_stop_ratio)
    return {"args": args, "clf": clf}


# ---------------------------------------------------------------- components (chosen at 06, 07, 08; used by 11-14)
def fit_svi(clf, z, y, z_test, y_test, prefix, correct_sf):
    """Estimator: class Gaussians by SVI (mngmm.run_inference, as in train(); PRNGKey(0) inside, no global RNG).
    The test data only feed the log lines."""
    import jax.numpy as jnp
    return clf.run_inference(jnp.array(z), jnp.array(y), jnp.array(z_test), jnp.array(y_test),
                             log_prefix=prefix, use_correct_scaling_factor=correct_sf)


def stop_default(clf, args):
    """Early-stop rule: mngmm's det(Sigma) rule, on when --with_early_stop (inactive at S0)."""
    clf.with_early_stop = args.with_early_stop
    clf.__dict__.pop("calculate_metrics_on_covariances", None)       # drop an override left by a variant


def predict_default(clf):
    """Classifier: argmax of the class log-densities over all num_classes slots (mngmm._predict)."""
    clf.__dict__.pop("_predict", None)


def _component(name):
    return globals()[name] if name in globals() else getattr(__import__("variants"), name)


def _clf(st):
    clf = st["clf"]
    clf.pca = st.get("pca")                                          # None before step 05
    return clf


def _jnp(p):
    import jax.numpy as jnp
    return {k: jnp.asarray(v) for k, v in p.items()}


def _np(p):
    return {k: np.asarray(v) for k, v in p.items()}


def _rel(st, key):
    """True labels mapped to cluster space by lut (step 10), as main.py does for evaluation."""
    return st["lut"][np.asarray(st[key]).astype(int)]


def _table(clf, rows):
    """The result table of mngmm.run(), same format (scripts/extract_vbcgcd.py reads it)."""
    from prettytable import PrettyTable
    t, acc = PrettyTable(["TestSet", "Correct", "Smaples", "Accuracy"]), {}
    for name, x, y in rows:
        correct, total, a = clf.test(x, y)
        t.add_row([name, correct, total, a])
        acc[name] = a
    print(t)
    return acc


# ---------------------------------------------------------------- steps 04-14
def s04_loader(st):
    """04 Sessions of the t1 protocol (main.Dataloader); the permutations draw the global RNG.
    In: args. Out: s0_x, s0_y (labeled S0), pool_x (S1 pool), test0_x (S0 test), test1_x (novel test),
    test_old_x, test_all_x (S1 evaluation), offset (first novel id); true_* labels of the pool and test sets."""
    import main
    a = st["args"]
    tr, te, te_old, te_all = (list(it) for it in main.Dataloader(a))
    assert len(tr) == 2, "t1: S0 and one online session"
    y = lambda d: np.asarray(d._y)
    return dict(s0_x=tr[0]._x, s0_y=y(tr[0]), pool_x=tr[1]._x, test0_x=te[0]._x, test1_x=te[1]._x,
                test_old_x=te_old[1]._x, test_all_x=te_all[1]._x, offset=a.base,
                true_pool_y=y(tr[1]), true_test0_y=y(te[0]), true_test1_y=y(te[1]),
                true_test_old_y=y(te_old[1]), true_test_all_y=y(te_all[1]))


def s05_pca(st):
    """05 PCA to num_dim, fit on the labeled S0 data (mngmm.pre_processing: fit_transform on the first call; the
    randomized solver draws the global RNG). In: s0_x, s0_y, pool_x, test0_x, test1_x.
    Out: pca, z_s0, z_pool, z_test0, z_test1 (num_dim features)."""
    clf = _clf(st)
    z_s0, _ = clf.pre_processing(st["s0_x"], st["s0_y"])
    t = clf.pca.transform
    return dict(pca=clf.pca, z_s0=z_s0, z_pool=t(st["pool_x"]), z_test0=t(st["test0_x"]), z_test1=t(st["test1_x"]))


def s06_gauss_svi(st):
    """06 Class Gaussians of S0 (mngmm.train at stage 0), default early-stop rule (inactive at S0).
    In: z_s0, s0_y; z_test0, true_test0_y (log lines only). Out: params0; fit (estimator for 11 and 13)."""
    clf, a = _clf(st), st["args"]
    stop_default(clf, a)
    clf.global_params, clf.label_offset, clf.n_seen = None, 0, clf.num_base
    p = fit_svi(clf, st["z_s0"], np.asarray(st["s0_y"]).astype(int), st["z_test0"], st["true_test0_y"],
                "stage_0_Flearning", False)
    return dict(params0=_np(p), fit=fit_svi.__name__)


def s07_early_stop(st):
    """07 Early-stop rule of the online SVI passes (used by 11 and 13). In: -. Out: stop."""
    return dict(stop=stop_default.__name__)


def s08_classifier(st):
    """08 Classifier (used by 12 and 14) and the S0 test table, printed as mngmm.run prints it.
    In: params0, test0_x, true_test0_y. Out: predict, acc0."""
    clf = _clf(st)
    predict_default(clf)
    clf.params = clf.global_params = _jnp(st["params0"])
    clf.label_offset, clf.n_seen = 0, clf.num_base
    rows = [(n, st["test0_x"], st["true_test0_y"]) for n in ("All", "Old", "Novel", "S0")]
    return dict(predict=predict_default.__name__, acc0=_table(clf, rows))


def s09_cluster(st):
    """09 k-means on the raw pool features (KMeansCluster, n_init = kmeans_n_init; random_state=None draws the
    global RNG). In: pool_x, offset. Out: ids (0..K-1, fit order), cluster (has predict_raw), inertia."""
    import main
    a = st["args"]
    cl = main.Clustering_alg(a.clustering_alg)(num_classes=a.increment, label_offset=st["offset"], n_init=a.kmeans_n_init)
    cl.fit(st["pool_x"])
    ids = cl.predict_raw(st["pool_x"])
    print(np.unique(ids + st["offset"], return_counts=True))         # cluster sizes, as main.py prints them
    return dict(ids=ids, cluster=cl, inertia=float(cl.model.inertia_))


def s10_naming(st):
    """10 Cluster names for evaluation only (D13): Hungarian on the novel test set.
    In: cluster, test1_x, true_test1_y, offset. Out: pi_test (cluster -> novel id - offset), lut (true id -> cluster space)."""
    from clustering.clustering_base import hungarian_map
    a, off = st["args"], st["offset"]
    pi = hungarian_map(st["cluster"].predict_raw(st["test1_x"]), np.asarray(st["true_test1_y"]).astype(int) - off,
                       a.increment)
    print(f"hungarian stage 1: mode=eval pi_test={pi}")
    lut = np.arange(a.num_classes)
    for c, t in pi.items():
        lut[off + t] = off + c
    return dict(pi_test=pi, lut=lut)


def s11_gauss_f(st):
    """11 Pass F: SVI on the whole pool, cluster ids as labels, started from the S0 Gaussians (mngmm.train).
    In: z_pool, ids, params0, fit, stop; z_test1, true_test1_y, lut (log lines only). Out: params_f."""
    clf, a, off = _clf(st), st["args"], st["offset"]
    _component(st["stop"])(clf, a)
    clf._set_label_offset(off)
    clf.global_params, clf.n_seen = _jnp(st["params0"]), off + a.increment
    p = _component(st["fit"])(clf, st["z_pool"], (st["ids"] + off).astype(int), st["z_test1"],
                              _rel(st, "true_test1_y"), "stage_1_Flearning", False)
    return dict(params_f=_np(p))


def s12_filter(st):
    """12 Old/new filter: keep the pool images that the pass-F classifier puts in a novel slot (mngmm.train).
    In: z_pool, params_f, predict, offset. Out: keep (bool per pool image)."""
    import jax.numpy as jnp
    clf, off = _clf(st), st["offset"]
    _component(st["predict"])(clf)
    clf.label_offset, clf.n_seen = off, off + st["args"].increment
    pred, _ = clf._predict(jnp.array(st["z_pool"]), _jnp(st["params_f"]))
    novel = pred >= off
    print(f"Number of Novel Samples: {novel.sum()} / {len(st['z_pool'])}")
    return dict(keep=np.asarray(novel))


def s13_gauss_s(st):
    """13 Pass S: SVI on the kept pool images, started again from the S0 Gaussians (mngmm.train).
    In: z_pool, ids, keep, params0, fit, stop; z_test1, true_test1_y, lut (log lines only). Out: params1."""
    clf, a, off = _clf(st), st["args"], st["offset"]
    _component(st["stop"])(clf, a)
    clf._set_label_offset(off)
    clf.global_params, clf.n_seen = _jnp(st["params0"]), off + a.increment
    k = st["keep"]
    p = _component(st["fit"])(clf, st["z_pool"][k], (st["ids"] + off).astype(int)[k], st["z_test1"],
                              _rel(st, "true_test1_y"), "stage_1_Slearning", a.use_correct_scaling_factor)
    return dict(params1=_np(p))


def s14_eval(st):
    """14 S1 test table as mngmm.run prints it; test labels mapped to cluster space by lut (D13).
    In: params1, predict, lut, test sets. Out: acc1."""
    clf, off = _clf(st), st["offset"]
    _component(st["predict"])(clf)
    clf.params = clf.global_params = _jnp(st["params1"])
    clf.label_offset, clf.n_seen = off, off + st["args"].increment
    rows = (("All", st["test_all_x"], _rel(st, "true_test_all_y")), ("Old", st["test_old_x"], _rel(st, "true_test_old_y")),
            ("Novel", st["test1_x"], _rel(st, "true_test1_y")), ("S0", st["test0_x"], st["true_test0_y"]))
    return dict(acc1=_table(clf, rows))


if __name__ == "__main__":
    from sklearn.metrics.cluster import contingency_matrix
    full = "--full" in sys.argv
    seeds = [int(a) for a in sys.argv[1:] if a != "--full"] or [0]
    ok = True
    for s in seeds:
        r = replay(s, stop=None if full else "cluster")
        ref_inertia, ref_table, ref = _reference(s)
        st = r["stages"][1]
        table = contingency_matrix(np.asarray(st["train"]._y).astype(int), st["raw"]).tolist()
        same = table == ref_table                          # the partition: what the method actually uses
        rel = abs(float(st["inertia"]) - ref_inertia) / ref_inertia
        msg = f"seed {s}: partition {'SAME' if same else 'DIFF'} | inertia {st['inertia']} vs run {ref_inertia} (rel {rel:.1e})"
        same = same and rel < 1e-6                         # float32 summation order: 1 ULP ~ 8e-8 here
        if full:
            pairs = [(st["acc"][k], ref[f"S1_{k}"]) for k in ("All", "Old", "New", "S0test")]
            same_acc = all(a == b for a, b in pairs)
            msg += f" | S1 All/Old/New/S0test {[round(a, 2) for a, _ in pairs]} -> {'SAME' if same_acc else 'DIFF'}"
            same = same and same_acc
        ok = ok and same
        print(msg)
    print("REPLAY OK" if ok else "REPLAY MISMATCH")
