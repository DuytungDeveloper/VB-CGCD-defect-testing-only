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
import argparse, ast, contextlib, functools, hashlib, inspect, io, json, os, pickle, re, subprocess, sys, tempfile, textwrap, time, types
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



# ---------------------------------------------------------------- runner, snapshots, hash chain (PL-QD6)
RNG_STEPS = ("04", "05", "09")                     # the only steps that draw the global numpy RNG
FILES = {"04": ("main.py", "dataloaders", "utils"), "05": ("classifier",), "09": ("clustering",)}
PACKAGES = ("numpy", "scipy", "scikit-learn", "jax", "jaxlib", "numpyro", "optax", "continuum")
PRESETS = {"gc10": GC10, "neudet": NEU}
LAB = Path(__file__).resolve().parent


def _fn(item):
    """Spec item 'name' or 'name:key=value,...' -> the function (this file, then lab/variants.py)."""
    name, _, kw = item.partition(":")
    f = globals()[name] if name in globals() else getattr(__import__("variants"), name)
    return functools.partial(f, **{k: ast.literal_eval(v) for k, v in (x.split("=", 1) for x in kw.split(","))}) if kw else f


def _check_out(k, item, out, default):
    if default:
        assert set(out) == set(OWNS[k]), f"step {k}: {item} returned {sorted(out)}"
        return
    others = {x for j in STEPS if j != k for x in OWNS[j]}
    bad = [x for x in out if x.startswith("true_") or x in others]
    assert not bad, f"step {k}: {item} may not write {bad}"


class _Tee(io.TextIOBase):
    """stdout to the console and to a buffer: each snapshot keeps the log lines of its step."""
    def __init__(self, out):
        self.out, self.buf = out, io.StringIO()

    def write(self, s):
        self.out.write(s)
        self.buf.write(s)
        return len(s)

    def flush(self):
        self.out.flush()


def _check_09(st):
    """After step 09, whatever functions ran there (QD21, 2026-10-20): the pool ids and the ids the cluster model
    gives the novel test set must be integers in [0, increment), one per image. Out-of-range ids raise no error
    later: confusion_matrix drops them at 10 (hungarian_map) and JAX indexing clamps them at 11 and 13."""
    inc = st["args"].increment
    for name, x, ids in (("ids", st["pool_x"], st["ids"]),
                         ("cluster.predict_raw(test1_x)", st["test1_x"], st["cluster"].predict_raw(st["test1_x"]))):
        ids = np.asarray(ids)
        ok = ids.ndim == 1 and len(ids) == len(x) and np.issubdtype(ids.dtype, np.integer) \
            and ids.min() >= 0 and ids.max() < inc
        assert ok, f"step 09: {name} must be {len(x)} integers in [0, {inc}), got shape {ids.shape}, " \
                   f"dtype {ids.dtype}, values {np.unique(ids)[:10]}"


def chain(st, spec=None, start="04", parent=None, snap=None, meta=None, upto="14"):
    """Run steps start..upto on st, in place (st is also returned).
    spec: {step: [items]}, default DEFAULT. The items of a step run in order: a variant replaces the default
    function or comes after it (lab/variants.py). Variants get st without the true_* keys.
    parent: {step: snapshot} of a chain made by this runner; at the entry of 04, 05, 09 the global RNG state is
    set to the parent's (common random numbers). Without a parent, 04 starts from the seed, as main.py does.
    snap: folder for one snapshot per step: out, rng_in, log, hash, spec, meta."""
    spec = spec or {k: [DEFAULT[k]] for k in STEPS}
    a = st["args"]
    hashes = step_hashes(spec, _base(a)) if snap is not None else {}
    if parent is None and start == "04":
        import torch, jax
        torch.manual_seed(a.seed); np.random.seed(a.seed); jax.random.PRNGKey(a.seed)     # main.py order
    for k in STEPS[STEPS.index(start):STEPS.index(upto) + 1]:
        if parent is not None and k in RNG_STEPS:
            np.random.set_state(parent[k]["rng_in"])
        rng_in, new, tee = np.random.get_state(), {}, _Tee(sys.stdout)
        sys.stdout = tee
        try:
            for item in spec[k]:
                default = item == DEFAULT[k]
                out = _fn(item)(st if default else {x: v for x, v in st.items() if not x.startswith("true_")})
                _check_out(k, item, out, default)
                st.update(out)
                new.update(out)
        finally:
            sys.stdout = tee.out
        if k == "09":
            _check_09(st)
        if snap is not None:
            blob = dict(step=k, out=new, rng_in=rng_in, log=tee.buf.getvalue(), hash=hashes[k], spec=spec, meta=meta or {})
            (Path(snap) / f"{k}.pkl").write_bytes(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
    return st


def _src_hash(src):
    """sha256 of the syntax tree without docstrings: comments and layout do not count."""
    tree = ast.parse(textwrap.dedent(src))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()[:16]


def _closure(fn, override=None, out=None):
    """Hashes of fn and of what it reaches in lab/ by global name: functions and classes (also inside dicts,
    lists, tuples) by syntax tree, plain constants by value. Keys file:name, the same in a script and a notebook."""
    out = {} if out is None else out
    fn = getattr(fn, "func", fn)                                     # functools.partial from a spec item
    key = f"{Path(inspect.getsourcefile(fn)).name}:{fn.__qualname__}"
    if key in out:
        return out
    out[key] = (override or {}).get(key) or _src_hash(inspect.getsource(fn))
    if inspect.isclass(fn):
        return out
    codes = [fn.__code__]
    while codes:
        c = codes.pop()
        codes += [x for x in c.co_consts if isinstance(x, types.CodeType)]
        for n in c.co_names:
            if n not in fn.__globals__ or n.startswith("__"):          # __name__ differs: script vs import
                continue
            obj = fn.__globals__[n]
            items = list(obj.values()) if isinstance(obj, dict) else list(obj) if isinstance(obj, (list, tuple)) else [obj]
            code = [o for o in items if inspect.isfunction(o) or inspect.isclass(o)]
            for o in code:
                try:
                    src = inspect.getsourcefile(o)
                except TypeError:                                    # built-in
                    src = None
                if src and Path(src).resolve().parent == LAB:
                    _closure(o, override, out)
            if not code and isinstance(obj, (str, int, float, tuple, list, dict, frozenset)):
                ckey = f"{Path(inspect.getsourcefile(fn)).name}:={n}"
                out[ckey] = (override or {}).get(ckey) or repr(obj)
    return out


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _file_hash(rel):
    """Syntax-tree hash of a pipeline .py file, or of every .py file under a folder."""
    p = ROOT / rel
    h = hashlib.sha256()
    for f in (sorted(p.rglob("*.py")) if p.is_dir() else [p]):
        h.update(f"{f.relative_to(ROOT)}\0{_src_hash(f.read_text())}\0".encode())
    return h.hexdigest()[:16]


def _base(args):
    """Inputs of step 04 besides code: configuration, features, split file, versions of the numeric packages."""
    import importlib, importlib.metadata as md
    f = Path(args.data_dir) / args.pretrained_model_name
    feats = {p.name: _sha_file(p) for p in sorted(f.glob("*.npy"))}
    split = getattr(importlib.import_module(f"dataloaders.{args.dataset}"), "SPLIT", None)

    def version(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None
    return {"args": sorted((k, str(v)) for k, v in vars(args).items()), "features": feats,
            "split": _sha_file(ROOT / split) if split else None, "packages": {p: version(p) for p in PACKAGES},
            "python": sys.version.split()[0], "xla_flags": _xla_flags()}


def step_hashes(spec, base, override=None):
    """h(k) = sha256(h(k-1), code of the items at step k with what they reach in lab/, pipeline files first used
    at k); h(04) also covers base and the runner (chain, new_state). Changing step k changes h(k)..h(14) only."""
    out, prev = {}, ""
    for k in STEPS:
        parts = {"prev": prev, "items": [(i, _closure(_fn(i), override)) for i in spec[k]],
                 "files": {f: _file_hash(f) for f in FILES.get(k, ())}}
        if k == "04":
            parts.update(base=base, runner=[_closure(chain, override), _closure(new_state, override)])
        prev = out[k] = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return out


def _run_dir(run):
    """Run folder from a run id or a tag (runs/*__<tag>, must be unique)."""
    d = ROOT / "runs" / run
    if d.is_dir():
        return d
    found = sorted((ROOT / "runs").glob(f"*__{run}"))
    assert len(found) == 1, f"run {run!r}: {len(found)} folders match"
    return found[0]


def snapshots(run):
    """{step: snapshot} of a v2 run; the steps it did not run come from its parent run."""
    d = _run_dir(run) / "snap"
    blobs = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(d.glob("*.pkl"))}
    assert blobs, f"no snapshots in {d}"
    parent = next(iter(blobs.values()))["meta"].get("parent")
    return {**{k: v for k, v in snapshots(parent).items() if k not in blobs}, **blobs} if parent else blobs


def _fresh(blobs, upto="14"):
    """Refuse snapshots made from a dirty tree, and stale ones: the hash chain recomputed from the code now."""
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    now = step_hashes(blobs[upto]["spec"], _base(args))
    for k in STEPS[:STEPS.index(upto) + 1]:
        if blobs[k]["meta"].get("dirty"):
            raise SystemExit(f"replay: snapshot {k} was made from a dirty tree")
        if now[k] != blobs[k]["hash"]:
            raise SystemExit(f"replay: snapshots stale from step {k} (code, features or packages changed): "
                             f"run the chain again from step {k}")


def load(run, upto="14", check=True):
    """State after step `upto` of a v2 run (runs/<run>/snap), with a fresh workbench (new_state)."""
    blobs = snapshots(run)
    if check:
        _fresh(blobs, upto)
    st = new_state(types.SimpleNamespace(**blobs["04"]["meta"]["args"]))
    for k in STEPS[:STEPS.index(upto) + 1]:
        st.update(blobs[k]["out"])
    return st


# ---------------------------------------------------------------- commands: run, check (C1), C2, C3
def _git():
    sh = lambda *c: subprocess.run(c, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    status = sh("git", "status", "--porcelain", "--", ".", ":(exclude)runs", ":(exclude)notes",
                ":(exclude)lab/*.ipynb", ":(exclude)lab/*.md")         # the same rule as scripts/exp.py
    return {"commit": sh("git", "rev-parse", "HEAD"), "dirty": bool(status)}


def _metrics(run_dir):
    """metrics.json from console.log with scripts/extract_vbcgcd.py: one definition of the metrics."""
    import importlib.util
    print("replay v2: done", flush=True)
    log = run_dir / "console.log"
    for _ in range(100):                                             # exp.py copies our output as it comes
        if log.exists() and "replay v2: done" in log.read_text(errors="replace"):
            break
        time.sleep(0.1)
    spec = importlib.util.spec_from_file_location("extract_vbcgcd", ROOT / "scripts" / "extract_vbcgcd.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    m = ex.extract(ex.clean_lines(log.read_text(errors="replace")))
    (run_dir / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    for k, v in m.items():
        print(f"{k:22s} {v}")


def cmd_run(argv):
    """Default chain:  exp.py run TAG --inputs F -- python lab/replay.py run --features F --seed S
    Variant:        exp.py run TAG --inputs runs/<parent>/snap -- python lab/replay.py run --from <parent>
                        --start 09 --at 09=v09a_overcluster+v09b_merge"""
    p = argparse.ArgumentParser(prog="replay.py run")
    p.add_argument("--features", help="features/<run_id>/<model>")
    p.add_argument("--seed", type=int)
    p.add_argument("--dataset", default="gc10", choices=sorted(PRESETS))
    p.add_argument("--set", action="append", default=[], help="key=value: change one setting (default chain only)")
    p.add_argument("--from", dest="parent", help="run id or tag of a v2 run: steps before --start come from it")
    p.add_argument("--start", default="04", choices=STEPS)
    p.add_argument("--at", action="append", default=[], help="NN=f1+f2: the functions of step NN, in order")
    o = p.parse_args(argv)
    if not os.environ.get("EXP_RUN_DIR"):
        sys.exit("replay run: run it through scripts/exp.py (the run folder holds the snapshots)")
    run_dir, git = Path(os.environ["EXP_RUN_DIR"]), _git()
    if git["dirty"]:
        sys.exit("replay run: commit the code first (same dirty rule as scripts/exp.py)")
    at = {k: v.split("+") for k, v in (x.split("=", 1) for x in o.at)}
    assert all(k in STEPS and k >= o.start for k in at), f"--at only for steps from --start {o.start}: {at}"
    parent = None
    if o.parent:
        assert not (o.features or o.seed is not None or o.set), "--from: features, seed and settings come from the parent"
        parent = snapshots(o.parent)
        _fresh(parent)                       # this run is compared with the parent: it must match the code now
        spec = {k: at.get(k, [DEFAULT[k]]) if k >= o.start else parent[k]["spec"][k] for k in STEPS}
        prev = STEPS[STEPS.index(o.start) - 1] if o.start != "04" else None
        st = load(o.parent, prev, check=False) if prev else new_state(types.SimpleNamespace(**parent["04"]["meta"]["args"]))
    else:
        if at or o.start != "04":
            sys.exit("replay run: variants and --start run from the snapshots of a chain: give --from <run>")
        over = {k: ast.literal_eval(v) if v[:1].isdigit() or v[:1] in "-[{(" or v in ("True", "False") else v
                for k, v in (x.split("=", 1) for x in o.set)}
        spec, st = None, new_state(make_args(o.features, o.seed, PRESETS[o.dataset], **over))
    import jax                                                       # loaded by new_state, after XLA_FLAGS
    device = jax.default_backend()
    if device != "gpu" and os.environ.get("REPLAY_ALLOW_CPU") != "1":
        sys.exit(f"replay run: JAX runs on {device}, not on the GPU (A1); REPLAY_ALLOW_CPU=1 is for tests only")
    if parent is not None:                   # snapshots made before 2026-10-21 have no device field: all ran on the GPU
        seen = {b["meta"].get("device", "gpu") for b in parent.values()}
        if seen != {device}:
            sys.exit(f"replay run: parent snapshots ran on {sorted(seen)}, this run on {device}: numbers would mix")
    meta = dict(git, run_id=run_dir.name, parent=_run_dir(o.parent).name if o.parent else None, start=o.start,
                device=device, args=vars(st["args"]))
    changed = {k: v for k, v in (spec or {}).items() if v != [DEFAULT[k]]}
    print(f"replay v2: {run_dir.name} | commit {git['commit'][:7]} | parent {meta['parent']} | start {o.start} | "
          f"device {device} | changed {changed}")
    for k in STEPS[:STEPS.index(o.start)]:                           # log lines of the steps taken from the parent
        sys.stdout.write(parent[k]["log"])
    (run_dir / "snap").mkdir()
    chain(st, spec, o.start, parent, run_dir / "snap", meta)
    _metrics(run_dir)


def _confusion(st):
    """Confusion matrix of the S1 test set, rows true ids, columns predicted true ids, as main.py prints it."""
    import jax.numpy as jnp
    from sklearn.metrics import confusion_matrix
    clf, off = _clf(st), st["offset"]
    _component(st["predict"])(clf)
    clf.label_offset, clf.n_seen = off, off + st["args"].increment
    pr, _ = clf._predict(jnp.array(clf.pca.transform(st["test_all_x"])), _jnp(st["params1"]))
    return confusion_matrix(np.asarray(st["true_test_all_y"]).astype(int), np.argsort(st["lut"])[np.asarray(pr)],
                            labels=np.arange(off + st["args"].increment))


def cmd_check(argv):
    """C1: python lab/replay.py check <v2 run> <main.py run>: equal at every snapshot (PL-QD6)."""
    from sklearn.metrics.cluster import contingency_matrix
    rd, fd = _run_dir(argv[0]), _run_dir(argv[1])
    blobs = snapshots(rd.name)
    _fresh(blobs)
    st = load(rd.name, check=False)
    a, m, mr = st["args"], json.loads((rd / "metrics.json").read_text()), json.loads((fd / "metrics.json").read_text())
    cfg = json.loads((fd / "config.json").read_text())
    log = (fd / "console.log").read_text(errors="replace") if (fd / "console.log").exists() else ""
    find = lambda pat: (lambda r: r.groups() if r else None)(re.search(pat, log))
    res = []
    seed = re.findall(r"--seed (\d+)", " ".join(cfg["command"]))
    feats = list(cfg.get("inputs") or {})
    res.append(("input: seed, features", int(seed[-1]) == a.seed and len(feats) == 1 and Path(feats[0]).resolve()
                == (Path(a.data_dir) / a.pretrained_model_name).resolve() if seed and feats else None))
    sm = fd / "artifacts" / "saved_models"
    for stage, key in (("0", "params0"), ("1", "params1")):
        for f, k in (("class_means", "class_means"), ("class_covariances", "class_covs")):
            p = sm / f"stage{stage}{f}.npy"
            ref = np.load(p) if p.exists() else None                 # GPU noise of main.py: ~1e-7 (A6)
            res.append((f"{key} {k} (max|diff| <= 1e-6 max|value|)", bool(np.abs(st[key][k].astype(np.float64) - ref).max()
                        <= 1e-6 * np.abs(ref).max()) if ref is not None else None))
    res.append(("metrics.json (tables S0, S1; cluster sizes; kept; early stop)", m == mr))
    y = np.asarray(st["true_pool_y"]).astype(int)
    x = find(r"xtab stage 1 train: rows=(\[[^\]]*\]) cols=(\[[^\]]*\]) (\[\[.*\]\])")
    res.append(("partition (contingency table)", [np.unique(y).tolist(), np.unique(st["ids"]).tolist(),
                contingency_matrix(y, st["ids"]).tolist()] == [ast.literal_eval(v) for v in x] if x else None))
    i = find(r"xtab stage 1 objective: inertia=(\S+)")
    res.append(("inertia (relative < 1e-6)", abs(st["inertia"] - float(i[0])) / float(i[0]) < 1e-6 if i else None))
    pi = find(r"hungarian stage 1: mode=eval pi_test=(\{[^}]*\})")
    res.append(("pi_test", ast.literal_eval(pi[0]) == st["pi_test"] if pi else None))
    res.append(("kept", int(st["keep"].sum()) == mr.get("S1_novel_kept")))
    c = find(r"confusion stage 1: rows=true cols=pred (\[\[.*\]\])")
    res.append(("confusion S1", _confusion(st).tolist() == ast.literal_eval(c[0]) if c else None))
    for name, ok in res:
        print(f"  {'OK  ' if ok else 'n/a ' if ok is None else 'DIFF'} {name}")
    if m != mr:
        print("  metrics differ:", {k: (m.get(k), mr.get(k)) for k in set(m) | set(mr) if m.get(k) != mr.get(k)})
    verdict = "DIFF" if any(ok is False for _, ok in res) else "INCOMPLETE" if any(ok is None for _, ok in res) else "OK"
    print(f"C1 {verdict}: {rd.name} = {fd.name}")


def _same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if hasattr(a, "components_"):                                    # PCA
        return _same({k: v for k, v in vars(a).items()}, {k: v for k, v in vars(b).items()})
    if hasattr(a, "model"):                                          # KMeansCluster: OpenMP sums, ULP-level
        return np.allclose(a.model.cluster_centers_, b.model.cluster_centers_, rtol=1e-6, atol=1e-6)
    if isinstance(a, float):
        return a == b or abs(a - b) <= 1e-6 * abs(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def cmd_check_resume(argv):
    """C2: python lab/replay.py check-resume <v2 run> [05,06,...]: continuing from the snapshot before step k, in a
    new process as a variant run does, gives the same snapshots k..14 as the run itself."""
    run, ks = argv[0], argv[1].split(",") if len(argv) > 1 else list(STEPS[1:])
    assert all(k in STEPS[1:] for k in ks), f"steps 05..14 only: {ks}"
    blobs = snapshots(run)
    _fresh(blobs)
    if len(ks) > 1:                                                  # one process per step
        me = [sys.executable, str(Path(__file__).resolve()), "check-resume", run]
        bad = [k for k in ks if subprocess.run(me + [k]).returncode]
        print(f"C2 {'OK' if not bad else 'DIFF'}: {run}")
        return
    k = ks[0]
    st = load(run, STEPS[STEPS.index(k) - 1], check=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        chain(st, blobs["14"]["spec"], k, blobs)
    diff = [x for j in STEPS[STEPS.index(k):] for x in blobs[j]["out"] if not _same(st[x], blobs[j]["out"][x])]
    print(f"  {'OK  ' if not diff else 'DIFF'} continue from {k} {diff or ''}", flush=True)
    sys.exit(1 if diff else 0)


def _burn05(st):
    """C3 only: draw extra numbers from the global RNG, then step 05 as usual."""
    np.random.rand(1000)
    return s05_pca(st)


def cmd_check_crn(argv):
    """C3: python lab/replay.py check-crn <v2 run>: a variant at 05 that draws extra RNG changes the PCA but not
    the k-means at 09, which reads the raw pool features (common random numbers). Control: the same k-means with
    the RNG shifted gives another result, so this check can fail."""
    run = argv[0]
    blobs = snapshots(run)
    _fresh(blobs)
    st = load(run, "04", check=False)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        chain(st, {**blobs["14"]["spec"], "05": ["_burn05"]}, "05", blobs, upto="10")
        np.random.set_state(blobs["09"]["rng_in"])
        np.random.rand(1000)                                         # control: the RNG at 09 without CRN
        ctl = s09_cluster(st)
    pca_moved = not _same(st["z_s0"], blobs["05"]["out"]["z_s0"])
    same = {x: _same(st[x], blobs[j]["out"][x]) for x, j in (("ids", "09"), ("inertia", "09"), ("pi_test", "10"))}
    power = not all(_same(ctl[x], blobs["09"]["out"][x]) for x in ("ids", "inertia"))
    print(f"  {'OK  ' if pca_moved else 'DIFF'} PCA changed by the extra draws")
    for x, ok in same.items():
        print(f"  {'OK  ' if ok else 'DIFF'} {x} unchanged")
    print(f"  {'OK  ' if power else 'n/a '} control: k-means with the RNG shifted gives another result")
    ok = pca_moved and all(same.values())
    print(f"C3 {'DIFF' if not ok else 'OK' if power else 'INCOMPLETE'}: {run}")


# ---------------------------------------------------------------- command C4
def cmd_check_hash(argv):
    """C4: python lab/replay.py check-hash: changing step k makes h(k)..h(14) change and nothing before;
    comments and docstrings do not count. No data, a few seconds."""
    spec, base = {k: [DEFAULT[k]] for k in STEPS}, {"C4": "fixed inputs"}
    h, ok = step_hashes(spec, base), True
    cases = [(f"edit {DEFAULT[k]}", f"replay.py:{DEFAULT[k]}", k) for k in STEPS] + [
        ("edit fit_svi (06, then 11 and 13 by name)", "replay.py:fit_svi", "06"),
        ("edit _rel (11, 13, 14)", "replay.py:_rel", "11"),
        ("edit the runner (chain)", "replay.py:chain", "04"),
        ("edit new_state", "replay.py:new_state", "04"),
        ("edit _check_09 (the runner's check after step 09)", "replay.py:_check_09", "04"),
        ("edit DEFAULT (a constant the runner reads)", "replay.py:=DEFAULT", "04")]
    for label, key, first in cases:
        h2 = step_hashes(spec, base, {key: "edited"})
        good = all((h2[k] == h[k]) == (STEPS.index(k) < STEPS.index(first)) for k in STEPS)
        print(f"  {'OK  ' if good else 'DIFF'} {label}: stale from {first}")
        ok &= good
    src = inspect.getsource(s12_filter)
    cosmetic = src.replace('"""12 Old/new', '"""12 (edited docstring) Old/new').replace(
        "novel = pred >= off", "novel = pred >= off      # a comment")
    real = src.replace("novel = pred >= off", "novel = pred > off")
    good = cosmetic != src and _src_hash(cosmetic) == _src_hash(src) and _src_hash(real) != _src_hash(src)
    print(f"  {'OK  ' if good else 'DIFF'} comments and docstrings do not count, code does")
    print(f"C4 {'OK' if ok and good else 'DIFF'}")


COMMANDS = {"run": cmd_run, "check": cmd_check, "check-resume": cmd_check_resume, "check-crn": cmd_check_crn,
            "check-hash": cmd_check_hash}


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
    COMMANDS[sys.argv[1]](sys.argv[2:])
elif __name__ == "__main__":                                 # v1 self-check (tag nb-v1)
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
