#!/usr/bin/env python
"""Replay main.py stage by stage for one seed (t1 protocol, --hungarian_on eval, labeled S0) and keep the
intermediate objects, for the notebooks in notes/nb and offline experiments of the PL line.

Same calls in the same order as main.py, so the global numpy RNG (loader permutations, the S0 PCA with the
randomized solver, k-means) is consumed identically. main.py itself is not changed.
  r = replay(seed)                   # full: S0 + S1 classifier, about 2 min on the laptop
  r = replay(seed, stop="cluster")   # stop right after the S1 clustering (the S0 SVI still runs)
  r["stages"][i]: train, test, test_old, test_all (DataPoint ._x ._y); for i >= 1 also cluster, raw (cluster ids),
                  inertia, pi_test (cluster -> novel offset), lut, offset, acc (full mode only)
  r["svi"]: (log_prefix, params) of every SVI pass (S0, then F and S of each session); r["clf"]: the classifier
Check against runs of the pipeline (default: the E1 runs gc10_frozen_km_evhung_d64_head_s*):
  python scripts/replay.py 0 1 2 3 4      # stop="cluster": inertia must equal the run's
  python scripts/replay.py --full 0       # full: S1 accuracies must equal the run's metrics.json
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
