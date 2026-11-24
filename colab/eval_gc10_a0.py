#!/usr/bin/env python
"""Evaluate Step A0 (duytung/02-gc10-tang-epoch-finetune): compare FT_EPOCHS=10, 20, 30 on GC10-DET."""
import argparse, contextlib, io, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab"))

import niw
import sessions
import score
from replay import _clf, _component, _jnp, s09_cluster


def eval_tag(tag):
    """Return dict of P, NL, FTR1 metrics (bal, Old, All) for one gc10 sessions run."""
    blobs = sessions.snapshots_sessions(tag)
    rng = blobs["S1_09"]["rng_in"]
    del blobs
    st, views = sessions.load(tag, check=False)
    import jax.numpy as jnp
    v = views[0]
    a = st["args"]
    base, inc, off = a.base, a.increment, int(v["offset"])

    zt = np.concatenate([np.asarray(v["z_test1"], np.float64), np.asarray(st["z_test0"], np.float64)])
    y1 = np.asarray(v["true_test1_y"]).astype(int)
    y = np.concatenate([y1, np.asarray(st["true_test0_y"]).astype(int)])
    lut = np.asarray(v["lut"])

    # P (pipeline SVI)
    clf = _clf(st)
    _component(st["predict"])(clf)
    clf.label_offset, clf.n_seen = off, off + inc
    Lp = np.concatenate([
        np.asarray(clf._predict(jnp.asarray(v["z_test1"]), _jnp(v["params1"]))[1], np.float64),
        np.asarray(clf._predict(jnp.asarray(st["z_test0"]), _jnp(v["params1"]))[1], np.float64),
    ])
    out = {"P": score._sc_1204(Lp, y, lut[y], off)}

    # NIW prior + base slots from S0
    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    d = z0.shape[1]
    off0 = niw.choose(niw.loo_curve(z0, y0))
    sts = niw.class_stats(z0, y0)
    prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
    base_rows = [z0[y0 == k] for k in range(base)]
    base_slots = [score._post_1406(R, prior) for R in base_rows]
    idx = score._index_1203(base_rows)

    zp, X = np.asarray(v["z_pool"], np.float64), np.asarray(v["pool_x"])
    ids, keep = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)

    lb = score._logd_1202("A3", base_slots, zt)

    # NL
    nl_rows = []
    for c in range(inc):
        rr = np.flatnonzero(keep & (ids == c))
        nl_rows.append(zp[rr[score._nl_keep_0910(zp[rr], base_rows, base_slots, prior, idx)]] if len(rr) else zp[:0])
    L_nl = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in nl_rows], zt)], 1)
    out["NL"] = score._sc_1204(L_nl, y, lut[y], off)

    # FTR1
    G = np.zeros(len(zp), bool)
    for c in range(inc):
        rr = np.flatnonzero(keep & (ids == c))
        if len(rr):
            G[rr[score._nl_keep_0910(zp[rr], base_rows, base_slots, prior, idx)]] = True
    ri = np.flatnonzero(G)
    np.random.set_state(rng)
    with contextlib.redirect_stdout(io.StringIO()):
        cl = s09_cluster(dict(args=a, offset=off, pool_x=X[ri]))
    lab = np.asarray(cl["ids"]).astype(int)
    ftr_rows = []
    for c in range(inc):
        rc = ri[lab == c]
        ftr_rows.append(zp[rc[score._nl_keep_0910(zp[rc], base_rows, base_slots, prior, idx)]] if len(rc) else zp[:0])
    lut_ftr = score._lut_1205(a, off, cl["cluster"], v["test1_x"], y1)
    L_ftr = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in ftr_rows], zt)], 1)
    out["FTR1"] = score._sc_1204(L_ftr, y, lut_ftr[y], off)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="10,20,30")
    ap.add_argument("--fts", default="0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out-json", default=None)
    o = ap.parse_args()

    epochs = [int(x) for x in o.epochs.split(",") if x]
    fts = [int(x) for x in o.fts.split(",") if x]
    seeds = [int(x) for x in o.seeds.split(",") if x]

    summary = {}
    print(f"{'Setting':<14} {'Arm':<5} {'bal (mean±std)':>18} {'Old (%)':>10} {'All (%)':>10} {'N':>4}")
    print("-" * 66)
    for ep in epochs:
        rows = {"P": [], "NL": [], "FTR1": []}
        for r in fts:
            for s in seeds:
                tag = f"colab_gc10_ep{ep}_r{r}_s{s}"
                try:
                    res = eval_tag(tag)
                    for arm in ("P", "NL", "FTR1"):
                        rows[arm].append(res[arm])
                except Exception as e:
                    print(f"[SKIP {tag}: {e}]")
        summary[f"ep{ep}"] = {}
        for arm in ("P", "NL", "FTR1"):
            if not rows[arm]:
                continue
            bals = [x["bal"] for x in rows[arm]]
            olds = [x["Old"] for x in rows[arm]]
            alls = [x["All"] for x in rows[arm]]
            summary[f"ep{ep}"][arm] = {
                "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
                "Old_mean": float(np.mean(olds)), "All_mean": float(np.mean(alls)),
                "n": len(bals),
            }
            print(f"{f'ep={ep}':<14} {arm:<5} {np.mean(bals):7.4f} ± {np.std(bals):6.4f} {np.mean(olds):10.2f} {np.mean(alls):10.2f} {len(bals):4d}")
    if o.out_json:
        Path(o.out_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
