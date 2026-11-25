#!/usr/bin/env python
"""Evaluate Step A2 (duytung/06-gc10-backbone-dinov2): DINOv2-with-registers on GC10-DET (FT_EPOCHS=20)."""
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
    """Return dict of P, N, NL, FTR1, Npre metrics (bal, Old, All) for one gc10 sessions run."""
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
    lb = score._logd_1202("A3", base_slots, zt)

    zp, X = np.asarray(v["z_pool"], np.float64), np.asarray(v["pool_x"])
    ids, keep = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)
    yp = np.asarray(v["true_pool_y"]).astype(int)

    # N (no NL filter)
    n_rows = [zp[keep & (ids == c)] for c in range(inc)]
    L_n = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in n_rows], zt)], 1)
    out["N"] = score._sc_1204(L_n, y, lut[y], off)

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

    # Npre (oracle novel pool rows)
    rn = np.flatnonzero(yp >= off)
    np.random.set_state(rng)
    with contextlib.redirect_stdout(io.StringIO()):
        cl_n = s09_cluster(dict(args=a, offset=off, pool_x=X[rn]))
    lab_n = np.asarray(cl_n["ids"]).astype(int)
    npre_rows = [zp[rn[lab_n == c]] for c in range(inc)]
    lut_npre = score._lut_1205(a, off, cl_n["cluster"], v["test1_x"], y1)
    L_npre = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in npre_rows], zt)], 1)
    out["Npre"] = score._sc_1204(L_npre, y, lut_npre[y], off)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="colab_gc10_dinov2reg_ep20")
    ap.add_argument("--model", default="dinov2_reg_vitb14")
    ap.add_argument("--fts", default="0")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--meta-json", default=None)
    ap.add_argument("--out-json", default=None)
    o = ap.parse_args()

    fts = [int(x) for x in o.fts.split(",") if x]
    seeds = [int(x) for x in o.seeds.split(",") if x]
    arms = ("P", "N", "NL", "FTR1", "Npre")
    rows = {a: [] for a in arms}
    for r in fts:
        for s in seeds:
            tag = f"{o.prefix}_r{r}_s{s}"
            try:
                res = eval_tag(tag)
                for a in arms:
                    rows[a].append(res[a])
            except Exception as e:
                print(f"[SKIP {tag}: {e}]")

    summary = {"prefix": o.prefix, "model": o.model, "fts": fts, "seeds": seeds, "arms": {}}
    if o.meta_json and Path(o.meta_json).exists():
        summary["telemetry"] = json.loads(Path(o.meta_json).read_text())

    print(f"=== KẾT QUẢ BƯỚC A2 (GC10-DET Backbone {o.model}, prefix={o.prefix}) ===")
    print(f"{'Arm':<6} {'bal (mean±std)':>18} {'Old (%)':>10} {'All (%)':>10} {'N':>4}")
    print("-" * 54)
    for a in arms:
        if not rows[a]:
            continue
        bals = [x["bal"] for x in rows[a]]
        olds = [x["Old"] for x in rows[a]]
        alls = [x["All"] for x in rows[a]]
        summary["arms"][a] = {
            "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
            "Old_mean": float(np.mean(olds)), "All_mean": float(np.mean(alls)),
            "n": len(bals),
        }
        print(f"{a:<6} {np.mean(bals):7.4f} ± {np.std(bals):6.4f} {np.mean(olds):10.2f} {np.mean(alls):10.2f} {len(bals):4d}")

    if "FTR1" in summary["arms"]:
        bal_ftr = summary["arms"]["FTR1"]["bal_mean"]
        all_ftr = summary["arms"]["FTR1"]["All_mean"]
        old_ftr = summary["arms"]["FTR1"]["Old_mean"]
        print("-" * 54)
        print(f"So với A0 DINOv1 (FTR1 bal=0.4089, Old=80.75%, All=71.47%): "
              f"Δbal={bal_ftr - 0.4089:+.4f} | ΔOld={old_ftr - 80.75:+.2f}% | ΔAll={all_ftr - 71.47:+.2f}%")
        verdict = "VƯỢT A0 (DINOv1) -> Đề xuất xác nhận held-out seeds & chốt vào main" if (bal_ftr > 0.4089 and all_ftr > 71.47) else "CẦN PHÂN TÍCH CHI TIẾT"
        print(f"Quyết định Bước A2: {verdict}")
        summary["verdict"] = verdict

    if o.out_json:
        Path(o.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
