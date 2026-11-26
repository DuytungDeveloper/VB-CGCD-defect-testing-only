#!/usr/bin/env python
"""Step A10 (duytung/14-neudet-dinov3-autoroi): Cross-validation of A8 Champion
(DINOv3 Auto-ROI Fusion + Imbalanced-DYDM + NIW Memory) on NEU-DET (6 steel defect classes, b3t1).

Evaluates both:
  - DINOv3 Full baseline (N, NL) if --full-prefix is provided
  - DINOv3 Auto-ROI Fusion (N, NL) + Imbalanced-DYDM (A10_Imb_head, A10_Imb_km, A10_Imb_blend50, A10_Imb_blend70, A10_Imb_blend50_NL)
Prints plain-Vietnamese summary + per-class novel accuracy table (pitted_surface, rolled-in_scale, scratches).
"""
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "lab"))

import niw
import replay
import score

HF_DINOV3 = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _get_feat(out):
    po = getattr(out, "pooler_output", None)
    return po.float() if po is not None else out.last_hidden_state[:, 0].float()


def load_merged_s0_backbone(ft_ckpt_path, device):
    from peft import BOFTConfig, get_peft_model
    from transformers import AutoModel

    base = AutoModel.from_pretrained(HF_DINOV3)
    candidates = ["output.dense", "mlp.fc1", "mlp.fc2", "o_proj", "up_proj", "down_proj"]
    lin_names = [n for n, m in base.named_modules() if isinstance(m, nn.Linear)]
    targets = [t for t in candidates if any(n.endswith(t) for n in lin_names)]
    print(f"A10 load_merged_s0_backbone({HF_DINOV3}): BOFT targets = {targets}", flush=True)

    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=targets,
        boft_dropout=0.1,
        bias="boft_only",
    )
    backbone = get_peft_model(base, peft_config)
    raw = torch.load(ft_ckpt_path, map_location="cpu")
    state = raw["state"] if isinstance(raw, dict) and "state" in raw else raw
    b_state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    if not b_state:
        b_state = state
    backbone.load_state_dict(b_state, strict=False)
    if hasattr(backbone, "merge_and_unload"):
        backbone = backbone.merge_and_unload()
    return backbone.to(device).eval()


def build_feature_to_path(feat_dir, split_path="splits/neu_split.json"):
    split = json.load(open(split_path))
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        names = split["train"][c]
        paths += [f"{split['root']}/{n}" if "/" in n else f"{split['root']}/train/images/{c}/{n}" for n in names]
        labels += [k] * len(names)
    feats = np.load(f"{feat_dir}/features-dinov3-vitb16-sl.npy")
    labs = np.load(f"{feat_dir}/labels-dinov3-vitb16-sl.npy")
    assert len(paths) == len(feats) and np.array_equal(np.asarray(labels), labs), \
        "NEU-DET A10 feature-to-path mapping mismatch!"
    return {feats[i].astype(np.float32).tobytes(): paths[i] for i in range(len(feats))}


class _UnlabeledPairDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, frozen_feats, weak_tfm, strong_tfm):
        self.paths = img_paths
        self.ff = frozen_feats
        self.weak = weak_tfm
        self.strong = strong_tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            return self.weak(im), self.strong(im), self.ff[i]


def _make_transforms():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    weak = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    strong = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomAffine(degrees=5, translate=(0.04, 0.04), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return weak, strong


def train_imbal_discovery_branch(base_backbone, img_paths, frozen_x, init_centroids, pi_prior, inc, device,
                                 epochs=12, batch_size=64, lr=5e-5, prior_w=0.3, kd_w=2.0, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = copy.deepcopy(base_backbone).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = (
            ".layer.11." in n or ".layers.11." in n
            or "layernorm" in n or n.startswith("norm.") or ".norm." in n or "pooler" in n
        )
    assert any(p.requires_grad for p in model.parameters())

    c0 = F.normalize(torch.tensor(init_centroids, dtype=torch.float32, device=device), dim=-1)
    head_W = nn.Parameter(c0.clone())
    pi_t = torch.tensor(pi_prior, dtype=torch.float32, device=device)

    weak_tfm, strong_tfm = _make_transforms()
    ff_norm = F.normalize(torch.tensor(frozen_x, dtype=torch.float32), dim=-1)
    ds = _UnlabeledPairDataset(img_paths, ff_norm, weak_tfm, strong_tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad] + [head_W]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    scale = 10.0

    model.train()
    for _ in range(epochs):
        for xw, xs, xf in dl:
            xw, xs, xf = xw.to(device), xs.to(device), xf.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    zw = F.normalize(_get_feat(model(xw)), dim=-1)
                    Wn = F.normalize(head_W.float(), dim=-1)
                    sw = scale * (zw @ Wn.T)
                    pw = F.softmax(sw / 0.5, dim=-1)
                    conf = pw.max(dim=-1).values
                    w_conf = conf / (conf.mean() + 1e-6)
                zs = F.normalize(_get_feat(model(xs)), dim=-1)
                Wn = F.normalize(head_W.float(), dim=-1)
                ss = scale * (zs @ Wn.T)
                log_qs = F.log_softmax(ss, dim=-1)
                qs = log_qs.exp()

                l_self = (w_conf * (-(pw * log_qs).sum(dim=-1))).mean()
                me = qs.mean(dim=0).clamp(min=1e-6)
                l_prior = (me * (torch.log(me) - torch.log(pi_t))).sum()
                l_kd = (1.0 - (zs * xf).sum(dim=-1)).mean()
                loss = l_self + prior_w * l_prior + kd_w * l_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    model.eval()
    eval_dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    z_list, p_list = [], []
    with torch.no_grad():
        Wn = F.normalize(head_W.float(), dim=-1)
        for xw, _, _ in eval_dl:
            xw = xw.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                zw = F.normalize(_get_feat(model(xw)), dim=-1)
            z_list.append(zw.cpu().numpy())
            p_list.append((scale * (zw @ Wn.T)).argmax(dim=-1).cpu().numpy())
    z_disc = np.concatenate(z_list, axis=0)
    lab_head = np.concatenate(p_list, axis=0).astype(int)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lab_head, z_disc


def _per_class_novel(L, y, lut_y, off, inc):
    pred = L.argmax(1)
    pc = {}
    new_mask = y >= off
    new_acc = 100.0 * float((pred[new_mask] == lut_y[new_mask]).mean()) if new_mask.any() else 0.0
    for k in range(inc):
        m = (y == off + k)
        pc[k] = 100.0 * float((pred[m] == lut_y[m]).mean()) if m.any() else 0.0
    return new_acc, pc


def _eval_replay_baselines(tag):
    st = replay.load(tag, check=False)
    a = st["args"]
    base, inc, off = a.base, a.increment, int(st["offset"])

    zt1 = np.asarray(st["z_test1"], np.float64)
    zt0 = np.asarray(st["z_test0"], np.float64)
    zt = np.concatenate([zt1, zt0])
    y1 = np.asarray(st["true_test1_y"]).astype(int)
    y = np.concatenate([y1, np.asarray(st["true_test0_y"]).astype(int)])
    lut_pipe = np.asarray(st["lut"])

    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    d = z0.shape[1]
    off0 = niw.choose(niw.loo_curve(z0, y0))
    sts = niw.class_stats(z0, y0)
    prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
    base_rows = [z0[y0 == k] for k in range(base)]
    base_slots = [score._post_1406(R, prior) for R in base_rows]
    idx = score._index_1203(base_rows)
    lb = score._logd_1202("A3", base_slots, zt)

    zp, X = np.asarray(st["z_pool"], np.float64), np.asarray(st["pool_x"], np.float32)
    ids, keep = np.asarray(st["ids"]).astype(int), np.asarray(st["keep"]).astype(bool)
    yp = np.asarray(st["true_pool_y"]).astype(int)
    pn = yp >= off
    nn_cnt = int(pn.sum())
    c09_pipe = 100.0 * score._hung_correct_1402(ids[pn], yp[pn] - off, inc) / max(nn_cnt, 1)

    n_rows = [zp[keep & (ids == c)] for c in range(inc)]
    L_n = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in n_rows], zt)], 1)
    na_n, pc_n = _per_class_novel(L_n, y, lut_pipe[y], off, inc)

    nl_rows = []
    for c in range(inc):
        rr = np.flatnonzero(keep & (ids == c))
        nl_rows.append(zp[rr[score._nl_keep_0910(zp[rr], base_rows, base_slots, prior, idx)]] if len(rr) else zp[:0])
    L_nl = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in nl_rows], zt)], 1)
    na_nl, pc_nl = _per_class_novel(L_nl, y, lut_pipe[y], off, inc)

    res = {
        "N": {**score._sc_1204(L_n, y, lut_pipe[y], off), "New": na_n, "per_class_novel": pc_n, "c09": c09_pipe},
        "NL": {**score._sc_1204(L_nl, y, lut_pipe[y], off), "New": na_nl, "per_class_novel": pc_nl, "c09": c09_pipe},
    }
    ctx = dict(st=st, a=a, base=base, inc=inc, off=off, zt=zt, zt1=zt1, y=y, y1=y1,
               prior=prior, base_rows=base_rows, base_slots=base_slots, idx=idx, lb=lb,
               zp=zp, X=X, ids=ids, keep=keep, yp=yp, pn=pn, nn_cnt=nn_cnt)
    return res, ctx


def eval_seed(tag, full_tag, base_backbone, feat2path, device, epochs=12):
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    out = {}
    if full_tag:
        full_res, _ = _eval_replay_baselines(full_tag)
        out["Full_N"] = full_res["N"]
        out["Full_NL"] = full_res["NL"]

    roi_res, c = _eval_replay_baselines(tag)
    out["ROI_N"] = roi_res["N"]
    out["ROI_NL"] = roi_res["NL"]

    a, inc, off = c["a"], c["inc"], c["off"]
    X, zp, ids, keep = c["X"], c["zp"], c["ids"], c["keep"]
    yp, pn, nn_cnt = c["yp"], c["pn"], c["nn_cnt"]
    base_rows, base_slots, prior, idx = c["base_rows"], c["base_slots"], c["prior"], c["idx"]
    lb, zt, zt1, y, y1 = c["lb"], c["zt"], c["zt1"], c["y"], c["y1"]

    ri = np.flatnonzero(keep)
    lab_n = ids[ri]
    init_c = np.zeros((inc, X.shape[1]), np.float32)
    counts = np.zeros(inc, np.float32)
    for k in range(inc):
        rc = ri[lab_n == k]
        counts[k] = len(rc)
        init_c[k] = X[rc].mean(axis=0) if len(rc) else X[ri].mean(axis=0)
    pi_emp = counts / max(counts.sum(), 1.0)
    pi_prior = 0.8 * pi_emp + 0.2 * (1.0 / inc)

    img_paths = [feat2path[X[r].tobytes()] for r in ri]
    lab_head, z_disc = train_imbal_discovery_branch(
        base_backbone, img_paths, X[ri], init_c, pi_prior, inc, device,
        epochs=epochs, seed=a.seed * 10 + 1
    )
    lab_km = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + 1).fit(z_disc).labels_.astype(int)

    xf_norm = X[ri] / (np.linalg.norm(X[ri], axis=1, keepdims=True) + 1e-8)
    z_blend = 0.5 * z_disc + 0.5 * xf_norm
    z_blend = z_blend / (np.linalg.norm(z_blend, axis=1, keepdims=True) + 1e-8)
    lab_blend = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + 1).fit(z_blend).labels_.astype(int)

    z_blend70 = 0.3 * z_disc + 0.7 * xf_norm
    z_blend70 = z_blend70 / (np.linalg.norm(z_blend70, axis=1, keepdims=True) + 1e-8)
    lab_blend70 = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + 1).fit(z_blend70).labels_.astype(int)

    for arm, lab, use_nl in (
        ("A10_Imb_head", lab_head, False),
        ("A10_Imb_km", lab_km, False),
        ("A10_Imb_blend50", lab_blend, False),
        ("A10_Imb_blend70", lab_blend70, False),
        ("A10_Imb_blend50_NL", lab_blend, True),
    ):
        new_slots = []
        for k in range(inc):
            rc = ri[lab == k]
            if len(rc):
                R_c = zp[rc[score._nl_keep_0910(zp[rc], base_rows, base_slots, prior, idx)]] if use_nl else zp[rc]
            else:
                R_c = zp[:0]
            new_slots.append(score._post_1406(R_c, prior))
        L1 = score._logd_1202("A3", new_slots, zt1)
        pred1 = L1.argmax(1)
        cm = np.zeros((inc, inc), dtype=int)
        for t_i, p_i in zip(y1 - off, pred1):
            if 0 <= t_i < inc and 0 <= p_i < inc:
                cm[t_i, p_i] += 1
        row_ind, col_ind = linear_sum_assignment(-cm)
        pi_map = {int(r): int(col) for r, col in zip(row_ind, col_ind)}
        lut = np.arange(off + inc)
        for k in range(inc):
            lut[off + k] = off + pi_map[k]
        L_all = np.concatenate([lb, score._logd_1202("A3", new_slots, zt)], 1)
        na, pc = _per_class_novel(L_all, y, lut[y], off, inc)

        full_lab = np.full(len(yp), -1, dtype=int)
        full_lab[ri] = lab
        m_kn = pn & (full_lab >= 0)
        c09_arm = 100.0 * score._hung_correct_1402(full_lab[m_kn], yp[m_kn] - off, inc) / max(nn_cnt, 1) if m_kn.any() else 0.0
        out[arm] = {**score._sc_1204(L_all, y, lut[y], off), "New": na, "per_class_novel": pc, "c09": c09_arm}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="colab_neu_d3_autoroi_ep10_r0")
    ap.add_argument("--full-prefix", default="colab_neu_dinov3_ep10_r0")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="splits/neu_split.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="")
    o = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"NEU-DET Step A10 (DINOv3 Auto-ROI + Imbalanced-DYDM) | device={device} | epochs={o.epochs} | seeds={o.seeds}", flush=True)
    t0 = time.time()

    split = json.load(open(o.split))
    novel_names = split["classes"][3:]
    base_backbone = load_merged_s0_backbone(o.ckpt, device)
    feat2path = build_feature_to_path(o.feat_dir, o.split)

    seeds = [int(x) for x in o.seeds.split(",") if x.strip()]
    per_seed = {}
    if o.out and Path(o.out).exists():
        try:
            old = json.loads(Path(o.out).read_text())
            per_seed = old.get("per_seed", {})
        except Exception:
            pass

    for s in seeds:
        sk = f"s{s}"
        if sk in per_seed:
            print(f"[seed {s}] đã có sẵn, bỏ qua.", flush=True)
            continue
        ts = time.time()
        tag = f"{o.prefix}_s{s}"
        ftag = f"{o.full_prefix}_s{s}" if o.full_prefix else ""
        res = eval_seed(tag, ftag, base_backbone, feat2path, device, epochs=o.epochs)
        per_seed[sk] = res
        print(
            f"[seed {s}] ({time.time()-ts:.1f}s) "
            f"ROI_N={res['ROI_N']['All']:.2f}% | "
            f"A10_head={res['A10_Imb_head']['All']:.2f}% (New={res['A10_Imb_head']['New']:.2f}%, bal={res['A10_Imb_head']['bal']:.4f}) | "
            f"A10_km={res['A10_Imb_km']['All']:.2f}% | "
            f"A10_blend70={res['A10_Imb_blend70']['All']:.2f}%",
            flush=True,
        )

    arms = list(next(iter(per_seed.values())).keys())
    summary = {"prefix": o.prefix, "epochs": o.epochs, "novel_classes": novel_names, "arms": {}, "per_seed": per_seed}

    print("\n=== BẢNG TỔNG HỢP DỄ HIỂU — BƯỚC A10 (NEU-DET DINOv3 Auto-ROI + Imbalanced-DYDM) ===")
    print("Ghi chú các cột:")
    print("  - Lớp cũ (Old %)        : Độ chính xác nhận diện 3 lớp lỗi đã học ở S0 (crazing, inclusion, patches).")
    print("  - Lớp mới (New %)       : % tổng số ảnh thuộc 3 lớp lỗi mới (pitted_surface, rolled-in_scale, scratches).")
    print("  - Cân bằng mới (bal)    : Trung bình độ chính xác từng lớp mới (0..1).")
    print("  - Gom cụm (c09 %)       : Độ thuần khi tự gom cụm 3 lớp mới chưa có nhãn.")
    print("  - Tổng thể (All %)      : Độ chính xác chung trên toàn bộ 6 lớp lỗi thép NEU-DET.\n")
    print(f"{'Phương pháp':<18s}  {'Lớp cũ(Old%)':>12s}  {'Lớp mới(New%)':>13s}  {'Cân bằng(bal)':>17s}  {'Gom cụm(c09%)':>13s}  {'Tổng(All%)':>10s}")
    print("-" * 92)
    for arm in arms:
        bals = [per_seed[s][arm]["bal"] for s in per_seed]
        olds = [per_seed[s][arm]["Old"] for s in per_seed]
        news = [per_seed[s][arm]["New"] for s in per_seed]
        alls = [per_seed[s][arm]["All"] for s in per_seed]
        c09s = [per_seed[s][arm]["c09"] for s in per_seed]
        pc_means = {}
        for k, nm in enumerate(novel_names):
            vals = [per_seed[s][arm]["per_class_novel"].get(str(k), per_seed[s][arm]["per_class_novel"].get(k, 0.0)) for s in per_seed]
            pc_means[nm] = float(np.mean(vals))
        summary["arms"][arm] = {
            "bal_mean": float(np.mean(bals)), "bal_std": float(np.std(bals)),
            "Old_mean": float(np.mean(olds)), "New_mean": float(np.mean(news)),
            "All_mean": float(np.mean(alls)), "All_std": float(np.std(alls)),
            "c09_mean": float(np.mean(c09s)),
            "per_class_novel": pc_means,
            "n": len(per_seed),
        }
        a_m = summary["arms"][arm]
        print(f"{arm:<18s}  {a_m['Old_mean']:11.2f}%  {a_m['New_mean']:12.2f}%  {a_m['bal_mean']:7.4f} ± {a_m['bal_std']:.4f}  {a_m['c09_mean']:12.2f}%  {a_m['All_mean']:9.2f}%")

    print("\n--- CHI TIẾT ĐỘ CHÍNH XÁC TỪNG LỚP LỖI MỚI NEU-DET (%) ---")
    hdr = f"{'Phương pháp':<18s}  " + "  ".join(f"{nm:>16s}" for nm in novel_names)
    print(hdr)
    print("-" * len(hdr))
    for arm in arms:
        pc = summary["arms"][arm]["per_class_novel"]
        print(f"{arm:<18s}  " + "  ".join(f"{pc[nm]:15.1f}%" for nm in novel_names))

    if o.out:
        Path(o.out).parent.mkdir(parents=True, exist_ok=True)
        Path(o.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if torch.cuda.is_available():
        print(f"peak_vram_mb = {torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)
    print(f"Tổng thời gian A10 DYDM: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
