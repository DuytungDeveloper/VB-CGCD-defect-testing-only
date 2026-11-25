#!/usr/bin/env python
"""Step A4 (duytung/08-gc10-dydm-imbalanced): Imbalanced-DYDM Discovery Branch + NIW Memory on GC10-DET.

Fixes the two domain-specific failure modes identified in Step A3 on GC10-DET:
  1. Replaces uniform centering + KL(me || Uniform) with Prior-Preserving KL(me || pi_prior) derived from
     FTR1's empirical cluster proportions + confidence-weighted self-training (no equipartition bias).
  2. Replaces RandomResizedCrop with Full-Frame Preserving augmentations (Resize + Flips + tiny affine +
     mild brightness/contrast) so small localized steel defects near edges are never cropped out.
  3. Evaluates FTR1, ImbDYDM_head, ImbDYDM_km, and ImbDYDM_blend (0.5 z_disc + 0.5 x_frozen) on s0..s4.
"""
import argparse, contextlib, copy, io, json, sys, time
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
import sessions
import score
from replay import s09_cluster


def load_merged_s0_backbone(ft_ckpt_path, device):
    """Load S0 BOFT fine-tuned DINOv1 from ft_ckpt.pt and merge adapters into a clean ViTModel."""
    from peft import BOFTConfig, get_peft_model
    from transformers import AutoModel

    base = AutoModel.from_pretrained("facebook/dino-vitb16")
    peft_config = BOFTConfig(
        boft_block_size=4,
        boft_n_butterfly_factor=2,
        target_modules=["output.dense", "mlp.fc1", "mlp.fc2"],
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


def build_feature_to_path(feat_dir, split_path="splits/gc10_split.json"):
    """Map exact 768d feature vector bytes -> GC10-DET train image path."""
    split = json.load(open(split_path))
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        names = split["train"][c]
        paths += [f"{split['root']}/{n}" for n in names]
        labels += [k] * len(names)
    feats = np.load(f"{feat_dir}/features-dino-vitb16-sl.npy")
    labs = np.load(f"{feat_dir}/labels-dino-vitb16-sl.npy")
    assert len(paths) == len(feats) and np.array_equal(np.asarray(labels), labs), \
        "GC10-DET feature-to-path mapping mismatch!"
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
    """Full-frame preserving augmentations so tiny steel surface defects are never cropped out."""
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
    """Imbalanced-DYDM Discovery Branch: prior-preserving KL + confidence-weighted self-training + stronger KD."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = copy.deepcopy(base_backbone).to(device)
    for n, p in model.named_parameters():
        p.requires_grad = (".layer.11." in n or ".layers.11." in n or "layernorm" in n or "pooler" in n)
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
                    zw = F.normalize(model(xw).pooler_output.float(), dim=-1)
                    Wn = F.normalize(head_W.float(), dim=-1)
                    sw = scale * (zw @ Wn.T)
                    # Sharpened pseudo-labels WITHOUT uniform centering
                    pw = F.softmax(sw / 0.5, dim=-1)
                    conf = pw.max(dim=-1).values
                    w_conf = conf / (conf.mean() + 1e-6)
                zs = F.normalize(model(xs).pooler_output.float(), dim=-1)
                Wn = F.normalize(head_W.float(), dim=-1)
                ss = scale * (zs @ Wn.T)
                log_qs = F.log_softmax(ss, dim=-1)
                qs = log_qs.exp()

                # Confidence-weighted self-training loss
                l_self = (w_conf * (-(pw * log_qs).sum(dim=-1))).mean()
                # Prior-preserving KL(me || pi_prior) instead of uniform equipartition
                me = qs.mean(dim=0).clamp(min=1e-6)
                l_prior = (me * (torch.log(me) - torch.log(pi_t))).sum()
                # Stronger KD anchor to frozen S0 manifold
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
                zw = F.normalize(model(xw).pooler_output.float(), dim=-1)
            z_list.append(zw.cpu().numpy())
            p_list.append((scale * (zw @ Wn.T)).argmax(dim=-1).cpu().numpy())
    z_disc = np.concatenate(z_list, axis=0)
    lab_head = np.concatenate(p_list, axis=0).astype(int)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return lab_head, z_disc


def eval_seed(tag, base_backbone, feat2path, device, epochs=12):
    """Run NL, FTR1, ImbDYDM_head, ImbDYDM_km, ImbDYDM_blend on GC10-DET t1 for one seed."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    blobs = sessions.snapshots_sessions(tag)
    rng = blobs["S1_09"]["rng_in"]
    del blobs
    st, views = sessions.load(tag, check=False)
    v = views[0]
    a = st["args"]
    base, inc, off = a.base, a.increment, int(v["offset"])

    zt = np.concatenate([np.asarray(v["z_test1"], np.float64), np.asarray(st["z_test0"], np.float64)])
    zt1 = np.asarray(v["z_test1"], np.float64)
    y1 = np.asarray(v["true_test1_y"]).astype(int)
    y = np.concatenate([y1, np.asarray(st["true_test0_y"]).astype(int)])
    lut_pipe = np.asarray(v["lut"])

    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    d = z0.shape[1]
    off0 = niw.choose(niw.loo_curve(z0, y0))
    sts = niw.class_stats(z0, y0)
    prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
    base_rows = [z0[y0 == k] for k in range(base)]
    base_slots = [score._post_1406(R, prior) for R in base_rows]
    idx = score._index_1203(base_rows)
    lb = score._logd_1202("A3", base_slots, zt)

    zp, X = np.asarray(v["z_pool"], np.float64), np.asarray(v["pool_x"], np.float32)
    ids, keep = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)
    yp = np.asarray(v["true_pool_y"]).astype(int)
    pn = yp >= off
    nn_cnt = int(pn.sum())

    # NL
    nl_rows = []
    for c in range(inc):
        rr = np.flatnonzero(keep & (ids == c))
        nl_rows.append(zp[rr[score._nl_keep_0910(zp[rr], base_rows, base_slots, prior, idx)]] if len(rr) else zp[:0])
    L_nl = np.concatenate([lb, score._logd_1202("A3", [score._post_1406(R, prior) for R in nl_rows], zt)], 1)
    out = {"NL": {**score._sc_1204(L_nl, y, lut_pipe[y], off),
                  "c09": 100.0 * score._hung_correct_1402(ids[pn], yp[pn] - off, inc) / max(nn_cnt, 1)}}

    # Filtered novel pool G
    G = np.zeros(len(zp), bool)
    for c in range(inc):
        rr = np.flatnonzero(keep & (ids == c))
        if len(rr):
            G[rr[score._nl_keep_0910(zp[rr], base_rows, base_slots, prior, idx)]] = True
    ri = np.flatnonzero(G)

    # FTR1 re-clustering
    np.random.set_state(rng)
    with contextlib.redirect_stdout(io.StringIO()):
        cl_f = s09_cluster(dict(args=a, offset=off, pool_x=X[ri]))
    lab_f = np.asarray(cl_f["ids"]).astype(int)

    init_c = np.zeros((inc, X.shape[1]), np.float32)
    counts = np.zeros(inc, np.float32)
    for c in range(inc):
        rc = ri[lab_f == c]
        counts[c] = len(rc)
        init_c[c] = X[rc].mean(axis=0) if len(rc) else X[ri].mean(axis=0)
    pi_emp = counts / max(counts.sum(), 1.0)
    pi_prior = 0.8 * pi_emp + 0.2 * (1.0 / inc)

    img_paths = [feat2path[X[r].tobytes()] for r in ri]
    lab_head, z_disc = train_imbal_discovery_branch(
        base_backbone, img_paths, X[ri], init_c, pi_prior, inc, device,
        epochs=epochs, seed=a.seed * 10 + 1
    )
    lab_km = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + 1).fit(z_disc).labels_.astype(int)

    # Residual feature blend (50% adapted + 50% frozen S0 normalized)
    xf_norm = X[ri] / (np.linalg.norm(X[ri], axis=1, keepdims=True) + 1e-8)
    z_blend = 0.5 * z_disc + 0.5 * xf_norm
    z_blend = z_blend / (np.linalg.norm(z_blend, axis=1, keepdims=True) + 1e-8)
    lab_blend = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + 1).fit(z_blend).labels_.astype(int)

    for arm, lab in (("FTR1", lab_f), ("ImbDYDM_head", lab_head), ("ImbDYDM_km", lab_km), ("ImbDYDM_blend", lab_blend)):
        new_slots = []
        for c in range(inc):
            rc = ri[lab == c]
            R_c = zp[rc[score._nl_keep_0910(zp[rc], base_rows, base_slots, prior, idx)]] if len(rc) else zp[:0]
            new_slots.append(score._post_1406(R_c, prior))
        L1 = score._logd_1202("A3", new_slots, zt1)
        pred1 = L1.argmax(1)
        cm = np.zeros((inc, inc), dtype=int)
        for p_c, t_c in zip(pred1, y1 - off):
            if 0 <= t_c < inc:
                cm[p_c, t_c] += 1
        r_ind, c_ind = linear_sum_assignment(-cm)
        lut = np.arange(a.num_classes)
        for r_c, t_c in zip(r_ind, c_ind):
            lut[off + t_c] = off + r_c

        L = np.concatenate([lb, score._logd_1202("A3", new_slots, zt)], 1)
        full_ids = np.full(len(zp), -1, dtype=int)
        full_ids[ri] = lab
        c09 = 100.0 * score._hung_correct_1402(full_ids[pn], yp[pn] - off, inc) / max(nn_cnt, 1)
        out[arm] = {**score._sc_1204(L, y, lut[y], off), "c09": c09}

    print(
        f"  [{tag}] FTR1 bal={out['FTR1']['bal']:.4f} All={out['FTR1']['All']:.2f} | "
        f"Imb_head bal={out['ImbDYDM_head']['bal']:.4f} All={out['ImbDYDM_head']['All']:.2f} | "
        f"Imb_km bal={out['ImbDYDM_km']['bal']:.4f} All={out['ImbDYDM_km']['All']:.2f} | "
        f"Imb_blend bal={out['ImbDYDM_blend']['bal']:.4f} All={out['ImbDYDM_blend']['All']:.2f}",
        flush=True,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="colab_gc10_ep20_r0")
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--ft-ckpt", required=True)
    ap.add_argument("--split", default="splits/gc10_split.json")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--meta-json", default=None)
    o = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GC10 Imbalanced-DYDM Step A4 | device={device} | epochs={o.epochs} | seeds={o.seeds}", flush=True)
    base_backbone = load_merged_s0_backbone(o.ft_ckpt, device)
    feat2path = build_feature_to_path(o.feat_dir, o.split)

    out_p = Path(o.out_json)
    all_res = json.loads(out_p.read_text()).get("per_seed", {}) if out_p.exists() else {}
    seeds = [int(s) for s in o.seeds.split(",") if s]

    for s in seeds:
        tag = f"{o.prefix}_s{s}"
        t0 = time.time()
        all_res[f"s{s}"] = eval_seed(tag, base_backbone, feat2path, device, epochs=o.epochs)
        print(f"Done {tag} in {time.time() - t0:.1f}s", flush=True)

    arms = ("NL", "FTR1", "ImbDYDM_head", "ImbDYDM_km", "ImbDYDM_blend")
    agg = {}
    for arm in arms:
        vals = [all_res[k][arm] for k in sorted(all_res) if arm in all_res[k]]
        if not vals:
            continue
        agg[arm] = {
            "bal_mean": float(np.mean([v["bal"] for v in vals])),
            "bal_std": float(np.std([v["bal"] for v in vals])),
            "Old_mean": float(np.mean([v["Old"] for v in vals])),
            "All_mean": float(np.mean([v["All"] for v in vals])),
            "c09_mean": float(np.mean([v["c09"] for v in vals])),
            "n": len(vals),
        }

    payload = {"prefix": o.prefix, "epochs": o.epochs, "arms": agg, "per_seed": all_res}
    if o.meta_json and Path(o.meta_json).exists():
        payload["telemetry"] = json.loads(Path(o.meta_json).read_text())
    out_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n=== KẾT QUẢ BƯỚC A4 (GC10-DET Imbalanced-DYDM + NIW Memory, ep=20, r0) ===")
    print(f"{'Arm':<14} {'bal (mean±std)':>18} {'Old (%)':>10} {'All (%)':>10} {'c09 (%)':>10} {'dbal vs FTR1':>14}")
    print("-" * 82)
    f_bal = agg.get("FTR1", {}).get("bal_mean", 0.0)
    for arm in arms:
        if arm not in agg:
            continue
        r = agg[arm]
        print(
            f"{arm:<14} {r['bal_mean']:7.4f} ± {r['bal_std']:6.4f} {r['Old_mean']:10.2f} "
            f"{r['All_mean']:10.2f} {r['c09_mean']:10.2f} {r['bal_mean'] - f_bal:+14.4f}"
        )


if __name__ == "__main__":
    main()
