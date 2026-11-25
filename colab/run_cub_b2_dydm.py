#!/usr/bin/env python
"""Step B2 (duytung/04-cub-tach-nhanh-kham-pha): Decoupled Discovery Branch (DYDM) + NIW Memory on CUB-200.

Reuses the existing S0-finetuned DINOv1 checkpoint (ft_ckpt.pt) and colab_cub_ft_l25_r0_s0..s4 snapshots
so Memory Branch (NIW + Student-t on frozen S0 PCA-384) is 100% identical to FTR1, while Discovery Branch
adapts only the last ViT block + cosine head per session with L_self-train + L_GER + L_KD.
"""
import argparse, contextlib, copy, io, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "lab"))
sys.path.append(str(ROOT / "feature_extractor"))

import niw
import sessions
import score
from replay import s09_cluster
from sl_finetuned_model import load_model


def load_merged_s0_backbone(ft_ckpt_path, device):
    """Load S0 BOFT fine-tuned DINOv1 from ft_ckpt.pt and merge adapters into a clean ViTModel."""
    lora_model = load_model(200, "dino_vitb16")
    raw = torch.load(ft_ckpt_path, map_location="cpu")
    state = raw["state"] if isinstance(raw, dict) and "state" in raw else raw
    lora_model.load_state_dict(state, strict=False)
    backbone = lora_model.backbone
    if hasattr(backbone, "merge_and_unload"):
        backbone = backbone.merge_and_unload()
    return backbone.to(device).eval()


def build_feature_to_hf_index(feat_dir, hf_train):
    """Map exact 768d feature vector bytes -> HuggingFace train dataset index (verified against labels)."""
    feats = np.load(f"{feat_dir}/features-dino-vitb16-sl.npy")
    labs = np.load(f"{feat_dir}/labels-dino-vitb16-sl.npy")
    n = len(hf_train)
    bs = 128
    n_batches = (n + bs - 1) // bs
    order = sorted(range(n_batches), key=lambda b: f"features_{b}.npy")
    hf_idx = np.concatenate([np.arange(b * bs, min((b + 1) * bs, n)) for b in order])
    texts = np.array(hf_train["text"])
    t2i = {t: i for i, t in enumerate(np.unique(texts))}
    hf_labs = np.array([t2i[t] for t in texts])
    assert len(hf_idx) == len(feats) and np.array_equal(hf_labs[hf_idx], labs), \
        "CUB-200 feature-to-HF index mapping mismatch!"
    return {feats[i].astype(np.float32).tobytes(): int(hf_idx[i]) for i in range(len(feats))}


class _UnlabeledPairDataset(torch.utils.data.Dataset):
    def __init__(self, pil_imgs, frozen_feats, weak_tfm, strong_tfm):
        self.imgs = pil_imgs
        self.ff = frozen_feats
        self.weak = weak_tfm
        self.strong = strong_tfm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        im = self.imgs[i].convert("RGB")
        return self.weak(im), self.strong(im), self.ff[i]


def _make_transforms():
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    weak = T.Compose([
        T.Resize(int(224 / 0.875), interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    strong = T.Compose([
        T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.6),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return weak, strong


def train_discovery_branch(base_backbone, pil_imgs, frozen_x, init_centroids, inc, device,
                           epochs=12, batch_size=64, lr=1e-4, ger_w=2.0, kd_w=1.0, seed=0):
    """Decoupled DYDM Discovery Branch: adapt last ViT block + cosine head on filtered novel pool G."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = copy.deepcopy(base_backbone).to(device)
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze ONLY last Transformer block + layernorm + pooler
    for p in model.encoder.layer[-1].parameters():
        p.requires_grad = True
    for p in model.layernorm.parameters():
        p.requires_grad = True
    if hasattr(model, "pooler") and model.pooler is not None:
        for p in model.pooler.parameters():
            p.requires_grad = True

    c0 = torch.tensor(init_centroids, dtype=torch.float32, device=device)
    c0 = F.normalize(c0, dim=-1)
    head_W = nn.Parameter(c0.clone())

    weak_tfm, strong_tfm = _make_transforms()
    ff_norm = F.normalize(torch.tensor(frozen_x, dtype=torch.float32), dim=-1)
    ds = _UnlabeledPairDataset(pil_imgs, ff_norm, weak_tfm, strong_tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad] + [head_W]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    scale = 10.0
    center = torch.zeros(1, inc, device=device)

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
                    pw = F.softmax((sw - center) / 0.4, dim=-1)
                    center = 0.9 * center + 0.1 * sw.mean(dim=0, keepdim=True)
                zs = F.normalize(model(xs).pooler_output.float(), dim=-1)
                Wn = F.normalize(head_W.float(), dim=-1)
                ss = scale * (zs @ Wn.T)
                log_qs = F.log_softmax(ss, dim=-1)
                qs = log_qs.exp()

                l_self = -(pw * log_qs).sum(dim=-1).mean()
                me = qs.mean(dim=0)
                l_ger = (me * torch.log(me + 1e-8)).sum() + float(np.log(inc))
                l_kd = (1.0 - (zs * xf).sum(dim=-1)).mean()
                loss = l_self + ger_w * l_ger + kd_w * l_kd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        sched.step()

    # Extract Discovery Branch predictions & adapted features on weak view
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


def eval_seed(tag, base_backbone, feat2hf, hf_train, device, epochs=12):
    """Run FTR1 vs DYDM_head vs DYDM_km across all 5 sessions of one CUB-200 seed."""
    from sklearn.cluster import KMeans

    blobs = sessions.snapshots_sessions(tag)
    rngs = [blobs[f"S{i}_09"]["rng_in"] for i in range(1, 6)]
    del blobs
    st, views = sessions.load(tag, check=False)
    a = st["args"]
    base, inc = a.base, a.increment

    z0, y0 = np.asarray(st["z_s0"], np.float64), np.asarray(st["s0_y"]).astype(int)
    d = z0.shape[1]
    off0 = niw.choose(niw.loo_curve(z0, y0))
    sts = niw.class_stats(z0, y0)
    prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
    base_rows = [z0[y0 == k] for k in range(base)]
    base_slots = [score._post_1406(R, prior) for R in base_rows]

    arms = ("FTR1", "DYDM_head", "DYDM_km")
    arm_rows = {k: list(base_rows) for k in arms}
    arm_slots = {k: list(base_slots) for k in arms}
    arm_luts = {k: np.arange(a.num_classes) for k in arms}
    res = {k: {"New": [], "Old": [], "All": [], "c09": []} for k in arms}

    for i, v in enumerate(views, start=1):
        off = int(v["offset"])
        zp, X = np.asarray(v["z_pool"], np.float64), np.asarray(v["pool_x"], np.float32)
        ids, keep = np.asarray(v["ids"]).astype(int), np.asarray(v["keep"]).astype(bool)
        yp = np.asarray(v["true_pool_y"]).astype(int)
        pn = yp >= off
        nn_cnt = int(pn.sum())

        # Test set of session i (all seen classes so far)
        zt = np.asarray(st["pca"].transform(v["test_all_x"]), np.float64)
        yt = np.asarray(v["true_test_all_y"]).astype(int)
        y1 = np.asarray(v["true_test1_y"]).astype(int)
        zt1 = np.asarray(v["z_test1"], np.float64)

        # 1) FTR1 filter G on old_slots of FTR1
        idx_f = score._index_1203(arm_rows["FTR1"])
        G = np.zeros(len(zp), bool)
        for c in range(inc):
            rr = np.flatnonzero(keep & (ids == c))
            if len(rr):
                G[rr[score._nl_keep_0910(zp[rr], arm_rows["FTR1"], arm_slots["FTR1"], prior, idx_f)]] = True
        ri = np.flatnonzero(G)

        # FTR1 re-clustering (Euclidean k-Means on raw X[ri])
        np.random.set_state(rngs[i - 1])
        with contextlib.redirect_stdout(io.StringIO()):
            cl_f = s09_cluster(dict(args=a, offset=off, pool_x=X[ri]))
        lab_f = np.asarray(cl_f["ids"]).astype(int)

        # Centroids from FTR1 k-Means to initialize Discovery Branch head
        init_c = np.zeros((inc, X.shape[1]), np.float32)
        for c in range(inc):
            rc = ri[lab_f == c]
            init_c[c] = X[rc].mean(axis=0) if len(rc) else X[ri].mean(axis=0)

        # Map ri -> raw PIL images from HuggingFace train split
        pil_imgs = [hf_train[feat2hf[X[r].tobytes()]]["image"] for r in ri]
        lab_head, z_disc = train_discovery_branch(
            base_backbone, pil_imgs, X[ri], init_c, inc, device,
            epochs=epochs, seed=a.seed * 10 + i
        )
        km_d = KMeans(n_clusters=inc, n_init=10, random_state=a.seed * 10 + i).fit(z_disc)
        lab_km = km_d.labels_.astype(int)

        labs_by_arm = {"FTR1": lab_f, "DYDM_head": lab_head, "DYDM_km": lab_km}
        for arm in arms:
            lab = labs_by_arm[arm]
            idx_a = score._index_1203(arm_rows[arm])
            new_rows, new_slots = [], []
            for c in range(inc):
                rc = ri[lab == c]
                if len(rc):
                    kp = score._nl_keep_0910(zp[rc], arm_rows[arm], arm_slots[arm], prior, idx_a)
                    R_c = zp[rc[kp]]
                else:
                    R_c = zp[:0]
                new_rows.append(R_c)
                new_slots.append(score._post_1406(R_c, prior))

            # Hungarian naming of the 20 new slots on test1 using the new NIW slots
            L1 = score._logd_1202("A3", new_slots, zt1)
            pred1 = L1.argmax(1)
            from scipy.optimize import linear_sum_assignment
            cm = np.zeros((inc, inc), dtype=int)
            for p_c, t_c in zip(pred1, y1 - off):
                if 0 <= t_c < inc:
                    cm[p_c, t_c] += 1
            r_ind, c_ind = linear_sum_assignment(-cm)
            lut = arm_luts[arm].copy()
            for r_c, t_c in zip(r_ind, c_ind):
                lut[off + t_c] = off + r_c
            arm_luts[arm] = lut

            arm_rows[arm].extend(new_rows)
            arm_slots[arm].extend(new_slots)

            L_all = score._logd_1202("A3", arm_slots[arm], zt)
            ok = L_all.argmax(1) == lut[yt]
            is_new = yt >= off
            res[arm]["New"].append(100.0 * float(ok[is_new].mean()))
            res[arm]["Old"].append(100.0 * float(ok[~is_new].mean()))
            res[arm]["All"].append(100.0 * float(ok.mean()))
            c09 = 100.0 * score._hung_correct_1402(lab[pn[ri]], yp[ri][pn[ri]] - off, inc) / max(nn_cnt, 1)
            res[arm]["c09"].append(float(c09))
        print(f"  [{tag} S{i}] FTR1 New={res['FTR1']['New'][-1]:.2f} c09={res['FTR1']['c09'][-1]:.2f} | "
              f"DYDM_head New={res['DYDM_head']['New'][-1]:.2f} c09={res['DYDM_head']['c09'][-1]:.2f} | "
              f"DYDM_km New={res['DYDM_km']['New'][-1]:.2f} c09={res['DYDM_km']['c09'][-1]:.2f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="colab_cub_ft_l25")
    ap.add_argument("--feat-dir", required=True, help="features/<run>/dino-vitb16-sl")
    ap.add_argument("--ckpt", required=True, help="features/<run>/ft_ckpt.pt")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out-json", required=True)
    o = ap.parse_args()

    from datasets import load_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DYDM Step B2 | device={device} | epochs={o.epochs} | seeds={o.seeds}", flush=True)

    hf_train = load_dataset("cassiekang/cub200_dataset", split="train")
    feat2hf = build_feature_to_hf_index(o.feat_dir, hf_train)
    base_backbone = load_merged_s0_backbone(o.ckpt, device)

    seeds = [int(x) for x in o.seeds.split(",") if x]
    all_res = {}
    for s in seeds:
        tag = f"{o.prefix}_r0_s{s}"
        t0 = time.time()
        all_res[f"s{s}"] = eval_seed(tag, base_backbone, feat2hf, hf_train, device, epochs=o.epochs)
        print(f"Done {tag} in {time.time() - t0:.1f}s", flush=True)

    arms = ("FTR1", "DYDM_head", "DYDM_km")
    summary = {"prefix": o.prefix, "epochs": o.epochs, "seeds": seeds, "arms": {}, "per_seed": all_res}
    print("\n=== KẾT QUẢ BƯỚC B2 (CUB-200 DYDM Discovery Branch + NIW Memory, r0, s0..s4) ===")
    print(f"{'Arm':<12} {'New (M_d)':>10} {'Old (%)':>10} {'S5_All (%)':>11} {'c09 (%)':>10} {'dM_d vs FTR1':>14}")
    print("-" * 72)
    ftr_md = float(np.mean([np.mean(all_res[f"s{s}"]["FTR1"]["New"]) for s in seeds]))
    for arm in arms:
        md = float(np.mean([np.mean(all_res[f"s{s}"][arm]["New"]) for s in seeds]))
        old = float(np.mean([np.mean(all_res[f"s{s}"][arm]["Old"]) for s in seeds]))
        s5 = float(np.mean([all_res[f"s{s}"][arm]["All"][-1] for s in seeds]))
        c09 = float(np.mean([np.mean(all_res[f"s{s}"][arm]["c09"]) for s in seeds]))
        dmd = md - ftr_md
        summary["arms"][arm] = {"M_d": md, "Old": old, "S5_All": s5, "c09": c09, "dM_d": dmd}
        print(f"{arm:<12} {md:10.3f} {old:10.3f} {s5:11.3f} {c09:10.3f} {dmd:+14.3f}")

    Path(o.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
