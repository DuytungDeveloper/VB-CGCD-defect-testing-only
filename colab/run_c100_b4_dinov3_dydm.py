#!/usr/bin/env python
"""Step B4 (duytung/15-cifar100-dinov3-dydm): DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m)
+ Pure Closed-Form NIW Memory + FTR1 + Decoupled DYDM Discovery Branch on CIFAR-100
(60,000 images, 100 classes: 50 base + 5 x 10 novel sessions S1..S5).

Runs pure closed-form NIW + FTR1 + DYDM directly from main.Dataloader(t5) without the slow
15-hour JAX SVI bottleneck, with exact incremental per-seed auto-resume JSON.
"""
import argparse, copy, json, sys, time, types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "lab"))

import main as vbcgcd_main
import niw
import score
from replay import DEFAULTS

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
    print(f"B4 load_merged_s0_backbone({HF_DINOV3}): BOFT targets = {targets}", flush=True)

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


def build_feature_to_hf_index(feat_dir, hf_train):
    feats = np.load(f"{feat_dir}/features-dinov3-vitb16-sl.npy")
    labs = np.load(f"{feat_dir}/labels-dinov3-vitb16-sl.npy")
    n = len(hf_train)
    bs = 128
    n_batches = (n + bs - 1) // bs
    order = sorted(range(n_batches), key=lambda b: f"features_{b}.npy")
    hf_idx = np.concatenate([np.arange(b * bs, min((b + 1) * bs, n)) for b in order])
    hf_labs = np.asarray(hf_train["fine_label"], dtype=np.int64)
    assert len(hf_idx) == len(feats) and np.array_equal(hf_labs[hf_idx], labs), \
        "CIFAR-100 DINOv3 feature-to-HF index mapping mismatch!"
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
                           epochs=6, batch_size=128, lr=1e-4, ger_w=2.0, kd_w=1.0, seed=0):
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

    weak_tfm, strong_tfm = _make_transforms()
    ff_norm = F.normalize(torch.tensor(frozen_x, dtype=torch.float32), dim=-1)
    ds = _UnlabeledPairDataset(pil_imgs, ff_norm, weak_tfm, strong_tfm)
    dl = torch.utils.data.DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, drop_last=False, num_workers=2)

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
                    zw = F.normalize(_get_feat(model(xw)), dim=-1)
                    Wn = F.normalize(head_W.float(), dim=-1)
                    sw = scale * (zw @ Wn.T)
                    pw = F.softmax((sw - center) / 0.4, dim=-1)
                    center = 0.9 * center + 0.1 * sw.mean(dim=0, keepdim=True)
                zs = F.normalize(_get_feat(model(xs)), dim=-1)
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

    model.eval()
    eval_dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
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


def eval_seed(seed, feat_dir, base_backbone, feat2hf, hf_train, device, epochs=6):
    f = Path(feat_dir)
    args = types.SimpleNamespace(**{
        **DEFAULTS,
        "dataset": "cifar100", "base": 50, "increment": 10, "num_classes": 100, "num_dim": 384,
        "load_mode": "t5", "data_dir": str(f.parent), "pretrained_model_name": f.name, "seed": seed,
    })
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, te, _, te_all = (list(it) for it in vbcgcd_main.Dataloader(args))

    s0_x, s0_y = np.asarray(tr[0]._x, np.float32), np.asarray(tr[0]._y, dtype=int)
    t0_x, t0_y = np.asarray(te[0]._x, np.float32), np.asarray(te[0]._y, dtype=int)

    pca = PCA(n_components=384, random_state=seed)
    z0 = np.asarray(pca.fit_transform(s0_x), np.float64)
    zt0 = np.asarray(pca.transform(t0_x), np.float64)

    base, inc = 50, 10
    d = z0.shape[1]
    off0 = niw.choose(niw.loo_curve(z0, s0_y))
    sts = niw.class_stats(z0, s0_y)
    prior = (sts["m0"], niw.K0, d + 1 + off0, off0 * niw.prior_cov(sts, 0))
    base_rows = [z0[s0_y == k] for k in range(base)]
    base_slots = [score._post_1406(R, prior) for R in base_rows]
    s0_acc = 100.0 * float((score._logd_1202("A3", base_slots, zt0).argmax(1) == t0_y).mean())
    print(f"  [c100 s{seed} S0] NIW S0_All = {s0_acc:.2f}% (off0={off0})", flush=True)

    arms = ("FTR1", "DYDM_head", "DYDM_km", "DYDM_blend50")
    arm_rows = {k: list(base_rows) for k in arms}
    arm_slots = {k: list(base_slots) for k in arms}
    arm_luts = {k: np.arange(100) for k in arms}
    res = {k: {"S0": s0_acc, "New": [], "Old": [], "All": [], "S0_keep": [], "c09": []} for k in arms}

    for i in range(1, 6):
        off = base + (i - 1) * inc
        X = np.asarray(tr[i]._x, np.float32)
        yp = np.asarray(tr[i]._y, dtype=int)
        zp = np.asarray(pca.transform(X), np.float64)
        pn = yp >= off
        nn_cnt = int(pn.sum())

        zt = np.asarray(pca.transform(np.asarray(te_all[i]._x, np.float32)), np.float64)
        yt = np.asarray(te_all[i]._y, dtype=int)
        zt1 = np.asarray(pca.transform(np.asarray(te[i]._x, np.float32)), np.float64)
        y1 = np.asarray(te[i]._y, dtype=int)

        # 1) Initial k-Means on pool X, then NIW leave-one-out old/new filter -> clean novel pool G (FTR1)
        ids0 = KMeans(n_clusters=inc, n_init=10, random_state=seed * 10 + i).fit(X).labels_.astype(int)
        idx_f = score._index_1203(arm_rows["FTR1"])
        G = np.zeros(len(zp), bool)
        for c in range(inc):
            rr = np.flatnonzero(ids0 == c)
            if len(rr):
                G[rr[score._nl_keep_0910(zp[rr], arm_rows["FTR1"], arm_slots["FTR1"], prior, idx_f)]] = True
        ri = np.flatnonzero(G)

        # FTR1 re-clustering on filtered novel pool X[ri]
        lab_f = KMeans(n_clusters=inc, n_init=10, random_state=seed * 100 + i).fit(X[ri]).labels_.astype(int)

        init_c = np.zeros((inc, X.shape[1]), np.float32)
        for c in range(inc):
            rc = ri[lab_f == c]
            init_c[c] = X[rc].mean(axis=0) if len(rc) else X[ri].mean(axis=0)

        pil_imgs = [hf_train[feat2hf[X[r].tobytes()]]["img"] for r in ri]
        lab_head, z_disc = train_discovery_branch(
            base_backbone, pil_imgs, X[ri], init_c, inc, device,
            epochs=epochs, seed=seed * 10 + i
        )
        lab_km = KMeans(n_clusters=inc, n_init=10, random_state=seed * 10 + i).fit(z_disc).labels_.astype(int)

        xf_norm = X[ri] / (np.linalg.norm(X[ri], axis=1, keepdims=True) + 1e-8)
        z_blend = 0.5 * z_disc + 0.5 * xf_norm
        z_blend = z_blend / (np.linalg.norm(z_blend, axis=1, keepdims=True) + 1e-8)
        lab_blend = KMeans(n_clusters=inc, n_init=10, random_state=seed * 10 + i).fit(z_blend).labels_.astype(int)

        labs_by_arm = {"FTR1": lab_f, "DYDM_head": lab_head, "DYDM_km": lab_km, "DYDM_blend50": lab_blend}
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

            L1 = score._logd_1202("A3", new_slots, zt1)
            pred1 = L1.argmax(1)
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
            is_s0 = yt < base
            res[arm]["New"].append(100.0 * float(ok[is_new].mean()))
            res[arm]["Old"].append(100.0 * float(ok[~is_new].mean()))
            res[arm]["All"].append(100.0 * float(ok.mean()))
            res[arm]["S0_keep"].append(100.0 * float(ok[is_s0].mean()))
            c09 = 100.0 * score._hung_correct_1402(lab[pn[ri]], yp[ri][pn[ri]] - off, inc) / max(nn_cnt, 1)
            res[arm]["c09"].append(float(c09))
        print(
            f"  [c100 s{seed} S{i}] FTR1 All={res['FTR1']['All'][-1]:.2f} New={res['FTR1']['New'][-1]:.2f} | "
            f"DYDM_km All={res['DYDM_km']['All'][-1]:.2f} New={res['DYDM_km']['New'][-1]:.2f} | "
            f"blend50 All={res['DYDM_blend50']['All'][-1]:.2f} New={res['DYDM_blend50']['New'][-1]:.2f}",
            flush=True,
        )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out-json", required=True)
    o = ap.parse_args()

    from datasets import load_dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CIFAR-100 Step B4 (DINOv3 + NIW + FTR1 + DYDM) | device={device} | epochs={o.epochs} | seeds={o.seeds}", flush=True)

    hf_train = load_dataset("uoft-cs/cifar100", split="train")
    feat2hf = build_feature_to_hf_index(o.feat_dir, hf_train)
    base_backbone = load_merged_s0_backbone(o.ckpt, device)

    seeds = [int(x) for x in o.seeds.split(",") if x.strip()]
    per_seed = {}
    if Path(o.out_json).exists():
        try:
            per_seed = json.loads(Path(o.out_json).read_text()).get("per_seed", {})
        except Exception:
            pass

    for s in seeds:
        sk = f"s{s}"
        if sk in per_seed:
            print(f"[seed {s}] đã có sẵn, bỏ qua.", flush=True)
            continue
        t0 = time.time()
        per_seed[sk] = eval_seed(s, o.feat_dir, base_backbone, feat2hf, hf_train, device, epochs=o.epochs)
        print(f"Done CIFAR-100 seed s{s} in {time.time() - t0:.1f}s", flush=True)

    arms = ("FTR1", "DYDM_head", "DYDM_km", "DYDM_blend50")
    done_seeds = sorted(per_seed.keys())
    summary = {"epochs": o.epochs, "seeds": done_seeds, "arms": {}, "per_seed": per_seed}

    ftr_md = float(np.mean([np.mean(per_seed[sk]["FTR1"]["New"]) for sk in done_seeds]))
    print("\n=== BẢNG TỔNG HỢP DỄ HIỂU — BƯỚC B4 (CIFAR-100 DINOv3 + DYDM + NIW, 60.000 ảnh, 5 phiên S1..S5) ===")
    print(f"{'Phương pháp':<14} {'S0 (%)':>8} {'Lớp mới(M_d)':>13} {'Lớp cũ(Old%)':>13} {'Quên(M_f)':>10} {'Gom cụm(c09%)':>14} {'S5_All%':>9} {'vs FTR1':>9}")
    print("-" * 98)
    for arm in arms:
        s0 = float(np.mean([per_seed[sk][arm]["S0"] for sk in done_seeds]))
        md = float(np.mean([np.mean(per_seed[sk][arm]["New"]) for sk in done_seeds]))
        old = float(np.mean([np.mean(per_seed[sk][arm]["Old"]) for sk in done_seeds]))
        mf = float(np.mean([per_seed[sk][arm]["S0"] - per_seed[sk][arm]["S0_keep"][-1] for sk in done_seeds]))
        s5 = float(np.mean([per_seed[sk][arm]["All"][-1] for sk in done_seeds]))
        s5_std = float(np.std([per_seed[sk][arm]["All"][-1] for sk in done_seeds]))
        c09 = float(np.mean([np.mean(per_seed[sk][arm]["c09"]) for sk in done_seeds]))
        dmd = md - ftr_md
        per_sess = {}
        for si in range(5):
            per_sess[f"S{si+1}"] = {
                "All": float(np.mean([per_seed[sk][arm]["All"][si] for sk in done_seeds])),
                "Old": float(np.mean([per_seed[sk][arm]["Old"][si] for sk in done_seeds])),
                "New": float(np.mean([per_seed[sk][arm]["New"][si] for sk in done_seeds])),
            }
        summary["arms"][arm] = {
            "S0": s0, "M_d": md, "Old": old, "M_f": mf, "S5_All": s5, "S5_All_std": s5_std,
            "c09": c09, "dM_d": dmd, "per_session": per_sess, "n": len(done_seeds),
        }
        print(f"{arm:<14} {s0:7.2f}% {md:12.2f}% {old:12.2f}% {mf:9.2f}% {c09:13.2f}% {s5:8.2f}% {dmd:+8.2f}%")

    Path(o.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(o.out_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if torch.cuda.is_available():
        print(f"peak_vram_mb = {torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)


if __name__ == "__main__":
    main()
