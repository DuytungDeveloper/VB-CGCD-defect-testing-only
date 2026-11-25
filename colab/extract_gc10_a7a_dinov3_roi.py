#!/usr/bin/env python
"""Step A7a (duytung/12-gc10-dinov3-roi): Label-free DINOv3 Auto-ROI (Patch Deviation + CLS Affinity + MEBin)
+ Full-Image L2 Feature Fusion on GC10-DET.

Reuses S0-finetuned DINOv3 (ft_ckpt.pt) and Full-image features from Step A2b (zero extra Phase 1 training!).
Computes diagnostic IoU vs XML (only for logging, never used for cropping).
"""
import argparse, json, os, shutil, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

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
    print(f"A7a load_merged_s0_backbone({HF_DINOV3}): BOFT targets = {targets}", flush=True)

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


def build_paths(split, part):
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        for n in split[part][c]:
            paths.append(f"{split['root']}/{n}")
            labels.append(k)
    return paths, np.asarray(labels, dtype=np.int64)


def mebin_box(saliency_map, pad=0.15):
    """Multi-threshold MEBin connected-component defect box (normalized x1,y1,x2,y2 in [0,1])."""
    h, w = saliency_map.shape
    s = saliency_map - saliency_map.min()
    mx = s.max()
    if mx < 1e-8:
        return (0.0, 0.0, 1.0, 1.0)
    s = s / mx

    best_score, best_box = -1.0, (0.0, 0.0, 1.0, 1.0)
    for q in (0.65, 0.72, 0.78, 0.84, 0.90):
        thr = float(np.quantile(s, q))
        mask = s >= thr
        lbl, n_cc = ndimage.label(mask)
        for cid in range(1, n_cc + 1):
            ys, xs = np.where(lbl == cid)
            area = len(ys) / float(h * w)
            if area < 0.008 or area > 0.65:
                continue
            mean_val = float(s[ys, xs].mean())
            sc = mean_val * (area ** 0.25)
            if sc > best_score:
                y1, y2 = ys.min() / h, (ys.max() + 1) / h
                x1, x2 = xs.min() / w, (xs.max() + 1) / w
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0.0, x1 - pad * bw)
                y1 = max(0.0, y1 - pad * bh)
                x2 = min(1.0, x2 + pad * bw)
                y2 = min(1.0, y2 + pad * bh)
                best_score = sc
                best_box = (x1, y1, x2, y2)
    return best_box


def xml_union_box(img_path, img_w, img_h):
    """Read ground-truth XML union box ONLY for diagnostic IoU reporting."""
    p = Path(img_path)
    xml_p = p.parent.parent / "lable" / (p.stem + ".xml")
    if not xml_p.exists():
        return None
    try:
        root = ET.parse(xml_p).getroot()
        xs1, ys1, xs2, ys2 = [], [], [], []
        for bb in root.findall(".//bndbox"):
            xs1.append(float(bb.findtext("xmin")) / img_w)
            ys1.append(float(bb.findtext("ymin")) / img_h)
            xs2.append(float(bb.findtext("xmax")) / img_w)
            ys2.append(float(bb.findtext("ymax")) / img_h)
        if not xs1:
            return None
        return (max(0.0, min(xs1)), max(0.0, min(ys1)), min(1.0, max(xs2)), min(1.0, max(ys2)))
    except Exception:
        return None


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-8)


@torch.no_grad()
def extract_auto_roi_features(model, paths, device, loc_size=384, batch_size=32):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    loc_tfm = T.Compose([
        T.Resize((loc_size, loc_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    crop_tfm = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    patch_sz = int(getattr(model.config, "patch_size", 16))
    grid = loc_size // patch_sz
    use_amp = device.startswith("cuda") and torch.cuda.is_available()

    feats, ious = [], []
    n_loc_ok = 0

    for i0 in range(0, len(paths), batch_size):
        batch_paths = paths[i0:i0 + batch_size]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))

        x_loc = torch.stack([loc_tfm(im) for im in imgs]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x_loc)
            hs = out.last_hidden_state.float()
        cls_tok = F.normalize(hs[:, 0:1, :], dim=-1)                # (B, 1, D)
        patch_tok = F.normalize(hs[:, 1 + n_reg:, :], dim=-1)       # (B, G*G, D)
        assert patch_tok.shape[1] == grid * grid, f"Expected {grid*grid} patches, got {patch_tok.shape[1]}"

        # Patch deviation from median background patch + CLS-patch affinity
        med = patch_tok.median(dim=1, keepdim=True).values
        med = F.normalize(med, dim=-1)
        dev = (1.0 - (patch_tok * med).sum(dim=-1)).clamp(min=0.0)          # (B, G*G)
        aff = (patch_tok * cls_tok).sum(dim=-1).clamp(min=0.0)              # (B, G*G)
        sal = (0.65 * dev + 0.35 * aff).reshape(-1, grid, grid).cpu().numpy()

        crops = []
        for im, sm, p in zip(imgs, sal, batch_paths):
            box = mebin_box(sm, pad=0.15)
            w, h = im.size
            if box != (0.0, 0.0, 1.0, 1.0):
                n_loc_ok += 1
            gt = xml_union_box(p, w, h)
            if gt is not None:
                ious.append(box_iou(box, gt))
            x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), max(int(box[2] * w), int(box[0] * w) + 4), max(int(box[3] * h), int(box[1] * h) + 4)
            crops.append(crop_tfm(im.crop((x1, y1, min(w, x2), min(h, y2)))))

        x_crop = torch.stack(crops).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            f_roi = _get_feat(model(x_crop))
        feats.append(f_roi.cpu().numpy())

    return np.concatenate(feats, axis=0), n_loc_ok, ious


def fuse_features(x_full, x_roi, alpha=0.5):
    """L2-normalized fusion preserving original x_full norms (exact same scale as A2b)."""
    norms = np.linalg.norm(x_full, axis=1, keepdims=True)
    nf = x_full / np.clip(norms, 1e-8, None)
    nr = x_roi / np.clip(np.linalg.norm(x_roi, axis=1, keepdims=True), 1e-8, None)
    fused = (1.0 - alpha) * nf + alpha * nr
    fused = fused / np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8, None)
    return (fused * norms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2b-feat-root", required=True, help="features/*__colab_gc10_dinov3_ep20_r0")
    ap.add_argument("--out-root", required=True, help="features/<run_id> output directory")
    ap.add_argument("--split", default="splits/gc10_split.json")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of Auto-ROI feature in fusion")
    ap.add_argument("--loc-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    o = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a2b_root = Path(o.a2b_feat_root)
    a2b_dir = a2b_root / "dinov3-vitb16-sl"
    ckpt = a2b_root / "ft_ckpt.pt"
    out_root = Path(o.out_root)
    out_dir = out_root / "dinov3-vitb16-sl"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ckpt, out_root / "ft_ckpt.pt")

    split = json.load(open(o.split))
    model = load_merged_s0_backbone(str(ckpt), device)

    all_ious, tot_loc, tot_imgs = [], 0, 0
    for part, pre in (("train", ""), ("test", "test_")):
        paths, labs = build_paths(split, part)
        x_full = np.load(a2b_dir / f"{pre}features-dinov3-vitb16-sl.npy")
        y_full = np.load(a2b_dir / f"{pre}labels-dinov3-vitb16-sl.npy")
        assert len(paths) == len(x_full) and np.array_equal(labs, y_full), f"Split mismatch on {part}!"
        x_roi, n_ok, ious = extract_auto_roi_features(model, paths, device, loc_size=o.loc_size, batch_size=o.batch_size)
        x_fused = fuse_features(x_full, x_roi, alpha=o.alpha)
        np.save(out_dir / f"{pre}features-dinov3-vitb16-sl.npy", x_fused)
        np.save(out_dir / f"{pre}labels-dinov3-vitb16-sl.npy", y_full)
        all_ious.extend(ious)
        tot_loc += n_ok
        tot_imgs += len(paths)
        print(f"A7a {part}: fused shape={x_fused.shape}, finite={bool(np.isfinite(x_fused).all())}, loc={n_ok}/{len(paths)}", flush=True)

    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    print(f"--dinov3_auto_roi localized {tot_loc}/{tot_imgs} images, diag_IoU_vs_XML={mean_iou:.3f} (n_xml={len(all_ious)}) in {time.time()-t0:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"peak_vram_mb = {torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)


if __name__ == "__main__":
    main()
