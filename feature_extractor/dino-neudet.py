#!/usr/bin/env python
"""NEU-DET phase 1: fine-tune DINO ViT-B/16 (BOFT) on the labeled base images, then extract features.

Reads splits/neu_split.json (scripts/make_neu_split.py). Fine-tunes ONLY on the first 192 train
images of each base class = the phase-2 labeled set, so no D9 leak by construction. Extracts
features for all train (1440) and test (360) images in split-file order and writes the 4 .npy
files read by utils/dataloader.py. No shards, hence no string-sorted merge (D17).
Run through the wrapper:
  python scripts/exp.py run neu_extract --inputs splits/neu_split.json -- bash scripts/run_phase1_neu.sh
"""
import argparse, json, os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch, torchvision
from datasets import Dataset, Image as HFImage
from sl_finetuned_model import finetune_dino

N_BASE = 3


def _xml_crop_box(img_path, pad_frac=0.15, min_side=64.0):
    """Return largest own-class (or any) XML bbox [x1, y1, x2, y2] with context padding, or None."""
    p = Path(img_path)
    xml = p.parent.parent / "label" / f"{p.stem}.xml"
    if not xml.exists():
        return None
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return None
    folder = p.parent.name
    own, other = [], []
    for o in root.iter("object"):
        b = o.find("bndbox")
        if b is None:
            continue
        try:
            vals = [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError):
            continue
        x1, y1, x2, y2 = vals
        if x2 <= x1 or y2 <= y1:
            continue
        name = (o.findtext("name") or "").strip()
        (own if name.split("_", 1)[0] == folder else other).append((x1, y1, x2, y2))
    cands = own or other
    if not cands:
        return None
    x1, y1, x2, y2 = max(cands, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    w, h = max(x2 - x1, min_side), max(y2 - y1, min_side)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    pw, ph = max(16.0, pad_frac * w), max(16.0, pad_frac * h)
    return [cx - 0.5 * w - pw, cy - 0.5 * h - ph, cx + 0.5 * w + pw, cy + 0.5 * h + ph]


@torch.no_grad()
def compute_auto_roi_boxes(paths, device="cuda", img_size=448, pad_frac=0.15, min_side=64.0):
    """Unsupervised AnomalyNCD-style ROI: frozen DINO patch deviation + CLS attention + MEBin sweep."""
    from PIL import Image
    from scipy.ndimage import label, uniform_filter
    from transformers import ViTModel

    uniq = list(dict.fromkeys(paths))
    model = ViTModel.from_pretrained("facebook/dino-vitb16", attn_implementation="eager").to(device).eval()
    tfm = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size), 3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    g = img_size // 16
    norm01 = lambda t: (t - t.min(dim=1, keepdim=True).values) / (t.max(dim=1, keepdim=True).values - t.min(dim=1, keepdim=True).values + 1e-6)

    boxes = {}
    for i in range(0, len(uniq), 32):
        batch_p = uniq[i:i + 32]
        imgs, sizes = [], []
        for p in batch_p:
            with Image.open(p) as im:
                im = im.convert("RGB")
                sizes.append(im.size)
                imgs.append(tfm(im))
        x = torch.stack(imgs).to(device)
        out = model(x, interpolate_pos_encoding=True, output_attentions=True)
        attn = norm01(out.attentions[-1][:, :, 0, 1:].mean(dim=1))
        pt = torch.nn.functional.normalize(out.last_hidden_state[:, 1:, :], dim=-1)
        bg = torch.nn.functional.normalize(pt.median(dim=1, keepdim=True).values, dim=-1)
        dev = norm01(1.0 - (pt * bg).sum(dim=-1))
        smaps = (0.5 * attn + 0.5 * dev).reshape(-1, g, g).cpu().numpy()

        for p, (W, H), smap in zip(batch_p, sizes, smaps):
            sm = uniform_filter(smap, size=3)
            best_score, best_box = -1.0, None
            for tau in (0.45, 0.52, 0.58, 0.65, 0.72):
                lbl, n_comp = label(sm >= tau)
                for cid in range(1, n_comp + 1):
                    ys, xs = np.where(lbl == cid)
                    frac = len(xs) / float(g * g)
                    if not (0.015 <= frac <= 0.65):
                        continue
                    sc = float(sm[ys, xs].mean()) * (1.0 - 0.3 * frac)
                    if sc > best_score:
                        best_score = sc
                        x1, x2 = xs.min() * (W / g), (xs.max() + 1) * (W / g)
                        y1, y2 = ys.min() * (H / g), (ys.max() + 1) * (H / g)
                        bw, bh = max(x2 - x1, min_side), max(y2 - y1, min_side)
                        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                        pw, ph = max(16.0, pad_frac * bw), max(16.0, pad_frac * bh)
                        best_box = [cx - 0.5 * bw - pw, cy - 0.5 * bh - ph, cx + 0.5 * bw + pw, cy + 0.5 * bh + ph]
            boxes[p] = best_box

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ious = []
    for p, b in boxes.items():
        gt = _xml_crop_box(p)
        if b is not None and gt is not None:
            ix1, iy1 = max(b[0], gt[0]), max(b[1], gt[1])
            ix2, iy2 = min(b[2], gt[2]), min(b[3], gt[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            ua = (b[2] - b[0]) * (b[3] - b[1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
            if ua > 0:
                ious.append(inter / ua)
    n_hit = sum(b is not None for b in boxes.values())
    iou_msg = f" | diag_IoU_vs_XML={np.mean(ious):.3f} (n={len(ious)})" if ious else ""
    print(f"dino-neudet: --roi_auto localized {n_hit}/{len(boxes)} images{iou_msg}")
    return boxes


def build(split, part, rows=None):
    sub = "train" if part == "train" else "validation"
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        names = split[part][c] if rows is None else split[part][c][:rows[c]]
        paths += [f"{split['root']}/{n}" if "/" in n else f"{split['root']}/{sub}/images/{c}/{n}"
                  for n in names]                  # GC10: "folder/name"; NEU: bare name
        labels += [k] * len(names)
    return paths, labels


def to_dataset(paths, labels, tfm, roi_xml=False, roi_map=None):
    if roi_map is not None:
        boxes = [roi_map.get(p) for p in paths]
    elif roi_xml:
        boxes = [_xml_crop_box(p) for p in paths]
        n_hit = sum(b is not None for b in boxes)
        print(f"dino-neudet: --roi_xml matched {n_hit}/{len(paths)} images")
    else:
        boxes = [None] * len(paths)
    ds = Dataset.from_dict({"image": paths, "labels": labels, "box": boxes}).cast_column("image", HFImage())
    def trans(ex):
        imgs = []
        for im, b in zip(ex["image"], ex["box"]):
            im = im.convert("RGB")
            if b is not None:
                W, H = im.size
                x1, y1 = max(0, int(round(b[0]))), max(0, int(round(b[1])))
                x2, y2 = min(W, int(round(b[2]))), min(H, int(round(b[3])))
                if x2 > x1 and y2 > y1:
                    im = im.crop((x1, y1, x2, y2))
            imgs.append(tfm(im))
        return {"images": imgs, "labels": torch.tensor(ex["labels"])}
    return ds.with_transform(trans)


@torch.no_grad()
def extract(model, ds, device, bs, cls=False, interp=False):
    model.to(device).eval()
    feats, labs = [], []
    for b in torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2):
        out = model(b["images"].to(device), interpolate_pos_encoding=True) if interp else model(b["images"].to(device))
        feats.append((out.last_hidden_state[:, 0] if cls else out.pooler_output).cpu().numpy())
        labs.append(b["labels"].numpy())
    return np.concatenate(feats), np.concatenate(labs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="splits/neu_split.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="dino_vitb16")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_finetune", action="store_true",
                    help="label-free: frozen DINO, CLS token of last_hidden_state, no fine-tuning")
    ap.add_argument("--resize", choices=["crop", "full"], default="crop",
                    help="crop = Resize 256 + CenterCrop 224 (NEU, CIFAR); full = whole image to 224x224 (GC10)")
    ap.add_argument("--img_size", type=int, nargs=2, default=[224, 224], metavar=("H", "W"),
                    help="--resize full only: target H W; other than 224 224 uses interpolate_pos_encoding")
    ap.add_argument("--roi_xml", action="store_true",
                    help="oracle ROI: crop largest XML defect bbox (+15% pad) before resize")
    ap.add_argument("--roi_auto", action="store_true",
                    help="unsupervised ROI: frozen DINO patch deviation + CLS attention + MEBin crop")
    a = ap.parse_args()
    torch.manual_seed(a.seed)                      # every seed works, including 0 (cf. D4)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    split = json.load(open(a.split))
    n_base = split.get("n_base", N_BASE)
    n_use = split.get("n_use") or {c: int(split["labeled_frac"] * len(split["train"][c])) for c in split["classes"]}
    resize = ([torchvision.transforms.Resize(tuple(a.img_size), 3)] if a.resize == "full" else          # whole image
              [torchvision.transforms.Resize(int(224 / 0.875), 3), torchvision.transforms.CenterCrop(224)])  # as CIFAR
    tfm = torchvision.transforms.Compose(resize + [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    roi_map = None
    if a.roi_auto:
        all_p = build(split, "train")[0] + build(split, "test")[0]
        roi_map = compute_auto_roi_boxes(all_p, device=a.device)

    if a.no_finetune:                              # label-free: no label is used anywhere here
        from transformers import ViTModel
        model = ViTModel.from_pretrained("facebook/dino-vitb16")
        name = a.model.replace("_", "-")           # dino-vitb16 (no "-sl")
        print("dino-neudet: NO fine-tuning (label-free); features = CLS token of last_hidden_state")
    else:
        p, l = build(split, "train", rows=n_use)
        keep = [i for i, y in enumerate(l) if y < n_base]
        ft = to_dataset([p[i] for i in keep], [l[i] for i in keep], tfm, roi_xml=a.roi_xml, roi_map=roi_map)
        print(f"dino-neudet: fine-tune on {len(ft)} labeled images of {n_base} base classes")
        model = finetune_dino(ft, len(split["classes"]), model_name=a.model)
        name = a.model.replace("_", "-") + "-sl"
    out = f"{a.output_dir}/{name}"
    os.makedirs(out, exist_ok=True)
    for part, pre in (("train", ""), ("test", "test_")):
        x, y = extract(model, to_dataset(*build(split, part), tfm, roi_xml=a.roi_xml, roi_map=roi_map),
                       a.device, a.batch_size, cls=a.no_finetune,
                       interp=(a.resize == "full" and a.img_size != [224, 224]))
        np.save(f"{out}/{pre}features-{name}.npy", x)
        np.save(f"{out}/{pre}labels-{name}.npy", y)
        print(f"dino-neudet: {part} features {x.shape}, per class {np.bincount(y).tolist()}, "
              f"range [{x.min():.4f}, {x.max():.4f}], finite {bool(np.isfinite(x).all())}")
    if torch.cuda.is_available():
        print(f"dino-neudet: peak_vram_mb = {torch.cuda.max_memory_allocated() / 1024**2:.1f}")


if __name__ == "__main__":
    main()
