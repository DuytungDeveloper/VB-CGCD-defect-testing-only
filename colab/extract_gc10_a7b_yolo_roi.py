#!/usr/bin/env python
"""Step A7b (duytung/12-gc10-dinov3-roi): Class-Agnostic 1-Class YOLOv8n Defect Proposal
trained STRICTLY on S0 labeled base classes (y < 5, zero novel-class labels!) + DINOv3 Full/ROI L2 Fusion.
"""
import argparse, json, os, shutil, time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
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


def build_paths(split, part, rows=None):
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        names = split[part][c] if rows is None else split[part][c][:rows[c]]
        for n in names:
            paths.append(f"{split['root']}/{n}")
            labels.append(k)
    return paths, np.asarray(labels, dtype=np.int64)


def read_xml_boxes(img_path, img_w, img_h):
    p = Path(img_path)
    xml_p = p.parent.parent / "lable" / (p.stem + ".xml")
    if not xml_p.exists():
        return []
    boxes = []
    try:
        root = ET.parse(xml_p).getroot()
        for bb in root.findall(".//bndbox"):
            x1 = max(0.0, float(bb.findtext("xmin")) / img_w)
            y1 = max(0.0, float(bb.findtext("ymin")) / img_h)
            x2 = min(1.0, float(bb.findtext("xmax")) / img_w)
            y2 = min(1.0, float(bb.findtext("ymax")) / img_h)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
    except Exception:
        pass
    return boxes


def union_box(boxes):
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter + 1e-8)


def train_s0_yolo_detector(split, work_tmp="/tmp/gc10_s0_yolo", epochs=30, imgsz=640, seed=0):
    """Train 1-class ('defect') YOLOv8n STRICTLY on S0 labeled base images (y < n_base)."""
    from ultralytics import YOLO

    n_base = split.get("n_base", 5)
    n_use = split.get("n_use") or {c: int(split["labeled_frac"] * len(split["train"][c])) for c in split["classes"]}
    paths, labs = build_paths(split, "train", rows=n_use)
    base_paths = [p for p, y in zip(paths, labs) if y < n_base]

    tmp = Path(work_tmp)
    if tmp.exists():
        shutil.rmtree(tmp)
    img_tr = tmp / "images/train"
    lbl_tr = tmp / "labels/train"
    img_tr.mkdir(parents=True)
    lbl_tr.mkdir(parents=True)

    n_with_box = 0
    for idx, p in enumerate(base_paths):
        with Image.open(p) as im:
            w, h = im.size
        boxes = read_xml_boxes(p, w, h)
        dst_img = img_tr / f"s0_{idx:04d}{Path(p).suffix}"
        os.symlink(os.path.abspath(p), dst_img)
        lines = []
        for x1, y1, x2, y2 in boxes:
            xc, yc, bw, bh = (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        if lines:
            n_with_box += 1
        (lbl_tr / f"s0_{idx:04d}.txt").write_text("\n".join(lines))

    yaml_p = tmp / "gc10_s0.yaml"
    yaml_p.write_text(f"path: {tmp}\ntrain: images/train\nval: images/train\nnc: 1\nnames: ['defect']\n")
    print(f"A7b YOLO: training 1-class yolov8n on {len(base_paths)} S0 base images ({n_with_box} with XML boxes), epochs={epochs}", flush=True)

    yolo = YOLO("yolov8n.pt")
    yolo.train(
        data=str(yaml_p),
        epochs=epochs,
        imgsz=imgsz,
        batch=32,
        device=0 if torch.cuda.is_available() else "cpu",
        seed=seed,
        workers=2,
        project=str(tmp / "runs"),
        name="s0_defect",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best_pt = tmp / "runs/s0_defect/weights/best.pt"
    if not best_pt.exists():
        best_pt = tmp / "runs/s0_defect/weights/last.pt"
    return YOLO(str(best_pt))


@torch.no_grad()
def extract_yolo_roi_features(yolo, model, paths, labs, n_base, device, conf=0.15, pad=0.15, batch_size=32):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    crop_tfm = T.Compose([
        T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    use_amp = device.startswith("cuda") and torch.cuda.is_available()
    feats, ious_base, ious_novel = [], [], []
    n_det = 0

    for i0 in range(0, len(paths), batch_size):
        b_paths = paths[i0:i0 + batch_size]
        b_labs = labs[i0:i0 + batch_size]
        preds = yolo.predict(b_paths, imgsz=640, conf=conf, verbose=False, device=0 if use_amp else "cpu")
        crops = []
        for p, y, res in zip(b_paths, b_labs, preds):
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                xyxyn = res.boxes.xyxyn.cpu().numpy() if (res.boxes is not None and len(res.boxes) > 0) else np.zeros((0, 4))
                if len(xyxyn) > 0:
                    n_det += 1
                    x1, y1 = float(xyxyn[:, 0].min()), float(xyxyn[:, 1].min())
                    x2, y2 = float(xyxyn[:, 2].max()), float(xyxyn[:, 3].max())
                    bw, bh = x2 - x1, y2 - y1
                    box = (max(0.0, x1 - pad * bw), max(0.0, y1 - pad * bh),
                           min(1.0, x2 + pad * bw), min(1.0, y2 + pad * bh))
                else:
                    box = (0.0, 0.0, 1.0, 1.0)

                gt = union_box(read_xml_boxes(p, w, h))
                if gt is not None:
                    iou = box_iou(box, gt)
                    if int(y) < n_base:
                        ious_base.append(iou)
                    else:
                        ious_novel.append(iou)

                px1, py1 = int(box[0] * w), int(box[1] * h)
                px2, py2 = max(int(box[2] * w), px1 + 4), max(int(box[3] * h), py1 + 4)
                crops.append(crop_tfm(im.crop((px1, py1, min(w, px2), min(h, py2)))))

        x_crop = torch.stack(crops).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            f_roi = _get_feat(model(x_crop))
        feats.append(f_roi.cpu().numpy())

    return np.concatenate(feats, axis=0), n_det, ious_base, ious_novel


def fuse_features(x_full, x_roi, alpha=0.5):
    norms = np.linalg.norm(x_full, axis=1, keepdims=True)
    nf = x_full / np.clip(norms, 1e-8, None)
    nr = x_roi / np.clip(np.linalg.norm(x_roi, axis=1, keepdims=True), 1e-8, None)
    fused = (1.0 - alpha) * nf + alpha * nr
    fused = fused / np.clip(np.linalg.norm(fused, axis=1, keepdims=True), 1e-8, None)
    return (fused * norms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a2b-feat-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--split", default="splits/gc10_split.json")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--yolo-epochs", type=int, default=30)
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
    n_base = split.get("n_base", 5)

    yolo = train_s0_yolo_detector(split, epochs=o.yolo_epochs)
    model = load_merged_s0_backbone(str(ckpt), device)

    all_b, all_n, tot_det, tot_imgs = [], [], 0, 0
    for part, pre in (("train", ""), ("test", "test_")):
        paths, labs = build_paths(split, part)
        x_full = np.load(a2b_dir / f"{pre}features-dinov3-vitb16-sl.npy")
        y_full = np.load(a2b_dir / f"{pre}labels-dinov3-vitb16-sl.npy")
        assert len(paths) == len(x_full) and np.array_equal(labs, y_full), f"Split mismatch on {part}!"
        x_roi, n_det, ib, ino = extract_yolo_roi_features(yolo, model, paths, labs, n_base, device, batch_size=o.batch_size)
        x_fused = fuse_features(x_full, x_roi, alpha=o.alpha)
        np.save(out_dir / f"{pre}features-dinov3-vitb16-sl.npy", x_fused)
        np.save(out_dir / f"{pre}labels-dinov3-vitb16-sl.npy", y_full)
        all_b.extend(ib)
        all_n.extend(ino)
        tot_det += n_det
        tot_imgs += len(paths)
        print(f"A7b {part}: fused shape={x_fused.shape}, det={n_det}/{len(paths)}", flush=True)

    iou_b = float(np.mean(all_b)) if all_b else 0.0
    iou_n = float(np.mean(all_n)) if all_n else 0.0
    iou_all = float(np.mean(all_b + all_n)) if (all_b or all_n) else 0.0
    print(f"--yolo_s0_roi detected {tot_det}/{tot_imgs} images, diag_IoU_all={iou_all:.3f} (base={iou_b:.3f}, novel={iou_n:.3f}) in {time.time()-t0:.1f}s", flush=True)
    if torch.cuda.is_available():
        print(f"peak_vram_mb = {torch.cuda.max_memory_allocated() / 1024**2:.1f}", flush=True)


if __name__ == "__main__":
    main()
