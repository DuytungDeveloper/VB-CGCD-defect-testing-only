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
import numpy as np
import torch, torchvision
from datasets import Dataset, Image as HFImage
from sl_finetuned_model import finetune_dino

N_BASE = 3


def build(split, part, rows=None):
    sub = "train" if part == "train" else "validation"
    paths, labels = [], []
    for k, c in enumerate(split["classes"]):
        names = split[part][c] if rows is None else split[part][c][:rows[c]]
        paths += [f"{split['root']}/{n}" if "/" in n else f"{split['root']}/{sub}/images/{c}/{n}"
                  for n in names]                  # GC10: "folder/name"; NEU: bare name
        labels += [k] * len(names)
    return paths, labels


def to_dataset(paths, labels, tfm):
    ds = Dataset.from_dict({"image": paths, "labels": labels}).cast_column("image", HFImage())
    def trans(ex):
        return {"images": [tfm(im.convert("RGB")) for im in ex["image"]], "labels": torch.tensor(ex["labels"])}
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
    a = ap.parse_args()
    torch.manual_seed(a.seed)                      # every seed works, including 0 (cf. D4)

    split = json.load(open(a.split))
    n_base = split.get("n_base", N_BASE)
    n_use = split.get("n_use") or {c: int(split["labeled_frac"] * len(split["train"][c])) for c in split["classes"]}
    resize = ([torchvision.transforms.Resize(tuple(a.img_size), 3)] if a.resize == "full" else          # whole image
              [torchvision.transforms.Resize(int(224 / 0.875), 3), torchvision.transforms.CenterCrop(224)])  # as CIFAR
    tfm = torchvision.transforms.Compose(resize + [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    if a.no_finetune:                              # label-free: no label is used anywhere here
        from transformers import ViTModel
        model = ViTModel.from_pretrained("facebook/dino-vitb16")
        name = a.model.replace("_", "-")           # dino-vitb16 (no "-sl")
        print("dino-neudet: NO fine-tuning (label-free); features = CLS token of last_hidden_state")
    else:
        p, l = build(split, "train", rows=n_use)
        keep = [i for i, y in enumerate(l) if y < n_base]
        ft = to_dataset([p[i] for i in keep], [l[i] for i in keep], tfm)
        print(f"dino-neudet: fine-tune on {len(ft)} labeled images of {n_base} base classes")
        model = finetune_dino(ft, len(split["classes"]), model_name=a.model)
        name = a.model.replace("_", "-") + "-sl"
    out = f"{a.output_dir}/{name}"
    os.makedirs(out, exist_ok=True)
    for part, pre in (("train", ""), ("test", "test_")):
        x, y = extract(model, to_dataset(*build(split, part), tfm), a.device, a.batch_size, cls=a.no_finetune,
                       interp=(a.resize == "full" and a.img_size != [224, 224]))
        np.save(f"{out}/{pre}features-{name}.npy", x)
        np.save(f"{out}/{pre}labels-{name}.npy", y)
        print(f"dino-neudet: {part} features {x.shape}, per class {np.bincount(y).tolist()}, "
              f"range [{x.min():.4f}, {x.max():.4f}], finite {bool(np.isfinite(x).all())}")


if __name__ == "__main__":
    main()
