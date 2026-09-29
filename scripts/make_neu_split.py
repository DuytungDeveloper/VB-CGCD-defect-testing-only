#!/usr/bin/env python
"""Write splits/neu_split.json: the fixed NEU-DET protocol for VB-CGCD (GD3, decided 2026-09-30).

  classes = image folders sorted alphabetically -> ids 0..5
  base    = ids 0..2 (50% labeled); novel = ids 3..5 in ONE online session (k = 3)
  train   = datasets/NEU-DET/train (240/class); test = datasets/NEU-DET/validation (60/class)
  train order per class = random permutation (seed 0): first 192 (80%) = labeled (base) or
          session data (novel); last 48 (20%) = pool for old-class samples. Test order = file index.
  Multi-class images keep their folder label (standard image-level protocol, no special cases).
Usage (repo root): python scripts/make_neu_split.py
"""
import hashlib, json, os, re
import numpy as np

ROOT, OUT, SEED, FRAC, N_TRAIN, N_TEST = "datasets/NEU-DET", "splits/neu_split.json", 0, 0.8, 240, 60


def num(f):                                   # crazing_12.jpg -> 12
    return int(re.search(r"_(\d+)\.\w+$", f).group(1))


def main():
    classes = sorted(os.listdir(f"{ROOT}/train/images"))
    assert classes == sorted(os.listdir(f"{ROOT}/validation/images")) and len(classes) == 6, classes
    rng = np.random.default_rng(SEED)
    split = {"root": ROOT, "classes": classes, "seed": SEED, "labeled_frac": FRAC, "train": {}, "test": {}}
    for c in classes:
        tr = sorted(os.listdir(f"{ROOT}/train/images/{c}"), key=num)
        te = sorted(os.listdir(f"{ROOT}/validation/images/{c}"), key=num)
        assert (len(tr), len(te)) == (N_TRAIN, N_TEST), (c, len(tr), len(te))
        assert not set(tr) & set(te), f"{c}: same file name in train and test"
        split["train"][c] = [tr[i] for i in rng.permutation(len(tr))]
        split["test"][c] = te
    n_lab = int(FRAC * N_TRAIN)
    print(f"classes (id order): {classes}")
    print(f"per class: train {N_TRAIN} = {n_lab} labeled/session + {N_TRAIN - n_lab} pool; test {N_TEST}")
    txt = json.dumps(split, indent=1) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and open(OUT).read() == txt:
        print(f"{OUT} unchanged")
    else:
        open(OUT, "w").write(txt)
        print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(txt.encode()).hexdigest())


if __name__ == "__main__":
    main()
