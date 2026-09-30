#!/usr/bin/env python
"""Write splits/gc10_split.json: the fixed GC10-DET protocol (GD3, decided 2026-09-30).

  classes = folders 1..10 -> ids 0..9; base = ids 0..4, novel = ids 5..9 (one session, k = 5)
  removed = the 12 image names that sit in two class folders with pixel-identical content (24 files)
  test    = at most ~20% of every class, split BY COIL (3rd filename field): a coil is assigned
            globally to train or test, so frames of one coil never straddle the split
  train order per class = random permutation (seed 0): first n_use (80%) = labeled (base) or
            session data (novel), rest = pool. Paths are "folder/name" relative to root.
  diagnostics (never used by the pipeline): multi-defect images, folder/XML mismatches, no XML.
Usage (repo root): python scripts/make_gc10_split.py
"""
import collections, hashlib, json, math, os
import xml.etree.ElementTree as ET
import numpy as np

ROOT, OUT, SEED, TEST_FRAC, LAB_FRAC, N_BASE = "datasets/GC10-DET", "splits/gc10_split.json", 0, 0.2, 0.8, 5
NAMES = ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot",
         "silk_spot", "inclusion", "rolled_pit", "crease", "waist_folding"]
NORM = {"10_yaozhed": "10_yaozhe"}


def main():
    folders = [str(i) for i in range(1, 11)]
    files = {f: sorted(os.listdir(f"{ROOT}/{f}")) for f in folders}
    seen = collections.Counter(n for f in folders for n in files[f])
    dup = sorted(n for n, c in seen.items() if c > 1)
    assert len(dup) == 12, f"expected 12 duplicated names, got {len(dup)}"
    imgs = {f: [n for n in files[f] if n not in set(dup)] for f in folders}
    coil = lambda n: n.split("_")[2]

    per_coil = collections.defaultdict(collections.Counter)          # coil -> {folder: n images}
    for f in folders:
        for n in imgs[f]: per_coil[coil(n)][f] += 1
    target = {f: math.ceil(TEST_FRAC * len(imgs[f])) for f in folders}
    rng = np.random.default_rng(SEED)
    coils = sorted(per_coil)
    test_coils, tcount = set(), collections.Counter()
    for i in rng.permutation(len(coils)):
        c = coils[i]
        if all(tcount[f] + k <= target[f] for f, k in per_coil[c].items()):
            test_coils.add(c); tcount.update(per_coil[c])

    split = {"root": ROOT, "classes": NAMES, "folders": folders, "n_base": N_BASE, "seed": SEED,
             "labeled_frac": LAB_FRAC, "test_frac_max": TEST_FRAC, "split_by": "coil = 3rd filename field",
             "removed_duplicates": dup, "train": {}, "test": {}, "n_use": {}}
    total = 0
    for f, name in zip(folders, NAMES):
        te = [f"{f}/{n}" for n in imgs[f] if coil(n) in test_coils]
        tr = [f"{f}/{n}" for n in imgs[f] if coil(n) not in test_coils]
        tr = [tr[i] for i in rng.permutation(len(tr))]
        split["train"][name], split["test"][name], split["n_use"][name] = tr, te, int(LAB_FRAC * len(tr))
        total += len(tr) + len(te)
        print(f"{f:>2} {name:14s} total {len(imgs[f]):4d} | train {len(tr):4d} (use {split['n_use'][name]:4d}, "
              f"pool {len(tr) - split['n_use'][name]:3d}) | test {len(te):3d} ({100 * len(te) / len(imgs[f]):4.1f}%)")
    assert total == 2312 - 24, total

    diag = {"multi_defect": [], "folder_not_in_xml": [], "no_xml": []}
    for f in folders:
        for n in imgs[f]:
            x = f"{ROOT}/label/{os.path.splitext(n)[0]}.xml"
            if not os.path.exists(x): diag["no_xml"].append(f"{f}/{n}"); continue
            s = {NORM.get(o.findtext("name"), o.findtext("name")) for o in ET.parse(x).getroot().iter("object")} - {"d"}
            if len(s) > 1: diag["multi_defect"].append(f"{f}/{n}")
            if not any(t.split("_")[0] == f for t in s): diag["folder_not_in_xml"].append(f"{f}/{n}")
    split["diagnostics"] = diag
    print(f"images {total} | test coils {len(test_coils)} of {len(coils)} | diagnostics "
          + str({k: len(v) for k, v in diag.items()}))

    txt = json.dumps(split, indent=1) + "\n"
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT) and open(OUT).read() == txt:
        print(f"{OUT} unchanged")
    else:
        open(OUT, "w").write(txt); print(f"wrote {OUT}")
    print("sha256", hashlib.sha256(txt.encode()).hexdigest())


if __name__ == "__main__":
    main()
