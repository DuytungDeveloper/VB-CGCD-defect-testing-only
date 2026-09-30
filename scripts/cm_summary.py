#!/usr/bin/env python
"""Summarise the 'confusion stage N' lines (main.py --print_confusion) of the runs tagged GROUP_s<N>.
Prints balanced accuracy (base / novel / all, mean +- std over runs), per-class recall and
base->novel share (confusions summed over runs), and the largest confusions.
Usage: python scripts/cm_summary.py GROUP [--split splits/gc10_split.json] [--stage 1]"""
import argparse, glob, json, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group"); ap.add_argument("--split", default="splits/gc10_split.json")
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    sp = json.load(open(a.split)); names = sp["classes"]; nb = sp.get("n_base", len(names) // 2)
    pat = re.compile(rf"confusion stage {a.stage}: rows=true cols=pred (\[\[.*\]\])")
    runs = sorted(f for f in glob.glob(f"runs/*__{a.group}_s*/console.log")
                  if re.search(rf"__{re.escape(a.group)}_s\d+/", f))
    cms = [np.array(json.loads(pat.search(open(f).read()).group(1))) for f in runs]
    rec = np.array([c.diagonal() / np.maximum(c.sum(1), 1) for c in cms]) * 100
    b = np.stack([rec[:, :nb].mean(1), rec[:, nb:].mean(1), rec.mean(1)], 1)
    sd = b.std(0, ddof=1) if len(b) > 1 else np.zeros(3)
    C = sum(cms); n = C.sum(1)
    print(f"{a.group}: {len(cms)} runs")
    print("balanced acc base / novel / all:", " / ".join(f"{m:.1f} +- {s:.1f}" for m, s in zip(b.mean(0), sd)))
    print("recall per class (summed):", {names[i]: round(100 * C[i, i] / max(n[i], 1)) for i in range(len(names))})
    print("base -> novel share:", {names[i]: round(100 * C[i, nb:].sum() / max(n[i], 1)) for i in range(nb)})
    off = sorted(((int(C[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(len(names)) if i != j),
                 reverse=True)[:6]
    print("top confusions:", off)


if __name__ == "__main__":
    main()
