#!/usr/bin/env python
"""Write the label-budget split variants (phase 1 now, phase 2 later).

File: splits/<ds>_cls<m><sel>_img<k>[_r<r>].json
  cls<m><sel>  m labeled classes, put first in "classes" (n_base = m). sel = first: first m classes of the
               source (the paper); top: m classes with the most train images (ties at random, seed 0);
               rand: m random classes (seed r). top when class sizes differ, rand when all are equal.
  img<k>       labeled images per labeled class: 10, 25, 50 (n_use = k), or 80pct (n_use of the source).
  r<r>         draw: r0 keeps the source train order (already a seed-0 permutation per class); r >= 1
               re-permutes each labeled class with default_rng([r, class id]). Draws are nested in k.
variant.ft_epochs keeps the fine-tuning steps of the source setting (batch 32 x accum 4, flush included).
Usage (repo root): python scripts/make_label_splits.py
"""
import copy, hashlib, json, math, os
import numpy as np

SRC = {"gc10": "splits/gc10_split.json", "neu": "splits/neu_split.json"}
spe = lambda n: math.ceil(math.ceil(n / 32) / 4)       # optimizer steps per epoch


def make(ds, src, sha, m, sel, k, r):
    s, cls = copy.deepcopy(src), src["classes"]
    n_use = dict(src.get("n_use") or {c: int(src["labeled_frac"] * len(src["train"][c])) for c in cls})
    target = 10 * spe(sum(n_use[c] for c in cls[:src.get("n_base", len(cls) // 2)]))
    if sel == "first":
        lab = cls[:m]
    elif sel == "top":
        tie = np.random.default_rng(0).permutation(len(cls))
        lab = [cls[i] for i in sorted(range(len(cls)), key=lambda i: (-len(src["train"][cls[i]]), tie[i]))[:m]]
    else:
        lab = [cls[i] for i in sorted(np.random.default_rng(r).choice(len(cls), m, replace=False))]
    order = [i for i, c in enumerate(cls) if c in lab] + [i for i, c in enumerate(cls) if c not in lab]
    s["classes"] = [cls[i] for i in order]
    if "folders" in s:
        s["folders"] = [src["folders"][i] for i in order]
    if k != "80pct":
        for c in lab:
            assert k <= n_use[c], (ds, c, k)
            if r:
                tr = src["train"][c]
                s["train"][c] = [tr[i] for i in np.random.default_rng([r, cls.index(c)]).permutation(len(tr))]
            n_use[c] = k
    s["n_base"], s["n_use"] = m, {c: n_use[c] for c in s["classes"]}
    n_lab = sum(n_use[c] for c in lab)
    ep = max(1, round(target / spe(n_lab)))
    s["variant"] = {"source": SRC[ds], "source_sha256": sha, "labeled_classes": lab, "selection": sel,
                    "k": k, "draw": r, "n_labeled": n_lab, "ft_epochs": ep, "ft_steps": ep * spe(n_lab)}
    return f"splits/{ds}_cls{m}{sel}_img{k}" + (f"_r{r}" if r is not None else "") + ".json", s


def main():
    out = []
    for ds, path in SRC.items():
        txt = open(path).read()
        src, sha = json.loads(txt), hashlib.sha256(txt.encode()).hexdigest()
        cls, half = src["classes"], src.get("n_base", len(src["classes"]) // 2)
        for k in (10, 25, 50):
            for r in (0, 1, 2):
                out.append(make(ds, src, sha, half, "first", k, r))
        sel = "rand" if len({len(src["train"][c]) for c in cls}) == 1 else "top"
        for m in (half, 2):
            for r in ((0, 1, 2) if sel == "rand" else (None,)):
                out.append(make(ds, src, sha, m, sel, "80pct", r))
    assert len({p for p, _ in out}) == len(out) == 26, len(out)
    assert not any(os.path.exists(p) for p, _ in out), [p for p, _ in out if os.path.exists(p)]
    for p, s in out:
        open(p, "w").write(json.dumps(s, indent=1) + "\n")
        v = s["variant"]
        print(f"{p:40s} base {s['n_base']} | labeled {v['n_labeled']:4d} | epochs {v['ft_epochs']:2d} "
              f"steps {v['ft_steps']:3d} | {', '.join(v['labeled_classes'])}")
    print(f"wrote {len(out)} files")


if __name__ == "__main__":
    main()
