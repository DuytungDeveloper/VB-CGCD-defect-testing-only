#!/usr/bin/env python
"""VB-CGCD only: read runs/<run>/console.log, write runs/<run>/metrics.json.

  python scripts/extract_vbcgcd.py runs/<run>

Log format comes from classifier/mngmm.py (one result table per stage)
and main.py (cluster sizes). Keys are flat, e.g. S1_All, S1_New, M_d.
"""
import json, re, sys
from pathlib import Path

ROW = re.compile(r"^\|\s*(All|Old|Novel|S0)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*([-\d.eE+]+|nan)\s*\|")
ESTOP = re.compile(r"Step (\d+):.*early_stop_flag = (True|False)")   # no "^": tqdm can share the line
NOVEL = re.compile(r"^Number of Novel Samples: (\d+) / (\d+)")


def clean_lines(text):
    out = []
    for line in text.split("\n"):
        parts = [s for s in line.split("\r") if s.strip()]   # keep the last tqdm redraw only
        if parts:
            out.append(parts[-1].rstrip())
    return out


def extract(lines):
    m = {}
    stage, table, buf = 0, None, None
    seen_novel = False
    for ln in lines:
        # cluster sizes: "(array([...]), array([...]))", may span several lines
        if buf is not None or ln.startswith("(array(["):
            buf = (buf or "") + " " + ln
            if "]))" in ln:
                arrays = re.findall(r"array\(\[([^\]]*)\]", buf)
                if len(arrays) == 2:
                    sizes = [int(x) for x in arrays[1].replace(",", " ").split()]
                    m[f"S{stage}_cluster_max_min"] = round(max(sizes) / max(min(sizes), 1), 2)
                buf = None
            continue
        if re.match(r"^\|\s*TestSet", ln):
            table = {}
            continue
        r = ROW.match(ln)
        if r and table is not None:
            table[r.group(1)] = float(r.group(2))
            if len(table) == 4:                        # table printed at the END of a stage
                m[f"S{stage}_All"] = table["All"]
                if stage > 0:
                    m[f"S{stage}_Old"] = table["Old"]
                    m[f"S{stage}_New"] = table["Novel"]
                m[f"S{stage}_S0test"] = table["S0"]
                stage, table, seen_novel = stage + 1, None, False
            continue
        r = NOVEL.match(ln)
        if r:
            m[f"S{stage}_novel_kept"] = int(r.group(1))
            seen_novel = True
            continue
        r = ESTOP.search(ln)
        if r:
            phase = "S" if seen_novel else "F"         # F = before novelty filter, S = after
            m[f"S{stage}_estop_{phase}"] = int(r.group(1))

    last = stage - 1
    if last >= 1:
        m["M_d"] = round(sum(m[f"S{i}_New"] for i in range(1, last + 1)) / last, 2)
        m["M_f"] = round(m["S0_S0test"] - m[f"S{last}_S0test"], 2)   # proxy, see notes
        m["final_All"] = m[f"S{last}_All"]
    m["n_stages"] = stage
    return m


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run = Path(sys.argv[1])
    metrics = extract(clean_lines((run / "console.log").read_text(errors="replace")))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for k, v in metrics.items():
        print(f"{k:22s} {v}")