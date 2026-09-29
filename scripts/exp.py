#!/usr/bin/env python
"""exp - minimal experiment runner.

  exp run TAG -- COMMAND ...     run COMMAND, save runs/<run>/config.json + console.log
      [--inputs PATH... | --inputs -] before "--": inputs for this run instead of runs/inputs.txt (- = none)
  exp table                      build runs/INDEX.txt from every run
  exp agg GROUP                  mean/std over runs tagged GROUP_s<N> -> runs/agg_GROUP.txt

After each run, an entry is appended to notes/LOG.md (fill in "why" and "note").

A run can store numbers in runs/<run>/metrics.json as a flat dict {"name": number}.
The command sees the run folder in the environment variable EXP_RUN_DIR.

Optional plain-text files, one item per line:
  runs/columns.txt   metric names shown in INDEX.txt and in the LOG result line
  runs/inputs.txt    files/folders the runs read; hashed at start, warns if content changed
"""
import datetime as dt, hashlib, json, os, platform, re, signal, statistics as st, subprocess, sys
from pathlib import Path


class Interrupted(Exception):
    pass


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip()


def now():
    return dt.datetime.now().astimezone()


def repo_root():
    return Path(sh("git rev-parse --show-toplevel") or ".").resolve()


def read_list(path):
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith("#")]


def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_runs(runs):
    """[(started, config, metrics)] for every run folder, oldest first."""
    rows = []
    for cfg_path in runs.glob("*/config.json"):
        cfg = load_json(cfg_path)
        rows.append((cfg.get("started", ""), cfg, load_json(cfg_path.parent / "metrics.json")))
    rows.sort(key=lambda r: (r[0], r[1].get("run_id", "")))
    return rows


# ---------------------------------------------------------------- what the run used
def git_info(root):
    # untracked files count as dirty; bookkeeping folders do not
    status = sh("git status --porcelain -- . ':(exclude)runs' ':(exclude)notes'", root)
    return {"commit": sh("git rev-parse HEAD", root), "dirty": bool(status), "dirty_files": status.splitlines()}


def env_info(root):
    # full package list, saved once per unique environment
    freeze = sh(f'"{sys.executable}" -m pip freeze')
    digest = hashlib.sha256(freeze.encode()).hexdigest()[:12]
    env_file = root / "runs" / "envs" / f"{digest}.txt"
    if not env_file.exists():
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(freeze + "\n")
    return {
        "id": digest,
        "pip_freeze": f"runs/envs/{digest}.txt",
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "gpu": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader") or None,
    }


def hash_path(p):
    """sha256 over file names + contents (a folder is walked in sorted order). None if missing."""
    if not p.exists():
        return None
    if p.is_file():
        files = [(p.name, p)]
    else:
        files = []
        for d, _, names in os.walk(p, followlinks=True):
            files += [(str((Path(d) / n).relative_to(p)), Path(d) / n) for n in names]
        files.sort()
    h = hashlib.sha256()
    for name, f in files:
        h.update(name.encode() + b"\0")
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()[:16]


def inputs_info(root, started, paths=None):
    if paths is None:
        paths = read_list(root / "runs" / "inputs.txt")
    if not paths:
        return {}, []
    print("exp: hashing inputs ...", flush=True)
    inputs = {p: hash_path(root / p) for p in paths}
    warnings = []
    earlier = [cfg for s, cfg, _ in load_runs(root / "runs") if s and s < started]
    for p, h in inputs.items():
        if h is None:
            warnings.append(f"input missing: {p}")
            continue
        prev = [cfg for cfg in earlier if cfg.get("inputs", {}).get(p)]
        if prev and prev[-1]["inputs"][p] != h:
            warnings.append(f"input CHANGED since {prev[-1]['run_id']}: {p}")
    return inputs, warnings


# ---------------------------------------------------------------- run
def cmd_run(tag, command, source="inputs.txt", paths=None):
    root = repo_root()
    start = now()
    run_id = start.strftime("%Y%m%d_%H%M%S") + "__" + tag
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)

    config = {
        "run_id": run_id,
        "tag": tag,
        "command": command,
        "cwd": str(root),
        "started": start.isoformat(timespec="seconds"),
        "git": git_info(root),
        "env": env_info(root),
        "python": sys.version.split()[0],
        "host": platform.node(),
    }
    config["inputs_source"] = source
    config["inputs"], config["warnings"] = inputs_info(root, config["started"], paths)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"exp: {run_id}", flush=True)
    for w in config["warnings"]:
        print(f"exp: WARNING {w}", flush=True)

    # Ctrl+C, kill, closed terminal -> stop the command but still record the run
    def on_signal(signum, frame):
        raise Interrupted(signal.Signals(signum).name)
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)

    env = dict(os.environ, PYTHONUNBUFFERED="1", EXP_RUN_DIR=str(run_dir))
    rc, reason = None, None
    with open(run_dir / "console.log", "wb") as log:
        p = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            for chunk in iter(lambda: p.stdout.read1(65536), b""):
                log.write(chunk)
                log.flush()                                  # so a follow-up step can read the log
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except OSError:                              # terminal is gone; keep logging
                    pass
            rc = p.wait()
        except Interrupted as e:
            reason = str(e)
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            rc = "interrupted"
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_IGN)                     # finish bookkeeping undisturbed

    end = now()
    config["finished"] = end.isoformat(timespec="seconds")
    config["duration_min"] = round((end - start).total_seconds() / 60, 1)
    config["exit_code"] = rc
    if reason:
        config["interrupt_signal"] = reason
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"\nexp: done {run_id} exit={rc}")

    for step in (lambda: cmd_table(quiet=True), lambda: append_log(root, run_dir, config)):
        try:
            step()
        except Exception as e:                               # bookkeeping errors never hide the result
            print(f"exp: ERROR in bookkeeping: {type(e).__name__}: {e}")
    return 130 if rc == "interrupted" else rc


# ---------------------------------------------------------------- formatting
def decimals(values):
    """4 decimals if every float in a column is below 1, else 2."""
    floats = [v for v in values if isinstance(v, float)]
    return 4 if floats and all(abs(v) < 1 for v in floats) else 2


def fmt(v, d=2):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return f"{v:.{d}f}"


def align(table, text_cols):
    w = [max(len(r[i]) for r in table) for i in range(len(table[0]))]
    return ["  ".join(c.ljust(w[i]) if i in text_cols else c.rjust(w[i]) for i, c in enumerate(r)).rstrip()
            for r in table]


def exit_text(cfg):
    rc = cfg.get("exit_code")
    if rc == "interrupted":
        return "int"
    if rc is None:
        return "?"                                           # never finished (crash, power loss)
    return str(rc)


# ---------------------------------------------------------------- LOG, table, agg
def append_log(root, run_dir, config):
    rows = load_runs(root / "runs")
    metrics = load_json(run_dir / "metrics.json")
    cols = [c for c in read_list(root / "runs" / "columns.txt") if c in metrics] or list(metrics)
    rc = config["exit_code"]
    if rc == "interrupted":
        result = f"INTERRUPTED ({config.get('interrupt_signal')})"
    elif rc != 0:
        result = f"FAILED exit={rc}"
    elif metrics:
        result = " | ".join(f"{c} {fmt(metrics[c], decimals([m.get(c) for _, _, m in rows]))}"
                            for c in cols if c in metrics)
    else:
        result = "(no metrics.json)"
    if config.get("warnings"):
        result += "   WARNING: " + "; ".join(config["warnings"])
    log = root / "notes" / "LOG.md"
    log.parent.mkdir(exist_ok=True)
    if not log.exists():
        log.write_text("# LOG\n\n## Open questions\n-\n\n---\n")
    with open(log, "a") as f:
        f.write(f"\n## {run_dir.name}\nwhy:\nresult: {result}\nnote:\n")


def cmd_table(quiet=False):
    runs = repo_root() / "runs"
    rows = load_runs(runs)
    cols = read_list(runs / "columns.txt") or list(dict.fromkeys(k for _, _, m in rows for k in m))
    dec = {c: decimals([m.get(c) for _, _, m in rows]) for c in cols}

    head = ["started", "tag", "exit", "min", "dirty", "env", "input"] + cols + ["run"]
    table = [head]
    for started, cfg, m in rows:
        inputs = cfg.get("inputs")
        first = next(iter(inputs.values())) if inputs else None
        table.append([started[:16].replace("T", " "), cfg.get("tag", ""), exit_text(cfg),
                      str(cfg.get("duration_min", "-")),
                      {True: "yes", False: "no"}.get(cfg.get("git", {}).get("dirty"), "?"),
                      cfg.get("env", {}).get("id", "?")[:6],
                      "?" if inputs is None else ("-" if not inputs else (first or "absent")[:6])]
                     + [fmt(m[c], dec[c]) if c in m else "-" for c in cols] + [cfg.get("run_id", "")])
    lines = align(table, {0, 1, len(head) - 1})
    (runs / "INDEX.txt").write_text("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines))


def cmd_agg(group):
    runs = repo_root() / "runs"
    pattern = re.compile(rf"^{re.escape(group)}_s(\d+)$")
    tagged = [(cfg, m) for _, cfg, m in load_runs(runs) if pattern.match(cfg.get("tag", ""))]
    picked = [(cfg, m) for cfg, m in tagged if cfg.get("exit_code") == 0 and m]
    if len(picked) < 2:
        sys.exit(f"exp: need >= 2 finished runs tagged {group}_s<N> with metrics, found {len(picked)}")
    paper = load_json(runs / "paper" / "metrics.json")
    keys = [k for k in picked[0][1]
            if all(isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool) for _, m in picked)]

    # consistency checks: warn, do not stop
    seeds = [pattern.match(cfg["tag"]).group(1) for cfg, _ in picked]
    warn = []
    checks = (("commit", lambda c: c.get("git", {}).get("commit")),
              ("env", lambda c: c.get("env", {}).get("id")),
              ("inputs", lambda c: json.dumps(c.get("inputs"), sort_keys=True)))
    for name, get in checks:
        if len({get(cfg) for cfg, _ in picked}) > 1:
            warn.append(f"WARNING {name} differs across runs")
    if any(cfg.get("git", {}).get("dirty") is not False for cfg, _ in picked):
        warn.append("WARNING some runs are dirty or unknown")
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        warn.append(f"WARNING duplicate seeds: {' '.join(dup)}")
    if len(tagged) > len(picked):
        warn.append(f"WARNING {len(tagged) - len(picked)} tagged run(s) skipped (failed or no metrics)")

    table = [["metric", "n", "mean", "std", "min", "max", "paper"]]
    for k in keys:
        v = [float(m[k]) for _, m in picked]
        d = decimals(v + ([float(paper[k])] if k in paper else []))
        table.append([k, str(len(v)), fmt(st.mean(v), d), fmt(st.stdev(v), d), fmt(min(v), d), fmt(max(v), d),
                      fmt(float(paper[k]), d) if k in paper else "-"])
    text = [f"group {group}   seeds {' '.join(seeds)}   std uses n-1   made {now().isoformat(timespec='seconds')}",
            "runs  " + " ".join(cfg["run_id"] for cfg, _ in picked)] + warn + [""] + align(table, {0})
    (runs / f"agg_{group}.txt").write_text("\n".join(text) + "\n")
    print("\n".join(text))


def main():
    argv = sys.argv[1:]
    if argv == ["table"]:
        return cmd_table()
    if len(argv) == 2 and argv[0] == "agg":
        return cmd_agg(argv[1])
    if len(argv) >= 4 and argv[0] == "run" and "--" in argv[2:]:
        sep = argv.index("--", 2)
        opts, command = argv[2:sep], argv[sep + 1:]
        if not opts:
            source, paths = "inputs.txt", None          # old behaviour
        elif opts == ["--inputs", "-"]:
            source, paths = "none", []
        elif opts[0] == "--inputs" and len(opts) > 1 and "-" not in opts[1:]:
            source, paths = "cli", opts[1:]
        else:
            sys.exit(__doc__)
        if command:
            sys.exit(cmd_run(argv[1], command, source, paths))
    sys.exit(__doc__)


if __name__ == "__main__":
    main()