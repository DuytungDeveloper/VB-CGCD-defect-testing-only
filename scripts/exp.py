#!/usr/bin/env python
"""exp - minimal experiment runner.

  exp run TAG -- COMMAND ...     run COMMAND, save runs/<run>/config.json + console.log
  exp table                      build runs/INDEX.txt from every run

After each run, an entry is appended to notes/LOG.md (fill in "why" and "note").

A run can store numbers in runs/<run>/metrics.json as a flat dict {"name": number}.
The command sees the run folder in the environment variable EXP_RUN_DIR.
Optional runs/columns.txt: metric names to show in INDEX.txt, one per line.
"""
import datetime as dt, json, os, platform, subprocess, sys
from pathlib import Path


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip()


def now():
    return dt.datetime.now().astimezone()


def repo_root():
    return Path(sh("git rev-parse --show-toplevel") or ".").resolve()


def git_info(root):
    # untracked files count as dirty; bookkeeping folders do not
    status = sh("git status --porcelain -- . ':(exclude)runs' ':(exclude)notes'", root)
    return {
        "commit": sh("git rev-parse HEAD", root),
        "dirty": bool(status),
        "dirty_files": status.splitlines(),
    }


def cmd_run(tag, command):
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
        "python": sys.version.split()[0],
        "host": platform.node(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"exp: {run_id}", flush=True)

    # run the command, show output live AND save it (like tee)
    env = dict(os.environ, PYTHONUNBUFFERED="1", EXP_RUN_DIR=str(run_dir))
    with open(run_dir / "console.log", "wb") as log:
        p = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for chunk in iter(lambda: p.stdout.read1(65536), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            log.write(chunk)
        rc = p.wait()

    end = now()
    config["finished"] = end.isoformat(timespec="seconds")
    config["duration_min"] = round((end - start).total_seconds() / 60, 1)
    config["exit_code"] = rc
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"exp: done {run_id} exit={rc}")
    cmd_table(quiet=True)
    append_log(root, run_dir, rc)
    return rc


def fmt(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    return f"{v:.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}"


def read_columns(runs):
    col_file = runs / "columns.txt"
    if not col_file.exists():
        return None
    return [l.strip() for l in col_file.read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")]


def append_log(root, run_dir, rc):
    m_path = run_dir / "metrics.json"
    metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
    cols = read_columns(root / "runs") or list(metrics)
    if rc != 0:
        result = f"FAILED exit={rc}"
    elif metrics:
        result = " | ".join(f"{c} {fmt(metrics[c])}" for c in cols if c in metrics)
    else:
        result = "(no metrics.json yet; copy from runs/INDEX.txt later)"
    log = root / "notes" / "LOG.md"
    log.parent.mkdir(exist_ok=True)
    if not log.exists():
        log.write_text("# LOG\n\n## Open questions\n-\n\n---\n")
    with open(log, "a") as f:
        f.write(f"\n## {run_dir.name}\nwhy:\nresult: {result}\nnote:\n")


def cmd_table(quiet=False):
    runs = repo_root() / "runs"
    rows = []
    for cfg_path in runs.glob("*/config.json"):
        cfg = json.loads(cfg_path.read_text())
        m_path = cfg_path.parent / "metrics.json"
        metrics = json.loads(m_path.read_text()) if m_path.exists() else {}
        rows.append((cfg.get("started", ""), cfg, metrics))
    rows.sort(key=lambda r: (r[0], r[1].get("run_id", "")))   # oldest first

    cols = read_columns(runs) or list(dict.fromkeys(k for _, _, m in rows for k in m))

    head = ["started", "tag", "exit", "min", "dirty"] + cols + ["run"]
    table = [head]
    for started, cfg, m in rows:
        table.append([started[:16].replace("T", " "), cfg.get("tag", ""),
                      str(cfg.get("exit_code", "-")), str(cfg.get("duration_min", "-")),
                      "yes" if cfg.get("git", {}).get("dirty") else "no"]
                     + [fmt(m[c]) if c in m else "-" for c in cols] + [cfg.get("run_id", "")])

    w = [max(len(r[i]) for r in table) for i in range(len(head))]
    text_cols = {0, 1, len(head) - 1}                  # left-align text, right-align numbers
    lines = ["  ".join(c.ljust(w[i]) if i in text_cols else c.rjust(w[i]) for i, c in enumerate(r)).rstrip()
             for r in table]
    (runs / "INDEX.txt").write_text("\n".join(lines) + "\n")
    if not quiet:
        print("\n".join(lines))


def main():
    argv = sys.argv[1:]
    if argv == ["table"]:
        return cmd_table()
    if len(argv) >= 4 and argv[0] == "run" and argv[2] == "--":
        sys.exit(cmd_run(argv[1], argv[3:]))
    sys.exit(__doc__)


if __name__ == "__main__":
    main()