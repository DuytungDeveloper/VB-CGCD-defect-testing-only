#!/usr/bin/env python
"""exp - minimal experiment runner.

  exp run TAG -- COMMAND ...
"""
import datetime as dt, json, os, platform, subprocess, sys
from pathlib import Path


def sh(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.stdout.strip()


def now():
    return dt.datetime.now().astimezone()


def git_info(root):
    # untracked files count as dirty; bookkeeping folders do not
    status = sh("git status --porcelain -- . ':(exclude)runs' ':(exclude)notes'", root)
    return {
        "commit": sh("git rev-parse HEAD", root),
        "dirty": bool(status),
        "dirty_files": status.splitlines(),
    }


def cmd_run(tag, command):
    root = Path(sh("git rev-parse --show-toplevel") or ".").resolve()
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
    with open(run_dir / "console.log", "wb") as log:
        env = dict(os.environ, PYTHONUNBUFFERED="1")   # keep stdout/stderr in true order
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
    return rc


def main():
    argv = sys.argv[1:]
    if len(argv) < 4 or argv[0] != "run" or "--" not in argv:
        sys.exit(__doc__)
    cut = argv.index("--")
    if cut != 2:
        sys.exit(__doc__)
    sys.exit(cmd_run(argv[1], argv[cut + 1:]))


if __name__ == "__main__":
    main()