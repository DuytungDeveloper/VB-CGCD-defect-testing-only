"""Multi-session dissection bench (QD28; lab/README.vi.md section 5): the steps of lab/replay.py run session by session.

S0 runs once: 04 loads every session (s04_sessions), 05 fits the PCA on S0, 06-08 are the defaults. Each online session
i = 1..n then runs the unchanged functions s09-s14 of lab/replay.py on a session view of the state (session_view):
offset, pool and test sets of session i, params0 = the model after session i-1, the PCA projections of the pool and
of the novel test set of session i. Only step 10 gets a wrapper: lut of session i = lut of session i-1 plus the map of
session i, as to_cluster of main.py. chain, new_state and the step functions of lab/replay.py are not changed, so the
snapshots of the GC10 chains do not go stale.

Status (commit 1 of 3): the runner in memory, no snapshots, and the command
  python lab/sessions.py c0 <gc10 v2 run>      C0: t1 through this runner equals every snapshot of a GC10 chain
Snapshots, the hash chain and the commands run, check, check-resume, check-crn, check-hash come in commits 2 and 3.

Log lines: s10_naming prints "hungarian stage 1:" whatever the session; the log of session i gets "hungarian stage i:",
as main.py prints it (decision of 2026-11-15). The pass prefixes "stage_1_Flearning" and "stage_1_Slearning" go to the
TensorBoard writer only, not to the console, so nothing else is rewritten.
"""
import argparse, contextlib, copy, io, os, sys, time, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "lab")):
    if p not in sys.path:
        sys.path.insert(0, p)
from replay import GC10, STEPS, new_state, chain, snapshots, _fresh, _compare   # noqa: E402
from variants import CUB_T5                                                      # noqa: E402

PRESETS = {"gc10": GC10, "cub200": CUB_T5}
N_SESSIONS = {"t1": 1, "t5": 5}
SESSION_KEYS = ("pool_x", "true_pool_y", "test1_x", "true_test1_y", "test_old_x", "true_test_old_y",
                "test_all_x", "true_test_all_y")
SESSION_OUT = ("ids", "cluster", "inertia", "pi_test", "lut", "params_f", "keep", "params1", "acc1")   # 09-14
STAGE_LINE = "hungarian stage 1:"


def new_state_sessions(args):
    """new_state for load_mode t1 or t5: the classifier object does not read load_mode, so new_state runs on a copy of
    args with load_mode t1 (it accepts t1 only) and the state then gets the real args (decision of 2026-11-15)."""
    assert args.load_mode in N_SESSIONS, f"load_mode {args.load_mode}: t1 or t5"
    a1 = copy.copy(args)
    a1.load_mode = "t1"
    st = new_state(a1)
    st["args"] = args
    return st


def s04_sessions(st):
    """04 All sessions of the protocol at once (main.Dataloader with args.load_mode), as main.py builds them, so the
    global RNG leaves this step in the state main.py has before the PCA of S0.
    In: args. Out: (keys of s04_loader for S0 and session 1, [session i = 1..n: dict of SESSION_KEYS])."""
    import main
    a = st["args"]
    tr, te, te_old, te_all = (list(it) for it in main.Dataloader(a))
    n = N_SESSIONS[a.load_mode]
    assert len(tr) == n + 1 and len(te) == n + 1, f"{a.load_mode}: S0 and {n} online sessions, got {len(tr)} and {len(te)}"
    y = lambda d: np.asarray(d._y)
    sess = [dict(pool_x=tr[i]._x, true_pool_y=y(tr[i]), test1_x=te[i]._x, true_test1_y=y(te[i]),
                 test_old_x=te_old[i]._x, true_test_old_y=y(te_old[i]), test_all_x=te_all[i]._x,
                 true_test_all_y=y(te_all[i])) for i in range(1, n + 1)]
    out = dict(s0_x=tr[0]._x, s0_y=y(tr[0]), test0_x=te[0]._x, true_test0_y=y(te[0]), offset=a.base, **sess[0])
    return out, sess


def run_s0(st):
    """Seeds in the order of main.py (as chain does at 04), 04 (s04_sessions), then the default steps 05-08 on st.
    Returns the list of sessions of s04_sessions."""
    import torch, jax
    a = st["args"]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    jax.random.PRNGKey(a.seed)
    out, sess = s04_sessions(st)
    st.update(out)
    chain(st, start="05", upto="08")
    return sess


def session_view(st, sess, i, prev):
    """State for session i (1-based): st after 08 without the outputs of 09-14, the sets of session i, offset,
    z_pool and z_test1 (PCA of S0; no RNG), params0 = params1 of session i-1 (prev; None for i = 1), session, seen."""
    a = st["args"]
    off = a.base + (i - 1) * a.increment
    view = {k: v for k, v in st.items() if k not in SESSION_OUT}
    view.update(sess[i - 1])
    t = st["pca"].transform
    view.update(offset=off, z_pool=t(view["pool_x"]), z_test1=t(view["test1_x"]), session=i, seen=off + a.increment)
    if prev is not None:
        view["params0"] = prev["params1"]
    return view


def run_session(st, sess, i, prev):
    """Session i: s09-s14 of lab/replay.py on session_view, lut made cumulative after 10. Returns (view, log of the
    session, with "hungarian stage i:")."""
    a = st["args"]
    view = session_view(st, sess, i, prev)
    off, buf = view["offset"], io.StringIO()
    with contextlib.redirect_stdout(buf):
        chain(view, start="09", upto="10")
        lut = prev["lut"].copy() if prev is not None else np.arange(a.num_classes)
        for c, t in view["pi_test"].items():
            lut[off + t] = off + c
        view["lut"] = lut
        chain(view, start="11", upto="14")
    log = buf.getvalue()
    assert log.count(STAGE_LINE) == 1, f"session {i}: {log.count(STAGE_LINE)} lines '{STAGE_LINE}' (expected 1)"
    log = log.replace(STAGE_LINE, f"hungarian stage {i}:")
    sys.stdout.write(log)
    return view, log


def run_all(st):
    """S0 and every session in memory. Returns (sessions, [view of session i], [log of session i])."""
    sess = run_s0(st)
    views, logs, prev = [], [], None
    for i in range(1, len(sess) + 1):
        prev, log = run_session(st, sess, i, prev)
        views.append(prev)
        logs.append(log)
    return sess, views, logs


def _device(blobs=None):
    import jax
    device = jax.default_backend()
    if device != "gpu" and os.environ.get("REPLAY_ALLOW_CPU") != "1":
        raise SystemExit(f"sessions: JAX runs on {device}, not on the GPU (A1); REPLAY_ALLOW_CPU=1 is for tests only")
    if blobs is not None and {b["meta"].get("device", "gpu") for b in blobs.values()} != {device}:
        raise SystemExit("sessions: the snapshots ran on another device")
    return device


def cmd_c0(argv):
    """C0 (QD28): python lab/sessions.py c0 <gc10 v2 run>. Runs the args of the chain (t1) through this runner in
    memory and compares every output key of every snapshot 04-14 with it, by the rule of replay.rerun (discrete equal,
    floats within 1e-6 x max|value|, PCA and cluster models by _same). Also: z_pool, z_test1 of the session view equal
    to the bit those of snapshot 05; the log of 09-14 (report only). Writes nothing."""
    p = argparse.ArgumentParser(prog="sessions.py c0")
    p.add_argument("run", help="run id or tag of a GC10 chain made by lab/replay.py run")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: splits/, features/)"
    blobs = snapshots(o.run)
    _fresh(blobs)
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    assert args.load_mode == "t1" and args.dataset == "gc10", "C0 runs on the t1 chains of GC10"
    st = new_state_sessions(args)
    device, t0 = _device(blobs), time.time()
    sess, views, logs = run_all(st)
    assert len(sess) == 1, "t1: one online session"
    view, rows = views[0], []
    for k in STEPS:
        src = st if k in ("04", "05", "06", "07", "08") else view
        for key, ref in blobs[k]["out"].items():
            kind, d, ok = _compare(src[key], ref, 1e-6) if key in src else ("missing", float("nan"), False)
            rows.append((k, key, kind, d, ok))
    for key in ("z_pool", "z_test1"):
        kind, d, ok = _compare(view[key], blobs["05"]["out"][key], 0.0)
        rows.append(("view", key, kind, d, ok))
    print(f"\nC0 {o.run} | device {device} | {time.time() - t0:.0f} s")
    for k, key, kind, d, ok in rows:
        print(f"  {'OK  ' if ok else 'DIFF'} {k:4s} {key:16s} {kind:22s} max|diff| {d:.3g}")
    same_log = logs[0] == "".join(blobs[k]["log"] for k in STEPS[STEPS.index("09"):])
    print(f"  log of 09-14 identical to the snapshots (report only): {same_log}")
    ok = all(r[4] for r in rows)
    print(f"C0 {'OK' if ok else 'DIFF'}: {sum(r[4] for r in rows)}/{len(rows)} keys")
    return ok


COMMANDS = {"c0": cmd_c0}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python lab/sessions.py {{{','.join(COMMANDS)}}} ...")
    sys.exit(0 if COMMANDS[sys.argv[1]](sys.argv[2:]) else 1)
