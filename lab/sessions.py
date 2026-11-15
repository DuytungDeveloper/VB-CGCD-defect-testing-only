"""Multi-session dissection bench (QD28; lab/README.vi.md section 5): the steps of lab/replay.py run session by session.

S0 runs once: 04 loads every session (s04_sessions), 05 fits the PCA on S0, 06-08 are the defaults. Each online session
i = 1..n then runs the unchanged functions s09-s14 of lab/replay.py on a session view of the state (session_view):
offset, pool and test sets of session i, params0 = the model after session i-1, the PCA projections of the pool and
of the novel test set of session i. Only step 10 gets a wrapper: lut of session i = lut of session i-1 plus the map of
session i, as to_cluster of main.py. chain, new_state and the step functions of lab/replay.py are not changed, so the
snapshots of the GC10 chains do not go stale.

Commands (commit 2 of 3):
  exp.py run TAG --inputs F -- python lab/sessions.py run --features F --seed S [--dataset gc10|cub200]
                                               default chain (gc10: t1, cub200: t5), snapshots in runs/<run>/snap
  python lab/sessions.py c0 <gc10 v2 run>      C0: t1 through this runner equals every snapshot of a GC10 chain
  load(run, upto) for notebooks.
Variants (--from, --start, --at), rerun and the checks C1-C4 come in commit 3.

Snapshots: 04.pkl ... 08.pkl, then S<i>_<NN>.pkl (i = 1..n, NN = 09-14). 04 also holds the sets of every session.
params_f (S<i>_11) and params1 (S<i>_13) keep only the class slots that differ from params0 of the session (_pack);
load puts them back. Hash chain: h(04)..h(08) as replay.step_hashes, with s04_sessions at 04 and the runner of this
file in h(04); h(S<i>_k) = sha256(h(previous step), "S<i>", code of step k) (session_hashes).
Common random numbers: rng_in is kept at every step; a later variant sets the global RNG to the parent's at 04, 05 and
every S<i>_09 (commit 3).

Log lines: s10_naming prints "hungarian stage 1:" whatever the session; the log of session i gets "hungarian stage i:",
as main.py prints it (decision of 2026-11-15). The pass prefixes "stage_1_Flearning" and "stage_1_Slearning" go to the
TensorBoard writer only, not to the console, so nothing else is rewritten.
"""
import argparse, contextlib, copy, hashlib, io, json, os, pickle, sys, time, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "lab")):
    if p not in sys.path:
        sys.path.insert(0, p)
from replay import (GC10, STEPS, DEFAULT, OWNS, FILES, make_args, new_state, chain, snapshots, _fresh, _compare,   # noqa: E402
                    _Tee, _fn, _closure, _file_hash, _base, _run_dir, _git, _metrics)
from variants import CUB_T5                                                      # noqa: E402

PRESETS = {"gc10": GC10, "cub200": CUB_T5}
LOAD_MODE = {"gc10": "t1", "cub200": "t5"}
N_SESSIONS = {"t1": 1, "t5": 5}
SESSION_KEYS = ("pool_x", "true_pool_y", "test1_x", "true_test1_y", "test_old_x", "true_test_old_y",
                "test_all_x", "true_test_all_y")
SESSION_OUT = ("ids", "cluster", "inertia", "pi_test", "lut", "params_f", "keep", "params1", "acc1")   # 09-14
S0_STEPS, S_STEPS = STEPS[:5], STEPS[5:]                                        # 04-08, 09-14
PACKED = {"11": "params_f", "13": "params1"}
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


def _step(st, k, live=True):
    """Default step k on st through chain, one step; returns (rng_in, log). live=False: the log is not printed here."""
    rng_in, buf = np.random.get_state(), io.StringIO()
    tee = _Tee(sys.stdout) if live else None
    with contextlib.redirect_stdout(tee if live else buf):
        chain(st, start=k, upto=k)
    return rng_in, (tee.buf if live else buf).getvalue()


def run_s0(st, save=None):
    """Seeds in the order of main.py (as chain does at 04), 04 (s04_sessions), then the default steps 05-08 on st.
    save(name, out, rng_in, log) is called after each step when given. Returns the list of sessions."""
    import torch, jax
    a = st["args"]
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    jax.random.PRNGKey(a.seed)
    rng_in, tee = np.random.get_state(), _Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        out, sess = s04_sessions(st)
    st.update(out)
    if save:
        save("04", {**out, "sessions": sess}, rng_in, tee.buf.getvalue())
    for k in S0_STEPS[1:]:
        rng_in, log = _step(st, k)
        if save:
            save(k, {x: st[x] for x in OWNS[k]}, rng_in, log)
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


def run_session(st, sess, i, prev, save=None):
    """Session i: s09-s14 of lab/replay.py on session_view, one step at a time; after 10 the lut is made cumulative
    and the log line of 10 gets "hungarian stage i:". save as in run_s0, with names S<i>_<NN>.
    Returns (view, log of the session)."""
    a = st["args"]
    view = session_view(st, sess, i, prev)
    off, logs = view["offset"], []
    for k in S_STEPS:
        rng_in, log = _step(view, k, live=k != "10")
        if k == "10":
            lut = prev["lut"].copy() if prev is not None else np.arange(a.num_classes)
            for c, t in view["pi_test"].items():
                lut[off + t] = off + c
            view["lut"] = lut
            assert log.count(STAGE_LINE) == 1, f"session {i}: {log.count(STAGE_LINE)} lines '{STAGE_LINE}' (expected 1)"
            log = log.replace(STAGE_LINE, f"hungarian stage {i}:")
            sys.stdout.write(log)
        logs.append(log)
        if save:
            out = {x: view[x] for x in OWNS[k]}
            if k in PACKED:
                out[PACKED[k]] = _pack(out[PACKED[k]], view["params0"], a.num_classes)
            save(f"S{i}_{k}", out, rng_in, log)
    return view, "".join(logs)


def run_all(st, save=None):
    """S0 and every session in memory. Returns (sessions, [view of session i], [log of session i])."""
    sess = run_s0(st, save)
    views, logs, prev = [], [], None
    for i in range(1, len(sess) + 1):
        prev, log = run_session(st, sess, i, prev, save)
        views.append(prev)
        logs.append(log)
    return sess, views, logs


# ---------------------------------------------------------------- snapshots: packing, hashes, load
def _pack(p, ref, n_classes):
    """Arrays of p whose first axis is the class slot keep only the slots that differ from ref; the rest whole.
    Checked on the spot: _unpack(_pack(p)) equals p to the bit."""
    out = {}
    for key, v in p.items():
        v, r = np.asarray(v), np.asarray(ref[key])
        if v.ndim >= 1 and v.shape == r.shape and v.dtype == r.dtype and v.shape[0] == n_classes:
            slots = np.flatnonzero((v.reshape(len(v), -1) != r.reshape(len(r), -1)).any(axis=1))
            out[key] = {"__slots__": slots, "values": v[slots].copy()}
        else:
            out[key] = v
    back = _unpack(out, ref)
    assert all(np.array_equal(np.asarray(back[key]), np.asarray(p[key]), equal_nan=True) for key in p), "_pack is not exact"
    return out


def _unpack(packed, ref):
    out = {}
    for key, v in packed.items():
        if isinstance(v, dict) and "__slots__" in v:
            a = np.array(ref[key], copy=True)
            a[v["__slots__"]] = v["values"]
            out[key] = a
        else:
            out[key] = v
    return out


def _sha(parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def session_hashes(args, n):
    """{name: hash} for 04..08 and S<i>_09..S<i>_14, i = 1..n (QD28). h(04) covers the inputs of replay._base, s04_sessions
    and the runner (chain, new_state and the functions of this file that run the steps)."""
    runner = [_closure(f) for f in (chain, new_state, new_state_sessions, run_s0, session_view, run_session, _pack)]
    out, prev = {}, ""
    for k in S0_STEPS:
        items = [("s04_sessions", _closure(s04_sessions))] if k == "04" else [(DEFAULT[k], _closure(_fn(DEFAULT[k])))]
        parts = {"prev": prev, "items": items, "files": {f: _file_hash(f) for f in FILES.get(k, ())}}
        if k == "04":
            parts.update(base=_base(args), runner=runner)
        prev = out[k] = _sha(parts)
    for i in range(1, n + 1):
        for k in S_STEPS:
            parts = {"prev": prev, "session": f"S{i}", "items": [(DEFAULT[k], _closure(_fn(DEFAULT[k])))],
                     "files": {f: _file_hash(f) for f in FILES.get(k, ())}}
            prev = out[f"S{i}_{k}"] = _sha(parts)
    return out


def snapshots_sessions(run):
    """{name: snapshot} of a run of this file."""
    d = _run_dir(run) / "snap"
    blobs = {p.stem: pickle.loads(p.read_bytes()) for p in sorted(d.glob("*.pkl"))}
    assert blobs and "04" in blobs and blobs["04"]["meta"].get("runner") == "sessions", f"{d}: not snapshots of lab/sessions.py"
    return blobs


def fresh_sessions(blobs):
    """Refuse snapshots made from a dirty tree, and stale ones (hash chain recomputed from the code now)."""
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    now = session_hashes(args, N_SESSIONS[args.load_mode])
    for name, h in now.items():
        b = blobs.get(name)
        if b is None:
            continue
        if b["meta"].get("dirty"):
            raise SystemExit(f"sessions: snapshot {name} was made from a dirty tree")
        if b["hash"] != h:
            raise SystemExit(f"sessions: snapshots stale from {name} (code, features or packages changed)")


def load(run, upto=None, check=True):
    """(st after 08, [view of session i]) of a run, up to snapshot `upto` (e.g. "08", "S3_12"; None = all), with a
    fresh workbench. params_f and params1 are unpacked against params0 of their session."""
    blobs = snapshots_sessions(run)
    if check:
        fresh_sessions(blobs)
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    names = list(S0_STEPS) + [f"S{i}_{k}" for i in range(1, N_SESSIONS[args.load_mode] + 1) for k in S_STEPS]
    names = names[:names.index(upto) + 1] if upto else [x for x in names if x in blobs]
    st, sess, views, prev = new_state_sessions(args), None, [], None
    for name in names:
        out = dict(blobs[name]["out"])
        if name == "04":
            sess = out.pop("sessions")
        if name in S0_STEPS:
            st.update(out)
            continue
        i, k = int(name[1:name.index("_")]), name[-2:]
        if k == "09":
            prev = views[-1] if views else None
            views.append(session_view(st, sess, i, prev))
        if k in PACKED:
            out[PACKED[k]] = _unpack(out[PACKED[k]], views[-1]["params0"])
        views[-1].update(out)
    return st, views


# ---------------------------------------------------------------- commands
def _device(blobs=None):
    import jax
    device = jax.default_backend()
    if device != "gpu" and os.environ.get("REPLAY_ALLOW_CPU") != "1":
        raise SystemExit(f"sessions: JAX runs on {device}, not on the GPU (A1); REPLAY_ALLOW_CPU=1 is for tests only")
    if blobs is not None and {b["meta"].get("device", "gpu") for b in blobs.values()} != {device}:
        raise SystemExit("sessions: the snapshots ran on another device")
    return device


def cmd_run(argv):
    """Default chain through scripts/exp.py: python lab/sessions.py run --features F --seed S [--dataset gc10|cub200].
    load_mode t1 for gc10 (one session), t5 for cub200. Snapshots in runs/<run>/snap; metrics.json from console.log
    with scripts/extract_vbcgcd.py (replay._metrics)."""
    p = argparse.ArgumentParser(prog="sessions.py run")
    p.add_argument("--features", required=True, help="features/<run_id>/<model>")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", default="cub200", choices=sorted(PRESETS))
    o = p.parse_args(argv)
    if not os.environ.get("EXP_RUN_DIR"):
        sys.exit("sessions run: run it through scripts/exp.py (the run folder holds the snapshots)")
    assert Path("main.py").exists(), "run from the repo root (relative paths: splits/, features/)"
    run_dir, git = Path(os.environ["EXP_RUN_DIR"]), _git()
    if git["dirty"]:
        sys.exit("sessions run: commit the code first (same dirty rule as scripts/exp.py)")
    args = make_args(o.features, o.seed, PRESETS[o.dataset], load_mode=LOAD_MODE[o.dataset])
    st = new_state_sessions(args)
    device = _device()
    n = N_SESSIONS[args.load_mode]
    hashes = session_hashes(args, n)
    meta = dict(git, run_id=run_dir.name, runner="sessions", device=device, args=vars(args), n_sessions=n)
    snap = run_dir / "snap"
    snap.mkdir()
    sizes = {}

    def save(name, out, rng_in, log):
        blob = dict(step=name, out=out, rng_in=rng_in, log=log, hash=hashes[name], meta=meta)
        data = pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL)
        (snap / f"{name}.pkl").write_bytes(data)
        sizes[name] = len(data)

    print(f"sessions: {run_dir.name} | commit {git['commit'][:7]} | {o.dataset} {args.load_mode} | seed {o.seed} | "
          f"device {device}")
    t0 = time.time()
    run_all(st, save)
    print(f"sessions: {len(sizes)} snapshots, {sum(sizes.values()) / 2 ** 20:.1f} MiB, largest "
          f"{max(sizes, key=sizes.get)} {max(sizes.values()) / 2 ** 20:.1f} MiB; {time.time() - t0:.0f} s")
    _metrics(run_dir)


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
        src = st if k in S0_STEPS else view
        for key, ref in blobs[k]["out"].items():
            kind, d, ok = _compare(src[key], ref, 1e-6) if key in src else ("missing", float("nan"), False)
            rows.append((k, key, kind, d, ok))
    for key in ("z_pool", "z_test1"):
        kind, d, ok = _compare(view[key], blobs["05"]["out"][key], 0.0)
        rows.append(("view", key, kind, d, ok))
    print(f"\nC0 {o.run} | device {device} | {time.time() - t0:.0f} s")
    for k, key, kind, d, ok in rows:
        print(f"  {'OK  ' if ok else 'DIFF'} {k:4s} {key:16s} {kind:22s} max|diff| {d:.3g}")
    same_log = logs[0] == "".join(blobs[k]["log"] for k in S_STEPS)
    print(f"  log of 09-14 identical to the snapshots (report only): {same_log}")
    ok = all(r[4] for r in rows)
    print(f"C0 {'OK' if ok else 'DIFF'}: {sum(r[4] for r in rows)}/{len(rows)} keys")
    return ok


COMMANDS = {"run": cmd_run, "c0": cmd_c0}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python lab/sessions.py {{{','.join(COMMANDS)}}} ...")
    res = COMMANDS[sys.argv[1]](sys.argv[2:])
    sys.exit(0 if res is None or res else 1)
