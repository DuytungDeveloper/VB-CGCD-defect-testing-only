"""Multi-session dissection bench (QD28; lab/README.vi.md section 5): the steps of lab/replay.py run session by session.

S0 runs once: 04 loads every session (s04_sessions), 05 fits the PCA on S0, 06-08 are the defaults. Each online session
i = 1..n then runs the unchanged functions s09-s14 of lab/replay.py on a session view of the state (session_view):
offset, pool and test sets of session i, params0 = the model after session i-1, the PCA projections of the pool and
of the novel test set of session i. Only step 10 gets a wrapper: lut of session i = lut of session i-1 plus the map of
session i, as to_cluster of main.py. chain, new_state and the step functions of lab/replay.py are not changed, so the
snapshots of the GC10 chains do not go stale.

Commands (commit 3a; QD28 amended 2026-11-15):
  exp.py run TAG --inputs F -- python lab/sessions.py run --features F --seed S [--dataset gc10|cub200]
                                               default chain (gc10: t1, cub200: t5), snapshots in runs/<run>/snap
  python lab/sessions.py c0 <gc10 v2 run>      C0 on one chain: t1 through this runner equals every snapshot
  exp.py run TAG -- python lab/sessions.py check-gc10     C0 on the 15 chains gc10_v2_ft_r<r>_s<N>, metrics.json
  python lab/sessions.py check <chain> <main.py run>       C1 (and C5) for one chain of this file
  python lab/sessions.py check-hash <chain>                C4: code change of one step for every session
  load(run, upto), rerun(run, name) for notebooks.
Variants (--from, --start, --at), C2, C3 come in commit 3b, after the E09.08 design.
The functions in the hash chain (new_state_sessions, s04_sessions, _step, run_s0, session_view, run_session, _pack,
_unpack and the constants they read) are not changed after the first CUB chain (2026-11-15): changing them makes every
snapshot of this file stale.

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


def session_hashes(args, n, override=None):
    """{name: hash} for 04..08 and S<i>_09..S<i>_14, i = 1..n (QD28). h(04) covers the inputs of replay._base, s04_sessions
    and the runner (chain, new_state and the functions of this file that run the steps). override: replay._closure
    override for the step functions (check-hash only)."""
    runner = [_closure(f) for f in (chain, new_state, new_state_sessions, run_s0, session_view, run_session, _pack)]
    out, prev = {}, ""
    for k in S0_STEPS:
        items = [("s04_sessions", _closure(s04_sessions))] if k == "04" else [(DEFAULT[k], _closure(_fn(DEFAULT[k]), override))]
        parts = {"prev": prev, "items": items, "files": {f: _file_hash(f) for f in FILES.get(k, ())}}
        if k == "04":
            parts.update(base=_base(args), runner=runner)
        prev = out[k] = _sha(parts)
    for i in range(1, n + 1):
        for k in S_STEPS:
            parts = {"prev": prev, "session": f"S{i}", "items": [(DEFAULT[k], _closure(_fn(DEFAULT[k]), override))],
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
    fresh workbench. params_f and params1 are unpacked against params0 of their session. st["sessions"] holds the
    sets of every session (snapshot 04)."""
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
    st["sessions"] = sess
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
    """C0 (QD28) on one chain: python lab/sessions.py c0 <gc10 v2 run>. Writes nothing."""
    p = argparse.ArgumentParser(prog="sessions.py c0")
    p.add_argument("run", help="run id or tag of a GC10 chain made by lab/replay.py run")
    o = p.parse_args(argv)
    return c0(o.run)


def c0(run):
    """C0: runs the args of the chain (t1) through this runner in memory and compares every output key of every
    snapshot 04-14 with it, by the rule of replay.rerun (discrete equal, floats within 1e-6 x max|value|, PCA and
    cluster models by _same). Also: z_pool, z_test1 of the session view equal to the bit those of snapshot 05; the log
    of 09-14 (report only). Returns True if every key passes."""
    assert Path("main.py").exists(), "run from the repo root (relative paths: splits/, features/)"
    blobs = snapshots(run)
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
    print(f"\nC0 {run} | device {device} | {time.time() - t0:.0f} s")
    for k, key, kind, d, ok in rows:
        print(f"  {'OK  ' if ok else 'DIFF'} {k:4s} {key:16s} {kind:22s} max|diff| {d:.3g}")
    same_log = logs[0] == "".join(blobs[k]["log"] for k in S_STEPS)
    print(f"  log of 09-14 identical to the snapshots (report only): {same_log}")
    ok = all(r[4] for r in rows)
    print(f"C0 {'OK' if ok else 'DIFF'}: {sum(r[4] for r in rows)}/{len(rows)} keys")
    return ok


def cmd_check_gc10(argv):
    """C0 on the 15 chains gc10_v2_ft_r<r>_s<N> (r0-r2, s0-s4), through scripts/exp.py:
    python lab/sessions.py check-gc10. metrics.json: c0_r<r>_s<N> (1 = every key passes), c0_ok (all 15)."""
    if not os.environ.get("EXP_RUN_DIR"):
        sys.exit("sessions check-gc10: run it through scripts/exp.py (metrics.json goes to the run folder)")
    if _git()["dirty"]:
        sys.exit("sessions check-gc10: commit the code first (same dirty rule as scripts/exp.py)")
    out = {}
    for r in range(3):
        for N in range(5):
            out[f"c0_r{r}_s{N}"] = int(c0(f"gc10_v2_ft_r{r}_s{N}"))
    out["c0_ok"] = int(all(v == 1 for v in out.values()))
    print("\n" + "\n".join(f"{k:12s} {v}" for k, v in out.items()))
    (Path(os.environ["EXP_RUN_DIR"]) / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    return out["c0_ok"] == 1


def cmd_check(argv):
    """C1 and C5 (QD28) for one chain of this file against its main.py baseline (deterministic mode):
    python lab/sessions.py check <chain> <main.py run>. Writes nothing.
    C1, for every session i: metrics.json of the two runs equal (tables S0-S5, cluster sizes, kept, stop steps, M_d,
    M_f); pi_test, the pool and test contingency tables and the confusion matrix equal the "hungarian / xtab /
    confusion stage i" lines of the baseline log; inertia within 1e-6 (relative); params1 of session i, and params0,
    within 1e-6 x max|value| of the stage<i> arrays the baseline saved. C5: the class slots kept in S<i>_11 and S<i>_13
    (_pack: the slots that differ from params0 of the session) all lie in [offset, offset + increment), so the slots of
    earlier classes keep their values to the bit and the slots not seen yet keep their S0 values."""
    import re, ast
    from sklearn.metrics.cluster import contingency_matrix
    from replay import _confusion
    p = argparse.ArgumentParser(prog="sessions.py check")
    p.add_argument("chain", help="run id or tag of a chain of lab/sessions.py run")
    p.add_argument("ref", help="run id or tag of the main.py baseline")
    o = p.parse_args(argv)
    assert Path("main.py").exists(), "run from the repo root (relative paths: splits/, features/)"
    rd, fd = _run_dir(o.chain), _run_dir(o.ref)
    blobs = snapshots_sessions(o.chain)
    fresh_sessions(blobs)
    st, views = load(o.chain, check=False)          # new_state before jax is imported (XLA_FLAGS)
    _device(blobs)
    a = st["args"]
    m, mr = json.loads((rd / "metrics.json").read_text()), json.loads((fd / "metrics.json").read_text())
    log = (fd / "console.log").read_text(errors="replace") if (fd / "console.log").exists() else ""
    find = lambda pat: (lambda r: r.groups() if r else None)(re.search(pat, log))
    sm = fd / "artifacts" / "saved_models"                     # where replay.py check looks (GC10)
    if not sm.is_dir():                                      # main.py: logs/<trail_name>_<time>/saved_models
        found = sorted((ROOT / "logs").glob(f"{fd.name}_*/saved_models"))
        assert len(found) <= 1, f"{len(found)} folders logs/{fd.name}_*/saved_models"
        sm = found[0] if found else sm
    print(f"  saved models of the baseline: {sm if sm.is_dir() else 'not found'}")

    def close(arr, f):
        ref = np.load(f) if f.exists() else None
        return bool(np.abs(np.asarray(arr).astype(np.float64) - ref).max() <= 1e-6 * np.abs(ref).max()) if ref is not None else None

    res = [("metrics.json (all sessions)", m == mr)]
    for f, k in (("class_means", "class_means"), ("class_covariances", "class_covs")):
        res.append((f"S0 params0 {k}", close(st["params0"][k], sm / f"stage0{f}.npy")))
    for i, v in enumerate(views, 1):
        off, inc = v["offset"], a.increment
        y, yt = np.asarray(v["true_pool_y"]).astype(int), np.asarray(v["true_test1_y"]).astype(int)
        pi = find(rf"hungarian stage {i}: mode=eval pi_test=(\{{[^}}]*\}})")
        res.append((f"S{i} pi_test", ast.literal_eval(pi[0]) == v["pi_test"] if pi else None))
        for name, yy, cc in (("train", y, np.asarray(v["ids"])), ("test", yt, np.asarray(v["cluster"].predict_raw(v["test1_x"])))):
            x = find(rf"xtab stage {i} {name}: rows=(\[[^\]]*\]) cols=(\[[^\]]*\]) (\[\[.*\]\])")
            res.append((f"S{i} contingency {name}", [np.unique(yy).tolist(), np.unique(cc).tolist(),
                        contingency_matrix(yy, cc).tolist()] == [ast.literal_eval(t) for t in x] if x else None))
        it = find(rf"xtab stage {i} objective: inertia=(\S+)")
        res.append((f"S{i} inertia (relative < 1e-6)", abs(v["inertia"] - float(it[0])) / float(it[0]) < 1e-6 if it else None))
        c = find(rf"confusion stage {i}: rows=true cols=pred (\[\[.*\]\])")
        res.append((f"S{i} confusion", _confusion(v).tolist() == ast.literal_eval(c[0]) if c else None))
        res.append((f"S{i} kept", int(np.asarray(v["keep"]).sum()) == mr.get(f"S{i}_novel_kept")))
        for f, k in (("class_means", "class_means"), ("class_covariances", "class_covs")):
            res.append((f"S{i} params1 {k}", close(v["params1"][k], sm / f"stage{i}{f}.npy")))
        slots = [s for nm in (f"S{i}_11", f"S{i}_13") for val in blobs[nm]["out"][PACKED[nm[-2:]]].values()
                 if isinstance(val, dict) for s in np.asarray(val["__slots__"]).tolist()]
        res.append((f"S{i} C5 changed slots in [{off}, {off + inc})", bool(slots) and all(off <= s < off + inc for s in slots)))
    for name, ok in res:
        print(f"  {'OK  ' if ok else 'n/a ' if ok is None else 'DIFF'} {name}")
    if m != mr:
        print("  metrics differ:", {k: (m.get(k), mr.get(k)) for k in set(m) | set(mr) if m.get(k) != mr.get(k)})
    verdict = "DIFF" if any(ok is False for _, ok in res) else "INCOMPLETE" if any(ok is None for _, ok in res) else "OK"
    print(f"C1 {verdict}: {rd.name} = {fd.name} ({sum(ok is True for _, ok in res)}/{len(res)})")
    return verdict == "OK"


def cmd_check_hash(argv):
    """C4 (QD28), the part for a code change of one step applied to every session (variants for one session: 3b):
    python lab/sessions.py check-hash <chain>. For steps 06, 09, 12, 14: a fake change of the code of the step (in
    memory, through the override of replay._closure) must leave every hash before it unchanged and change every hash
    from it on (from S1_k for a session step). A change of a docstring must change no hash. Writes nothing."""
    import inspect
    from replay import _src_hash
    p = argparse.ArgumentParser(prog="sessions.py check-hash")
    p.add_argument("chain", help="run id or tag of a chain of lab/sessions.py run")
    o = p.parse_args(argv)
    blobs = snapshots_sessions(o.chain)
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    n = N_SESSIONS[args.load_mode]
    h0 = session_hashes(args, n)
    names = list(h0)
    res = [("hashes now = hashes of the snapshots", all(blobs[x]["hash"] == h0[x] for x in names if x in blobs))]
    for k in ("06", "09", "12", "14"):
        key = f"replay.py:{DEFAULT[k]}"
        h1 = session_hashes(args, n, override={key: "fake"})
        first = names.index(k if k in S0_STEPS else f"S1_{k}")
        ok = all(h1[x] == h0[x] for x in names[:first]) and all(h1[x] != h0[x] for x in names[first:])
        res.append((f"fake change of {DEFAULT[k]}: unchanged before {names[first]}, changed from it on", ok))
    from replay import s12_filter
    src = inspect.getsource(s12_filter)
    doc = s12_filter.__doc__
    res.append(("docstring change of s12_filter: same source hash", _src_hash(src) == _src_hash(src.replace(doc, doc + " (changed)", 1))))
    for name, ok in res:
        print(f"  {'OK  ' if ok else 'DIFF'} {name}")
    ok = all(r[1] for r in res)
    print(f"C4 {'OK' if ok else 'DIFF'} (part for one step in every session; one-session variants in 3b)")
    return ok


def rerun(run, name, rtol=1e-6):
    """Notebooks: run snapshot `name` (e.g. "06", "S3_11") again from the snapshot before it, with the global RNG set
    to its rng_in, and compare each output key with it by the rule of replay.rerun (discrete equal, floats within
    rtol x max|value|, PCA and cluster models by _same). Step 04 runs from a fresh state with the seeds of main.py.
    The lut of a step 10 is made cumulative as in run_session. Writes nothing. Returns ({key: new value}, ok)."""
    blobs = snapshots_sessions(run)
    fresh_sessions(blobs)
    args = types.SimpleNamespace(**blobs["04"]["meta"]["args"])
    names = list(S0_STEPS) + [f"S{i}_{k}" for i in range(1, N_SESSIONS[args.load_mode] + 1) for k in S_STEPS]
    assert name in names, f"rerun: {name!r} is not one of {names}"
    ref = dict(blobs[name]["out"])
    if name == "04":
        st = new_state_sessions(args)
        import torch, jax
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        jax.random.PRNGKey(args.seed)
        out, sess = s04_sessions(st)
        new = {**out, "sessions": sess}
        _device(blobs)
    else:
        st, views = load(run, names[names.index(name) - 1], check=False)
        _device(blobs)                                           # after new_state (XLA_FLAGS)
        np.random.set_state(blobs[name]["rng_in"])
        if name in S0_STEPS:
            _step(st, name)
            new = {x: st[x] for x in OWNS[name]}
        else:
            i, k = int(name[1:name.index("_")]), name[-2:]
            if k == "09":
                views.append(session_view(st, st["sessions"], i, views[-1] if views else None))
            v = views[-1]
            _step(v, k, live=k != "10")
            if k == "10":                                            # as run_session (not shared: hash chain)
                lut = views[-2]["lut"].copy() if len(views) > 1 else np.arange(args.num_classes)
                for c, t in v["pi_test"].items():
                    lut[v["offset"] + t] = v["offset"] + c
                v["lut"] = lut
            new = {x: v[x] for x in OWNS[k]}
            if k in PACKED:
                ref[PACKED[k]] = _unpack(ref[PACKED[k]], v["params0"])
    rows = []
    for key, r in ref.items():
        kind, d, ok = _compare(new[key], r, rtol) if key != "sessions" else \
            ("sessions", float("nan"), all(_compare(x[q], y[q], 0.0)[2] for x, y in zip(new[key], r) for q in y))
        rows.append((key, kind, d, ok))
    for key, kind, d, ok in rows:
        print(f"  {'OK  ' if ok else 'DIFF'} {key:12s} {kind:22s} max|diff| {d:.3g}")
    ok = all(r[3] for r in rows)
    print(f"rerun {name}: {'OK' if ok else 'DIFF'} ({sum(r[3] for r in rows)}/{len(rows)} keys)")
    return new, ok


COMMANDS = {"run": cmd_run, "c0": cmd_c0, "check-gc10": cmd_check_gc10, "check": cmd_check, "check-hash": cmd_check_hash}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(f"usage: python lab/sessions.py {{{','.join(COMMANDS)}}} ...")
    res = COMMANDS[sys.argv[1]](sys.argv[2:])
    sys.exit(0 if res is None or res else 1)
