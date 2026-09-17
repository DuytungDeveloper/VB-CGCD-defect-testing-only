#!/usr/bin/env python
"""Soi input/output cua tung module VB-CGCD.

Usage (chay tu thu muc goc cua repo):
    python scripts/probe.py cluster     # chay duoc ngay, khong can du lieu
    python scripts/probe.py features    # sau khi trich dac trung
    python scripts/probe.py loader      # sau khi trich dac trung
    python scripts/probe.py ckpt        # sau phase 2; them runs/<run> de chon run (mac dinh: moi nhat)
    python scripts/probe.py all | tee /tmp/probe_$(date +%Y%m%d).txt
"""
import sys, glob, os, re
import numpy as np

# repo root, de import clustering/ va dataloaders/ khi chay "python scripts/probe.py"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _feature_dir():
    """Dong dau cua runs/inputs.txt, vd features/<run_id>/dino-vitb16-sl (giong run_phase2_cl.sh)."""
    try:
        with open("runs/inputs.txt") as f:
            lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    except OSError:
        lines = []
    return lines[0] if lines else "features/MISSING/dino-vitb16-sl"


FEAT = _feature_dir()
MODEL = os.path.basename(FEAT)   # dino-vitb16-sl
DATA = os.path.dirname(FEAT)     # features/<run_id>


def hdr(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def arr(name, a):
    a = np.asarray(a)
    print(f"  {name:24s} shape={str(a.shape):18s} dtype={a.dtype}")
    if a.size and np.issubdtype(a.dtype, np.number):
        print(f"  {'':24s} min={a.min():.4f} max={a.max():.4f} "
              f"mean={a.mean():.4f} std={a.std():.4f}")


def label_stats(y, indent=2):
    """In thong ke nhan, an toan voi mang rong."""
    p = " " * indent
    y = np.asarray(y)
    if y.size == 0:
        print(f"{p}(rong — day la stage 0 cua test_old, dung nhu thiet ke)")
        return
    u, c = np.unique(y.astype(int), return_counts=True)
    print(f"{p}n_class={len(u):3d}  id=[{u.min()}..{u.max()}]  "
          f"n_sample={y.size:6d}  mau/lop min={c.min()} max={c.max()}")


# ---------------------------------------------------------------- features
def probe_features(d=DATA):
    hdr(f"MODULE: feature_extractor  ->  {d}/{MODEL}/")

    merged = sorted(glob.glob(f"{d}/{MODEL}/*.npy"))
    if not merged:
        print(f"  (trong) — chua chay dino-cifar100.py --finetuned?")
        return

    # 4 file da merge — day moi la thu dataloader doc
    expected = {
        f"features-{MODEL}.npy":      (50000, 768),
        f"labels-{MODEL}.npy":        (50000,),
        f"test_features-{MODEL}.npy": (10000, 768),
        f"test_labels-{MODEL}.npy":   (10000,),
    }
    for f in merged:
        base = os.path.basename(f)
        a = np.load(f)
        exp = expected.get(base)
        ok = "OK " if (exp is None or a.shape == exp) else "SAI"
        print(f"\n[{base}]  {os.path.getsize(f) / 1e6:.1f} MB  "
              f"{ok}{'' if exp is None else f' (ky vong {exp})'}")
        arr("array", a)
        if a.ndim == 1:
            label_stats(a)
        else:
            # Dac trung lay tu .pooler_output = tanh(...) -> phai nam trong [-1, 1].
            # Neu vuot ra ngoai, ban dang lay nham tang dau ra.
            lo, hi = float(a.min()), float(a.max())
            flag = "OK" if -1.001 <= lo and hi <= 1.001 else "CANH BAO: ngoai [-1,1]"
            print(f"  {'tanh-bound check':24s} [{lo:.4f}, {hi:.4f}]  {flag}")
            print(f"  {'NaN / Inf':24s} {np.isnan(a).sum()} / {np.isinf(a).sum()}")

    # Thu muc shard trung gian: chi dem, khong in tung file
    for sub in ("dino_vitb16_features", "dino_vitb16_labels",
                "dino_vitb16_test_features", "dino_vitb16_test_labels"):
        n = len(glob.glob(f"{d}/{sub}/*.npy"))
        if n:
            print(f"\n[{sub}/]  {n} shard (check_phase1 dem so file nay, dung xoa)")


# ------------------------------------------------------------------ loader
def _iter_stages(ld):
    """Chuan hoa: yield (x, y) tu map-of-DataPoint HOAC continuum scenario."""
    for stage in ld:
        if hasattr(stage, "_x"):          # DataPoint hoac continuum TaskSet
            yield stage._x, stage._y
        elif isinstance(stage, tuple):
            yield stage[0], stage[1]
        else:
            yield None, None


def probe_loader():
    hdr("MODULE: dataloaders.cifar100.CIFAR100Loader.makeT5Loader()")
    from types import SimpleNamespace
    from dataloaders.cifar100 import CIFAR100Loader

    args = SimpleNamespace(data_dir=DATA, pretrained_model_name=MODEL)
    train, test_novel, test_old, test_all = CIFAR100Loader(args).makeT5Loader()

    print("\nLUU Y: train/test_old/test_all la `map` object (dataloader.py:206,233,236)")
    print("       -> khong co len(), khong index duoc, CHI DUYET DUOC MOT LAN.")
    print("       test_novel la continuum ClassIncremental -> co len().")
    print("       Phan tu la DataPoint voi ._x / ._y, khong phai tuple.\n")

    for nm, ld in [("train", train), ("test_novel", test_novel),
                   ("test_old", test_old), ("test_all", test_all)]:
        n = f", len={len(ld)}" if hasattr(ld, "__len__") else ""
        print(f"\n[{nm}]  type={type(ld).__name__}{n}")
        for i, (x, y) in enumerate(_iter_stages(ld)):
            if x is None:
                print(f"  stage {i}: (khong doc duoc)")
                continue
            x = np.asarray(x)
            print(f"  stage {i}: X{str(x.shape):16s}", end="  ")
            label_stats(y, indent=0)

    print("\nDoi chieu Bang 8 (paper): 50 lop base x 400 mau = 20000 o stage 0;")
    print("moi stage online: 10 lop moi x 400 + 25 mau/lop cho MOI lop da gap.")
    print("  -> stage 1 ky vong 4000 + 25*50 = 5250 mau")
    print("  -> stage 5 ky vong 4000 + 25*90 = 6250 mau")


# ----------------------------------------------------------------- cluster
def probe_cluster():
    hdr("MODULE: clustering.gmm.GMMCluster")
    from clustering.gmm import GMMCluster

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(m, 1, (200, 384)) for m in (0, 3, 6)])
    y = np.repeat([0, 1, 2], 200)

    print("\nINPUT (du lieu tong hop, 3 cum tach bach)")
    arr("features", X)
    arr("labels", y)

    g = GMMCluster(num_classes=3, label_offset=0, random_state=0)
    g.fit(X)
    raw = g._pre_predict(X)
    algn = g.predict(X, y, with_known=False)

    print("\nOUTPUT")
    arr("_pre_predict (cluster id)", raw)
    arr("predict (sau Hungarian)", algn)
    print(f"  {'acc sau align':24s} {(algn == y).mean():.4f}")
    print(f"\n  GaussianMixture: means{g.model.means_.shape} "
          f"covariances{g.model.covariances_.shape} "
          f"covariance_type={g.model.covariance_type}")

    print("\n  GHI CHU 1: paper Phu luc C.3 noi dung k-means. Code dung GMM")
    print("             (clustering/gmm.py:14). Repo khong co k-means.")
    print("  GHI CHU 2: main.py:136 tao GMMCluster KHONG truyen random_state")
    print("             -> None -> rut tu global np.random (da seed o main.py:104).")
    print("             Probe nay dung random_state=0 nen KHONG giong hanh vi that.")
    print("  GHI CHU 3: predict() nhan NHAN THAT de chay Hungarian")
    print("             (clustering_base.py:54,59). Tren luong that su khong nhan,")
    print("             buoc nay khong ton tai.")


# -------------------------------------------------------------------- ckpt
def _stage_key(p):
    m = re.search(r"stage(\d+)", os.path.basename(p))
    return (int(m.group(1)) if m else 99, os.path.basename(p))


def probe_ckpt():
    hdr("MODULE: classifier.mngmm  ->  checkpoints")
    runs = sorted(glob.glob("runs/*/artifacts/saved_models"))
    if len(sys.argv) > 2:
        d = sys.argv[2].rstrip("/") + "/artifacts/saved_models"
    else:
        d = runs[-1] if runs else ""
    print(f"  run: {d}")
    fs = sorted(glob.glob(f"{d}/*.npy"), key=_stage_key) if d else []
    if not fs:
        print("  (trong) — chua chay phase 2, hoac thieu symlink runs/<run>/artifacts?")
        return

    print("  LUU Y: ten file dinh lien 'stage0class_covariances.npy' —")
    print("         mngmm.py:298-299 thieu dau '/' trong f-string.\n")

    for f in fs:
        a = np.load(f, mmap_mode="r")
        print(f"[{os.path.relpath(f)}]  {os.path.getsize(f) / 1e6:.1f} MB")
        arr("array", np.asarray(a[:1]))  # tranh doc het 59 MB de lay dtype
        print(f"  {'full shape':24s} {a.shape}")

        if a.ndim == 3:                    # class_covariances
            d = a.shape[1]
            print(f"  {'per-class stats':24s} "
                  f"(trace/d = 1.0000 nghia la Sigma VAN o khoi tao I)")
            for k in (0, 1, 49, 50, 59):
                if k >= a.shape[0]:
                    continue
                S = np.asarray(a[k])
                ev = np.linalg.eigvalsh(S)
                sign, logdet = np.linalg.slogdet(S)
                print(f"    class {k:3d} | trace/d={np.trace(S) / d:8.4f} "
                      f"| logdet={logdet:10.3f} (sign={sign:+.0f}) "
                      f"| ||S-I||_F={np.linalg.norm(S - np.eye(d)):8.4f} "
                      f"| eig=[{ev.min():.4f}, {ev.max():.4f}] "
                      f"| neg_eig={(ev < 0).sum()}")
            print(f"  -> neg_eig > 0 nghia la Sigma MAT tinh xac dinh duong.")
            print(f"     class_covs la numpyro.param KHONG rang buoc; mngmm.py:160")
            print(f"     co nhanh bat NaN, dau hieu tac gia tung gap su co nay.")
        elif a.ndim == 2:                  # class_means
            M = np.asarray(a)
            nz = np.linalg.norm(M, axis=1)
            print(f"  {'||mu_k|| per class':24s} min={nz.min():.5f} "
                  f"max={nz.max():.5f} mean={nz.mean():.5f}")
            print(f"  {'so lop co mu == 0':24s} {(nz == 0).sum()} / {M.shape[0]}"
                  f"   (lop chua tung gap van o khoi tao 0)")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fn = dict(features=probe_features, loader=probe_loader,
              cluster=probe_cluster, ckpt=probe_ckpt)
    if cmd == "all":
        for f in fn.values():
            try:
                f()
            except Exception as e:
                print(f"\n  LOI: {type(e).__name__}: {e}")
    elif cmd in fn:
        fn[cmd]()
    else:
        print(__doc__)
        sys.exit(1)