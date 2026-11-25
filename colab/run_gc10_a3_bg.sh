#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a3_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a3_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/gc10_a3_telemetry.json}"
DYDM_EPOCHS="${DYDM_EPOCHS:-12}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"

TOTAL_T0=$(date +%s)
trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc, total_sec=$(($(date +%s) - TOTAL_T0))) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== START GC10 STEP A3 DYDM (stage=$STAGE, epochs=$DYDM_EPOCHS) at $(date) ==="

python - <<PY
import json, os, platform, subprocess
from pathlib import Path

cpu_model = platform.processor() or "unknown"
try:
    for ln in Path("/proc/cpuinfo").read_text().splitlines():
        if "model name" in ln:
            cpu_model = ln.split(":", 1)[1].strip()
            break
except Exception:
    pass

ram_gb = 0.0
try:
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith("MemTotal:"):
            ram_gb = round(int(ln.split()[1]) / 1024 / 1024, 2)
            break
except Exception:
    pass

gpu_info = "NO_GPU"
try:
    gpu_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True
    ).strip()
except Exception:
    pass

import torch, jax
meta = {
    "cpu_model": cpu_model,
    "cpu_cores": os.cpu_count(),
    "ram_total_gb": ram_gb,
    "gpu": gpu_info,
    "torch_version": torch.__version__,
    "torch_cuda": torch.cuda.is_available(),
    "jax_version": jax.__version__,
    "jax_devices": [str(d) for d in jax.devices()],
    "stages": {},
}
p = Path("$META_JSON")
if p.exists():
    try:
        meta["stages"] = json.loads(p.read_text()).get("stages", {})
    except Exception:
        pass
p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"HARDWARE | CPU: {cpu_model} ({os.cpu_count()} cores) | RAM: {ram_gb} GB | GPU: {gpu_info}")
print(f"FRAMEWORKS | Torch {torch.__version__} (CUDA={torch.cuda.is_available()}) | JAX {jax.__version__} ({jax.devices()})")
PY

record_stage() {
  local key="$1" sec="$2" hw_file="$3"
  python - <<PY
import json
from pathlib import Path
p = Path("$META_JSON")
d = json.loads(p.read_text()) if p.exists() else {"stages": {}}
hw = json.loads(Path("$hw_file").read_text()) if Path("$hw_file").exists() else {}
hw["sec"] = int("$sec")
d.setdefault("stages", {})["$key"] = hw
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
  rm -f "$hw_file"
}

link_gc10_dataset() {
  mkdir -p datasets
  if [ -d "/content/GC10-DET/1" ]; then
    rm -f datasets/GC10-DET
    ln -sfn "/content/GC10-DET" datasets/GC10-DET
    return 0
  fi
  for cand in "/content/drive/MyDrive/Datasets/GC10-DET" "$WORK_DIR/GC10-DET" "$WORK_DIR/Datasets/GC10-DET"; do
    if [ -d "$cand/1" ]; then
      echo ">>> Copy GC10-DET từ Drive sang /content/GC10-DET..."
      cp -r "$cand" /content/GC10-DET
      rm -f datasets/GC10-DET
      ln -sfn "/content/GC10-DET" datasets/GC10-DET
      return 0
    fi
  done
  echo "ERROR: Không tìm thấy GC10-DET trên Drive!" >&2
  return 1
}

sync_gc10_from_drive() {
  mkdir -p features runs
  for d in "$DRIVE_REPO"/features/*__colab_gc10_ep20_r0; do
    [ -d "$d/dino-vitb16-sl" ] || continue
    b=$(basename "$d")
    [ -d "features/$b" ] || { echo ">>> Đồng bộ features GC10 ep20 từ Drive: $b"; cp -r "$d" "features/$b"; }
  done
  for s in 0 1 2 3 4; do
    local tag="colab_gc10_ep20_r0_s${s}"
    local src
    src=$(ls -d "$DRIVE_REPO"/runs/*__"${tag}"/snap/S1_14.pkl 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$src" ]; then
      local sdir
      sdir=$(dirname "$(dirname "$src")")
      local b
      b=$(basename "$sdir")
      [ -d "runs/$b" ] || { echo ">>> Đồng bộ snapshot GC10 từ Drive: $b"; cp -r "$sdir" "runs/$b"; }
    fi
  done
}

link_gc10_dataset
sync_gc10_from_drive

FEAT_ROOT=$(ls -d features/*__colab_gc10_ep20_r0/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$FEAT_ROOT" ]; then
  echo "ERROR: Không tìm thấy features/*__colab_gc10_ep20_r0/dino-vitb16-sl trên Drive!" >&2
  exit 1
fi
CKPT_PATH="$(dirname "$FEAT_ROOT")/ft_ckpt.pt"
if [ ! -f "$CKPT_PATH" ]; then
  echo "ERROR: Không tìm thấy $CKPT_PATH trên Drive!" >&2
  exit 1
fi
echo ">>> Sử dụng FEAT_DIR=$FEAT_ROOT và CKPT=$CKPT_PATH"

run_seed() {
  local s="$1"
  local out_s="$WORK_DIR/gc10_a3_s${s}.json"
  if [ -f "$out_s" ]; then
    echo ">>> Đã có kết quả $out_s, bỏ qua seed s${s}."
    return 0
  fi
  echo ">>> Chạy DYDM Discovery Branch trên GC10 cho seed s${s} ($(date))..."
  local t0 hw_tmp mon_pid
  t0=$(date +%s)
  hw_tmp="/tmp/hw_gc10_a3_s${s}.json"
  python colab/hw_monitor.py --out "$hw_tmp" &
  mon_pid=$!
  python colab/run_gc10_a3_dydm.py \
    --prefix colab_gc10_ep20 \
    --feat-dir "$FEAT_ROOT" \
    --ckpt "$CKPT_PATH" \
    --seeds "$s" \
    --epochs "$DYDM_EPOCHS" \
    --out-json "$out_s"
  kill -INT "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  record_stage "r0_s${s}" "$(($(date +%s) - t0))" "$hw_tmp"
}

run_summary() {
  python - <<PY | tee "$SUMMARY_OUT"
import json, numpy as np
from pathlib import Path

work = Path("$WORK_DIR")
seeds = [int(x) for x in "$SEEDS_CSV".split(",") if x]
meta = json.loads(Path("$META_JSON").read_text()) if Path("$META_JSON").exists() else {}
stages = meta.get("stages", {})
cpu_pk = max((v.get("cpu_peak_pct", 0) for v in stages.values()), default=0)
ram_pk = max((v.get("ram_peak_gb", 0) for v in stages.values()), default=0)
gpu_pk = max((v.get("gpu_util_peak_pct", 0) for v in stages.values()), default=0)
vram_pk = max((v.get("gpu_vram_peak_mb", 0) for v in stages.values()), default=0)
tot_sec = sum(v.get("sec", 0) for v in stages.values())

print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC A3 - GC10 DYDM) ===")
print(f"CPU          : {meta.get('cpu_model')} ({meta.get('cpu_cores')} cores) | Peak CPU: {cpu_pk}%")
print(f"RAM          : Peak {ram_pk} GB / {meta.get('ram_total_gb')} GB")
print(f"GPU          : {meta.get('gpu')} | Peak GPU Util: {gpu_pk}% | Peak GPU VRAM: {vram_pk} MB")
print(f"PyTorch      : {meta.get('torch_version')} (CUDA={meta.get('torch_cuda')}) | JAX {meta.get('jax_version')} ({meta.get('jax_devices')})")
print(f"Tổng thời gian các công đoạn: {tot_sec} s ({tot_sec/60:.1f} phút)")
print("Chi tiết từng công đoạn:")
for k, v in stages.items():
    print(f"  - {k:<10}: {v.get('sec',0):4d}s | CPU peak {v.get('cpu_peak_pct',0):5.1f}% (avg {v.get('cpu_avg_pct',0):4.1f}%) | RAM peak {v.get('ram_peak_gb',0):5.2f} GB | GPU peak {v.get('gpu_util_peak_pct',0):5.1f}% | VRAM peak {v.get('gpu_vram_peak_mb',0):6.1f} MB")

per_seed = {}
for s in seeds:
    p = work / f"gc10_a3_s{s}.json"
    if p.exists():
        per_seed[f"s{s}"] = json.loads(p.read_text())["per_seed"][f"s{s}"]

arms = ("NL", "FTR1", "DYDM_head", "DYDM_km")
out = {"prefix": "colab_gc10_ep20", "epochs": int("$DYDM_EPOCHS"), "seeds": list(per_seed.keys()),
       "telemetry": meta, "arms": {}, "per_seed": per_seed}

print("\n=== KẾT QUẢ BƯỚC A3 (GC10-DET DYDM Discovery Branch + NIW Memory, ep=20, r0) ===")
print(f"{'Arm':<12} {'bal (mean±std)':>18} {'Old (%)':>10} {'All (%)':>10} {'c09 (%)':>10} {'dbal vs FTR1':>14}")
print("-" * 80)
if per_seed:
    ftr_bal = float(np.mean([per_seed[k]["FTR1"]["bal"] for k in per_seed]))
    for arm in arms:
        bals = [per_seed[k][arm]["bal"] for k in per_seed]
        olds = [per_seed[k][arm]["Old"] for k in per_seed]
        alls = [per_seed[k][arm]["All"] for k in per_seed]
        c09s = [per_seed[k][arm]["c09"] for k in per_seed]
        bm, bs = float(np.mean(bals)), float(np.std(bals))
        om, am, cm = float(np.mean(olds)), float(np.mean(alls)), float(np.mean(c09s))
        dbal = bm - ftr_bal
        out["arms"][arm] = {"bal_mean": bm, "bal_std": bs, "Old_mean": om, "All_mean": am, "c09_mean": cm, "dbal_vs_FTR1": dbal}
        print(f"{arm:<12} {bm:7.4f} ± {bs:6.4f} {om:10.2f} {am:10.2f} {cm:10.2f} {dbal:+14.4f}")

Path("$SUMMARY_JSON").write_text(json.dumps(out, indent=2, ensure_ascii=False))
PY
}

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
case "$STAGE" in
  auto)
    for s in "${SEEDS[@]}"; do
      run_seed "$s"
    done
    run_summary
    ;;
  s0|s1|s2|s3|s4)
    run_seed "${STAGE#s}"
    run_summary
    ;;
  summary)
    run_summary
    ;;
  *)
    echo "Stage không hợp lệ: $STAGE (chọn: auto | s0..s4 | summary)" >&2
    exit 2
    ;;
esac
