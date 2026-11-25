#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a8_autoroi_dydm_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a8_autoroi_dydm_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/gc10_a8_autoroi_dydm_telemetry.json}"
EPOCHS="${EPOCHS:-12}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"

TOTAL_T0=$(date +%s)
trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc, total_sec=$(($(date +%s) - TOTAL_T0))) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== START GC10 STEP A8 (A7a DINOV3 AUTO-ROI + IMBALANCED-DYDM, epochs=$EPOCHS) at $(date) ==="

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
    "model": "DINOv3 Auto-ROI Fusion + Imbalanced-DYDM",
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
        old = json.loads(p.read_text())
        meta["stages"] = old.get("stages", {})
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
  if [ -d "datasets/GC10-DET/1" ] && [ ! -L "datasets/GC10-DET" ]; then
    return 0
  fi
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
  echo "ERROR: Không tìm thấy GC10-DET tại /content/drive/MyDrive/Datasets/GC10-DET" >&2
  return 1
}

sync_a7a_from_drive() {
  mkdir -p features runs
  for d in "$DRIVE_REPO"/features/*__colab_gc10_d3_autoroi_ep20_r0; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "features/$b" ] || { echo ">>> Đồng bộ features A7a từ Drive: $b"; cp -r "$d" "features/$b"; }
  done
  for d in "$DRIVE_REPO"/runs/*__colab_gc10_d3_autoroi_ep20_r0_s*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "runs/$b" ] || { echo ">>> Đồng bộ snapshot A7a từ Drive: $b"; cp -r "$d" "runs/$b"; }
  done
}

link_gc10_dataset
sync_a7a_from_drive

FEAT_ROOT=$(ls -d features/*__colab_gc10_d3_autoroi_ep20_r0 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$FEAT_ROOT" ] || [ ! -f "$FEAT_ROOT/ft_ckpt.pt" ]; then
  echo "ERROR: Không tìm thấy features/*__colab_gc10_d3_autoroi_ep20_r0/ft_ckpt.pt (từ Bước A7a)!" >&2
  exit 2
fi
FEAT_DIR="$FEAT_ROOT/dinov3-vitb16-sl"
FT_CKPT="$FEAT_ROOT/ft_ckpt.pt"
echo ">>> Sử dụng FEAT_DIR=$FEAT_DIR và CKPT=$FT_CKPT"

IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
for s in "${SEEDS[@]}"; do
  t0=$(date +%s)
  hw_tmp="/tmp/hw_gc10_a8_s${s}.json"
  python colab/hw_monitor.py --out "$hw_tmp" &
  mon_pid=$!
  echo ">>> Chạy Bước A8 (A7a + Imbalanced-DYDM) cho seed s${s} ($(date))..."
  python colab/run_gc10_a8_autoroi_dydm.py \
    --prefix "colab_gc10_d3_autoroi_ep20_r0" \
    --feat-dir "$FEAT_DIR" \
    --ft-ckpt "$FT_CKPT" \
    --seeds "$s" \
    --epochs "$EPOCHS" \
    --out-json "$SUMMARY_JSON" \
    --meta-json "$META_JSON"
  kill -INT "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  record_stage "r0_s${s}" "$(($(date +%s) - t0))" "$hw_tmp"
done

{
  python - <<PY
import json
from pathlib import Path
d = json.loads(Path("$META_JSON").read_text())
stages = d.get("stages", {})
cpu_pk = max((v.get("cpu_peak_pct", 0) for v in stages.values()), default=0)
ram_pk = max((v.get("ram_peak_gb", 0) for v in stages.values()), default=0)
gpu_pk = max((v.get("gpu_util_peak_pct", 0) for v in stages.values()), default=0)
vram_pk = max((v.get("gpu_vram_peak_mb", 0) for v in stages.values()), default=0)
tot_sec = sum(v.get("sec", 0) for v in stages.values())
print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC A8 - GC10 A7a + IMBALANCED-DYDM) ===")
print(f"CPU          : {d.get('cpu_model')} ({d.get('cpu_cores')} cores) | Peak CPU: {cpu_pk}%")
print(f"RAM          : Peak {ram_pk} GB / {d.get('ram_total_gb')} GB")
print(f"GPU          : {d.get('gpu')} | Peak GPU Util: {gpu_pk}% | Peak GPU VRAM: {vram_pk} MB")
print(f"PyTorch      : {d.get('torch_version')} (CUDA={d.get('torch_cuda')}) | JAX {d.get('jax_version')} ({d.get('jax_devices')})")
print(f"Tổng thời gian các công đoạn: {tot_sec} s ({tot_sec/60:.1f} phút)")
print("Chi tiết từng công đoạn:")
for k, v in stages.items():
    print(f"  - {k:<10}: {v.get('sec',0):4d}s | CPU peak {v.get('cpu_peak_pct',0):5.1f}% (avg {v.get('cpu_avg_pct',0):4.1f}%) | RAM peak {v.get('ram_peak_gb',0):5.2f} GB | GPU peak {v.get('gpu_util_peak_pct',0):5.1f}% | VRAM peak {v.get('gpu_vram_peak_mb',0):6.1f} MB")
print()
s = json.loads(Path("$SUMMARY_JSON").read_text())
agg = s.get("arms", {})
n_bal = agg.get("N", {}).get("bal_mean", 0.0)
print("=== KẾT QUẢ BƯỚC A8 (GC10-DET DINOv3 Auto-ROI Fusion + Imbalanced-DYDM, ep=20, r0) ===")
print(f"{'Arm':<18} {'bal (mean±std)':>18} {'Old (%)':>10} {'All (%)':>10} {'c09 (%)':>10} {'dbal vs N':>12}")
print("-" * 84)
for arm in ("N", "NL", "A8_Imb_head", "A8_Imb_km", "A8_Imb_blend50", "A8_Imb_blend70", "A8_Imb_blend50_NL"):
    if arm not in agg:
        continue
    r = agg[arm]
    print(f"{arm:<18} {r['bal_mean']:7.4f} ± {r['bal_std']:6.4f} {r['Old_mean']:10.2f} {r['All_mean']:10.2f} {r['c09_mean']:10.2f} {r['bal_mean'] - n_bal:+12.4f}")
PY
} | tee "$SUMMARY_OUT"
