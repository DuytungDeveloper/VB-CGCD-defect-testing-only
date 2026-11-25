#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/cub_b2_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/cub_b2_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/cub_b2_telemetry.json}"
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

echo "=== START CUB-200 STEP B2 DYDM (stage=$STAGE, epochs=$DYDM_EPOCHS) at $(date) ==="

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

sync_cub_from_drive() {
  mkdir -p features runs
  for d in "$DRIVE_REPO"/features/*__colab_cub_ft_l25_r0; do
    [ -d "$d/dino-vitb16-sl" ] || continue
    b=$(basename "$d")
    [ -d "features/$b" ] || { echo ">>> Đồng bộ features CUB từ Drive: $b"; cp -r "$d" "features/$b"; }
  done
  for s in 0 1 2 3 4; do
    local tag="colab_cub_ft_l25_r0_s${s}"
    local src
    src=$(ls -d "$DRIVE_REPO"/runs/*__"${tag}"/snap/S5_14.pkl 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$src" ]; then
      local sdir
      sdir=$(dirname "$(dirname "$src")")
      local b
      b=$(basename "$sdir")
      [ -d "runs/$b" ] || { echo ">>> Đồng bộ snapshot CUB từ Drive: $b"; cp -r "$sdir" "runs/$b"; }
    fi
  done
}

sync_cub_from_drive

FEAT_ROOT=$(ls -d features/*__colab_cub_ft_l25_r0/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$FEAT_ROOT" ]; then
  echo "ERROR: Không tìm thấy features/*__colab_cub_ft_l25_r0/dino-vitb16-sl trên Drive!" >&2
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
  local out_s="$WORK_DIR/cub_b2_s${s}.json"
  if [ -f "$out_s" ]; then
    echo ">>> Đã có kết quả $out_s, bỏ qua seed s${s}."
    return 0
  fi
  echo ">>> Chạy DYDM Discovery Branch cho seed s${s} ($(date))..."
  local t0 hw_tmp mon_pid
  t0=$(date +%s)
  hw_tmp="/tmp/hw_cub_b2_s${s}.json"
  python colab/hw_monitor.py --out "$hw_tmp" &
  mon_pid=$!
  python colab/run_cub_b2_dydm.py \
    --prefix colab_cub_ft_l25 \
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

print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC B2 - DYDM) ===")
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
    p = work / f"cub_b2_s{s}.json"
    if p.exists():
        per_seed[f"s{s}"] = json.loads(p.read_text())["per_seed"][f"s{s}"]

arms = ("FTR1", "DYDM_head", "DYDM_km")
out = {"prefix": "colab_cub_ft_l25", "epochs": int("$DYDM_EPOCHS"), "seeds": list(per_seed.keys()),
       "telemetry": meta, "arms": {}, "per_seed": per_seed}

print("\n=== KẾT QUẢ BƯỚC B2 (CUB-200 DYDM Discovery Branch + NIW Memory, r0) ===")
print(f"{'Arm':<12} {'New (M_d)':>10} {'Old (%)':>10} {'S5_All (%)':>11} {'c09 (%)':>10} {'dM_d vs FTR1':>14} {'dOld vs FTR1':>14}")
print("-" * 86)
if per_seed:
    ftr_md = float(np.mean([np.mean(per_seed[k]["FTR1"]["New"]) for k in per_seed]))
    ftr_old = float(np.mean([np.mean(per_seed[k]["FTR1"]["Old"]) for k in per_seed]))
    for arm in arms:
        md = float(np.mean([np.mean(per_seed[k][arm]["New"]) for k in per_seed]))
        old = float(np.mean([np.mean(per_seed[k][arm]["Old"]) for k in per_seed]))
        s5 = float(np.mean([per_seed[k][arm]["All"][-1] for k in per_seed]))
        c09 = float(np.mean([np.mean(per_seed[k][arm]["c09"]) for k in per_seed]))
        dmd, dold = md - ftr_md, old - ftr_old
        out["arms"][arm] = {"M_d": md, "Old": old, "S5_All": s5, "c09": c09, "dM_d": dmd, "dOld": dold}
        print(f"{arm:<12} {md:10.3f} {old:10.3f} {s5:11.3f} {c09:10.3f} {dmd:+14.3f} {dold:+14.3f}")

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
