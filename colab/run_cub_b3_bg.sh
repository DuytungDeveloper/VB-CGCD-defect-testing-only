#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
FT_EPOCHS="${FT_EPOCHS:-10}"
FT_SEED="${FT_SEED:-0}"
DYDM_EPOCHS="${DYDM_EPOCHS:-12}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/cub_b3_dinov3_dydm_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/cub_b3_dinov3_dydm_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/cub_b3_dinov3_dydm_telemetry.json}"

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

echo "=== START CUB-200 STEP B3 (DINOv3 + NIW + FTR1 + DYDM, FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${FT_SEED}) at $(date) ==="

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
    "model": "CUB-200 B3 (DINOv3 + NIW + FTR1 + DYDM Discovery Branch)",
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

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for pat in "*__colab_cub_dinov3_l25_r${FT_SEED}*"; do
    for d in "$DRIVE_REPO"/features/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "features/$b" ] || { echo ">>> Đồng bộ features từ Drive: $b"; cp -r "$d" "features/$b"; }
    done
    for d in "$DRIVE_REPO"/runs/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "runs/$b" ] || { echo ">>> Đồng bộ run từ Drive: $b"; cp -r "$d" "runs/$b"; }
    done
  done
}

backup_to_drive() {
  mkdir -p "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in features/*__colab_cub_dinov3_l25_r"${FT_SEED}"; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
  done
  for d in runs/*__colab_cub_dinov3_l25_r"${FT_SEED}"_s*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ -d "$d/snap" ]; then
      rm -rf "$DRIVE_REPO/runs/$b"
      cp -r "$d" "$DRIVE_REPO/runs/$b"
    fi
  done
}

sync_from_drive

P1_TAG="colab_cub_dinov3_l25_r${FT_SEED}"

# 1. Phase 1 DINOv3 fine-tune on CUB-200 (leak-free l25 split: 2500 imgs of 100 base classes)
P1_ROOT=$(ls -d features/*__"${P1_TAG}" 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$P1_ROOT" ] && [ -f "$P1_ROOT/dinov3-vitb16-sl/features-dinov3-vitb16-sl.npy" ] && [ -f "$P1_ROOT/ft_ckpt.pt" ]; then
  echo ">>> Đã có Phase 1 CUB-200 DINOv3 tại ${P1_ROOT}, bỏ qua."
else
  TS=$(date -u +%Y%m%d_%H%M%S)
  P1_ROOT="features/${TS}__${P1_TAG}"
  mkdir -p "$P1_ROOT"
  echo ">>> Chạy Phase 1 CUB-200 DINOv3 (l25 leak-free, FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${FT_SEED}) lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_b3_p1.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  FT_LABELED_IDX="splits/cub_labeled_hf_idx.npy" \
  FT_EPOCHS="$FT_EPOCHS" \
  FT_SEED="$FT_SEED" \
  FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}" \
  FT_ACCUM="${FT_ACCUM:-4}" \
  FT_CKPT_PATH="$P1_ROOT/ft_ckpt.pt" \
    python -u feature_extractor/dino-cub200.py \
      --finetuned \
      --model dinov3_vitb16 \
      --output_dir "$P1_ROOT" \
      --device cuda \
      --seed "$FT_SEED" \
      --batch-size 128
  rm -rf "$P1_ROOT/dinov3_vitb16_features" "$P1_ROOT/dinov3_vitb16_labels" \
         "$P1_ROOT/dinov3_vitb16_test_features" "$P1_ROOT/dinov3_vitb16_test_labels"
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "p1_cub_dinov3" "$(($(date +%s) - T0))" "$HW_TMP"
  backup_to_drive
fi
P1_FEAT="${P1_ROOT}/dinov3-vitb16-sl"

# 2. Phase 2 sessions (5 sessions S1..S5) for s0..s4
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
for s in "${SEEDS[@]}"; do
  STAG="${P1_TAG}_s${s}"
  if ls runs/*__"${STAG}"/snap/S5_14.pkl >/dev/null 2>&1; then
    echo ">>> Đã xong Phase 2 ${STAG} (S1..S5), bỏ qua."
    continue
  fi
  echo ">>> Chạy Phase 2 CUB-200 ${STAG} (5 phiên S1..S5) lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_b3_p2_s${s}.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  PREV_DIR=$(ls -d runs/*__"${STAG}" 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "$PREV_DIR" ] && [ -d "$PREV_DIR/snap" ]; then
    python scripts/exp.py run "$STAG" --inputs "$P1_FEAT" -- \
      python lab/sessions.py run --features "$P1_FEAT" --seed "$s" --dataset cub200 --resume "$PREV_DIR"
    rm -rf "$PREV_DIR"
  else
    [ -n "$PREV_DIR" ] && rm -rf "$PREV_DIR"
    python scripts/exp.py run "$STAG" --inputs "$P1_FEAT" -- \
      python lab/sessions.py run --features "$P1_FEAT" --seed "$s" --dataset cub200
  fi
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "p2_s${s}" "$(($(date +%s) - T0))" "$HW_TMP"
  backup_to_drive
done

# 3. DYDM Discovery Branch + NIW Memory per seed (auto-resume)
for s in "${SEEDS[@]}"; do
  if python -c "import json; d=json.load(open('$SUMMARY_JSON')); exit(0 if 's${s}' in d.get('per_seed',{}) else 1)" 2>/dev/null; then
    echo ">>> Đã có kết quả B3 DYDM s${s}, bỏ qua."
    continue
  fi
  echo ">>> Chạy Bước B3 DYDM (DINOv3) CUB-200 cho seed s${s} lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_b3_dydm_s${s}.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  python colab/run_cub_b3_dinov3_dydm.py \
    --prefix "$P1_TAG" \
    --feat-dir "$P1_FEAT" \
    --ckpt "${P1_ROOT}/ft_ckpt.pt" \
    --seeds "$s" \
    --epochs "$DYDM_EPOCHS" \
    --out-json "$SUMMARY_JSON"
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "dydm_s${s}" "$(($(date +%s) - T0))" "$HW_TMP"
done

# 4. Final plain-Vietnamese summary + hardware peak report
python - <<PY | tee "$SUMMARY_OUT"
import json
from pathlib import Path

meta = json.loads(Path("$META_JSON").read_text()) if Path("$META_JSON").exists() else {}
res = json.loads(Path("$SUMMARY_JSON").read_text())
stages = meta.get("stages", {})
tot_sec = sum(int(v.get("sec", 0)) for v in stages.values())
peak_cpu = max([float(v.get("cpu_peak_pct", 0)) for v in stages.values()] + [0.0])
peak_ram = max([float(v.get("ram_peak_gb", 0)) for v in stages.values()] + [0.0])
peak_gpu = max([float(v.get("gpu_util_peak_pct", 0)) for v in stages.values()] + [0.0])
peak_vram = max([float(v.get("gpu_vram_peak_mb", 0)) for v in stages.values()] + [0.0])

print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC B3 - CUB-200 DINOv3) ===")
print(f"CPU          : {meta.get('cpu_model')} ({meta.get('cpu_cores')} cores) | Peak CPU: {peak_cpu:.1f}%")
print(f"RAM          : Peak {peak_ram:.2f} GB / {meta.get('ram_total_gb')} GB")
print(f"GPU          : {meta.get('gpu')} | Peak GPU Util: {peak_gpu:.1f}% | Peak GPU VRAM: {peak_vram:.1f} MB")
print(f"PyTorch      : {meta.get('torch_version')} (CUDA={meta.get('torch_cuda')}) | JAX {meta.get('jax_version')} ({meta.get('jax_devices')})")
print(f"Tổng thời gian: {tot_sec} s ({tot_sec/60:.1f} phút)\n")

print("=== BẢNG TỔNG HỢP DỄ HIỂU — BƯỚC B3 (CUB-200, 11.788 ảnh, 200 lớp, 5 phiên S1..S5) ===")
print("Đối chiếu mốc chuẩn:")
print("  - Bài báo gốc VB-CGCD (ICML 2025) : S5_All = 66.44% | M_d = 56.46%")
print("  - Đỉnh Nhóm (FTR1 trên DINOv1)    : S5_All = 74.68% | M_d = 70.55% | Old = 79.44%")
print("  - Bước B2 (DYDM_km trên DINOv1)   : S5_All = 75.02% | M_d = 71.45% | Old = 79.50%\n")

print(f"{'Phương pháp (DINOv3)':<18} {'S0 (%)':>8} {'Lớp mới(M_d)':>13} {'Lớp cũ(Old%)':>13} {'Gom cụm(c09%)':>14} {'S5_All% ± std':>15}")
print("-" * 88)
for arm, m in res["arms"].items():
    print(f"{arm:<18} {m['S0']:7.2f}% {m['M_d']:12.2f}% {m['Old']:12.2f}% {m['c09']:13.2f}% {m['S5_All']:6.2f} ± {m['S5_All_std']:4.2f}%")

print("\n--- CHI TIẾT TỪNG PHIÊN S1..S5 (All% / Old% / New%) ---")
for arm, m in res["arms"].items():
    ps = m["per_session"]
    row = " | ".join(f"{s}: {ps[s]['All']:.2f}/{ps[s]['Old']:.2f}/{ps[s]['New']:.2f}" for s in ("S1","S2","S3","S4","S5"))
    print(f"  {arm:<14}: {row}")
PY
