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
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/neu_a10_dinov3_autoroi_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/neu_a10_dinov3_autoroi_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/neu_a10_dinov3_autoroi_telemetry.json}"

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

echo "=== START NEU-DET STEP A10 (DINOv3 Full vs Auto-ROI Fusion + Imbalanced-DYDM, FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${FT_SEED}) at $(date) ==="

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
    "model": "NEU-DET A10 (DINOv3 Auto-ROI Fusion + Imbalanced-DYDM + NIW Memory)",
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

link_neu_dataset() {
  mkdir -p datasets
  if [ -d "datasets/NEU-DET/train/images" ] && [ ! -L "datasets/NEU-DET" ]; then
    return 0
  fi
  if [ -d "/content/NEU-DET/train/images" ]; then
    rm -f datasets/NEU-DET
    ln -sfn "/content/NEU-DET" datasets/NEU-DET
    return 0
  fi
  for cand in "/content/drive/MyDrive/Datasets/NEU-DET" "$WORK_DIR/NEU-DET" "$WORK_DIR/Datasets/NEU-DET" "$DRIVE_REPO/datasets/NEU-DET"; do
    if [ -d "$cand/train/images" ]; then
      echo ">>> Copy NEU-DET từ Drive sang /content/NEU-DET..."
      cp -r "$cand" /content/NEU-DET
      rm -f datasets/NEU-DET
      ln -sfn "/content/NEU-DET" datasets/NEU-DET
      return 0
    fi
  done
  echo "ERROR: Không tìm thấy NEU-DET (cần có train/images và validation/images) tại /content/drive/MyDrive/Datasets/NEU-DET" >&2
  return 1
}

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for pat in "*__colab_neu_dinov3_ep${FT_EPOCHS}_r${FT_SEED}*" "*__colab_neu_d3_autoroi_ep${FT_EPOCHS}_r${FT_SEED}*"; do
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
  for pat in "*__colab_neu_dinov3_ep${FT_EPOCHS}_r${FT_SEED}" "*__colab_neu_d3_autoroi_ep${FT_EPOCHS}_r${FT_SEED}"; do
    for d in features/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
    done
  done
  for pat in "*__colab_neu_dinov3_ep${FT_EPOCHS}_r${FT_SEED}_s*" "*__colab_neu_d3_autoroi_ep${FT_EPOCHS}_r${FT_SEED}_s*"; do
    for d in runs/$pat; do
      [ -d "$d" ] || continue
      b=$(basename "$d")
      if [ -f "$d/snap/14.pkl" ]; then
        rm -rf "$DRIVE_REPO/runs/$b"
        cp -r "$d" "$DRIVE_REPO/runs/$b"
      fi
    done
  done
}

link_neu_dataset
sync_from_drive

P1_TAG="colab_neu_dinov3_ep${FT_EPOCHS}_r${FT_SEED}"
ROI_TAG="colab_neu_d3_autoroi_ep${FT_EPOCHS}_r${FT_SEED}"

# 1. Phase 1 DINOv3 fine-tune on NEU-DET
P1_ROOT=$(ls -d features/*__"${P1_TAG}" 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$P1_ROOT" ] && [ -f "$P1_ROOT/dinov3-vitb16-sl/features-dinov3-vitb16-sl.npy" ] && [ -f "$P1_ROOT/ft_ckpt.pt" ]; then
  echo ">>> Đã có Phase 1 NEU-DET DINOv3 tại ${P1_ROOT}, bỏ qua."
else
  echo ">>> Chạy Phase 1 NEU-DET DINOv3 (FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${FT_SEED}) lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_a10_p1.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  FT_EPOCHS="$FT_EPOCHS" FT_SEED="$FT_SEED" python scripts/exp.py run "$P1_TAG" --inputs splits/neu_split.json -- \
    bash scripts/run_phase1_neu.sh --split splits/neu_split.json --resize full --model dinov3_vitb16
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "p1_neu" "$(($(date +%s) - T0))" "$HW_TMP"
  P1_ROOT=$(ls -d features/*__"${P1_TAG}" | sort | tail -n 1)
  backup_to_drive
fi
P1_FEAT="${P1_ROOT}/dinov3-vitb16-sl"

# 2. DINOv3 Auto-ROI Fusion on NEU-DET
ROI_ROOT=$(ls -d features/*__"${ROI_TAG}" 2>/dev/null | sort | tail -n 1 || true)
if [ -n "$ROI_ROOT" ] && [ -f "$ROI_ROOT/dinov3-vitb16-sl/features-dinov3-vitb16-sl.npy" ] && [ -f "$ROI_ROOT/ft_ckpt.pt" ]; then
  echo ">>> Đã có NEU-DET Auto-ROI Fusion tại ${ROI_ROOT}, bỏ qua."
else
  TS=$(date -u +%Y%m%d_%H%M%S)
  ROI_ROOT="features/${TS}__${ROI_TAG}"
  echo ">>> Trích xuất NEU-DET DINOv3 Auto-ROI Fusion -> ${ROI_ROOT} lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_a10_roi.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  python colab/extract_neu_a10_dinov3_roi.py \
    --p1-feat-root "$P1_ROOT" \
    --out-root "$ROI_ROOT" \
    --split splits/neu_split.json \
    --alpha 0.5 \
    --loc-size 384 \
    --batch-size 32
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "roi_neu" "$(($(date +%s) - T0))" "$HW_TMP"
  backup_to_drive
fi
ROI_FEAT="${ROI_ROOT}/dinov3-vitb16-sl"

# 3. Phase 2 replay (Full baseline & Auto-ROI Fusion) for s0..s4
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
for mode in "full" "roi"; do
  for s in "${SEEDS[@]}"; do
    if [ "$mode" = "full" ]; then
      STAG="${P1_TAG}_s${s}"
      FDIR="$P1_FEAT"
    else
      STAG="${ROI_TAG}_s${s}"
      FDIR="$ROI_FEAT"
    fi
    if ls runs/*__"${STAG}"/snap/14.pkl >/dev/null 2>&1; then
      echo ">>> Đã xong Phase 2 ${STAG}, bỏ qua."
      continue
    fi
    echo ">>> Chạy Phase 2 NEU-DET ${STAG} lúc $(date)..."
    T0=$(date +%s)
    HW_TMP="/tmp/hw_a10_p2_${mode}_s${s}.json"
    python colab/hw_monitor.py --out "$HW_TMP" &
    MON_PID=$!
    rm -rf runs/*__"${STAG}" 2>/dev/null || true
    python scripts/exp.py run "$STAG" --inputs "$FDIR" -- \
      python lab/replay.py run --features "$FDIR" --seed "$s" --dataset neudet
    kill -INT "$MON_PID" 2>/dev/null || true
    wait "$MON_PID" 2>/dev/null || true
    record_stage "p2_${mode}_s${s}" "$(($(date +%s) - T0))" "$HW_TMP"
    backup_to_drive
  done
done

# 4. Imbalanced-DYDM (A10) per seed with auto-resume
for s in "${SEEDS[@]}"; do
  if python -c "import json; d=json.load(open('$SUMMARY_JSON')); exit(0 if 's${s}' in d.get('per_seed',{}) else 1)" 2>/dev/null; then
    echo ">>> Đã có kết quả A10 ImbDYDM s${s}, bỏ qua."
    continue
  fi
  echo ">>> Chạy A10 Imbalanced-DYDM NEU-DET cho seed s${s} lúc $(date)..."
  T0=$(date +%s)
  HW_TMP="/tmp/hw_a10_dydm_s${s}.json"
  python colab/hw_monitor.py --out "$HW_TMP" &
  MON_PID=$!
  python colab/run_neu_a10_autoroi_dydm.py \
    --prefix "$ROI_TAG" \
    --full-prefix "$P1_TAG" \
    --feat-dir "$ROI_FEAT" \
    --ckpt "${ROI_ROOT}/ft_ckpt.pt" \
    --split splits/neu_split.json \
    --seeds "$s" \
    --epochs "$DYDM_EPOCHS" \
    --out "$SUMMARY_JSON"
  kill -INT "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  record_stage "dydm_s${s}" "$(($(date +%s) - T0))" "$HW_TMP"
done

# 5. Export final Vietnamese summary + hardware peak report
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

print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC A10 - NEU-DET) ===")
print(f"CPU          : {meta.get('cpu_model')} ({meta.get('cpu_cores')} cores) | Peak CPU: {peak_cpu:.1f}%")
print(f"RAM          : Peak {peak_ram:.2f} GB / {meta.get('ram_total_gb')} GB")
print(f"GPU          : {meta.get('gpu')} | Peak GPU Util: {peak_gpu:.1f}% | Peak GPU VRAM: {peak_vram:.1f} MB")
print(f"PyTorch      : {meta.get('torch_version')} (CUDA={meta.get('torch_cuda')}) | JAX {meta.get('jax_version')} ({meta.get('jax_devices')})")
print(f"Tổng thời gian: {tot_sec} s ({tot_sec/60:.1f} phút)\n")

print("=== BẢNG TỔNG HỢP DỄ HIỂU — BƯỚC A10 (NEU-DET DINOv3 Full vs Auto-ROI + Imbalanced-DYDM) ===")
print("Ghi chú:")
print("  - Lớp cũ (Old %)     : Độ chính xác nhận diện 3 lớp lỗi đã học ở S0 (crazing, inclusion, patches).")
print("  - Lớp mới (New %)    : % tổng số ảnh thuộc 3 lớp lỗi mới chưa có nhãn (pitted_surface, rolled-in_scale, scratches).")
print("  - Cân bằng mới (bal) : Trung bình độ chính xác từng lớp mới (0..1).")
print("  - Gom cụm (c09 %)    : Độ thuần khi tự gom cụm 3 lớp mới chưa có nhãn.")
print("  - Tổng thể (All %)   : Độ chính xác chung trên toàn bộ 6 lớp lỗi thép NEU-DET.\n")

print(f"{'Phương pháp':<18s}  {'Lớp cũ(Old%)':>12s}  {'Lớp mới(New%)':>13s}  {'Cân bằng(bal)':>17s}  {'Gom cụm(c09%)':>13s}  {'Tổng(All%±std)':>14s}")
print("-" * 96)
for arm, m in res["arms"].items():
    print(f"{arm:<18s}  {m['Old_mean']:11.2f}%  {m['New_mean']:12.2f}%  {m['bal_mean']:7.4f} ± {m['bal_std']:.4f}  {m['c09_mean']:12.2f}%  {m['All_mean']:6.2f} ± {m['All_std']:5.2f}%")

novel_names = res["novel_classes"]
print("\n--- CHI TIẾT ĐỘ CHÍNH XÁC TỪNG LỚP LỖI MỚI NEU-DET (%) ---")
hdr = f"{'Phương pháp':<18s}  " + "  ".join(f"{nm:>16s}" for nm in novel_names)
print(hdr)
print("-" * len(hdr))
for arm, m in res["arms"].items():
    pc = m["per_class_novel"]
    print(f"{arm:<18s}  " + "  ".join(f"{pc[nm]:15.1f}%" for nm in novel_names))
PY
