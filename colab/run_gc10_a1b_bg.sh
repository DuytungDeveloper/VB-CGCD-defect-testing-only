#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a1b_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a1b_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/gc10_a1b_telemetry.json}"
FT_EPOCHS="${FT_EPOCHS:-20}"
FTS_CSV="${FTS_CSV:-0}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"

TOTAL_T0=$(date +%s)
trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc, total_sec=$(($(date +%s) - TOTAL_T0))) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== START GC10 STEP A1b AUTO ROI (stage=$STAGE, FT_EPOCHS=$FT_EPOCHS) at $(date) ==="

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
    "stages_sec": {},
}
p = Path("$META_JSON")
if p.exists():
    try:
        old = json.loads(p.read_text())
        meta["stages_sec"] = old.get("stages_sec", {})
        for k in ("peak_vram_mb", "auto_roi_diag"):
            if k in old:
                meta[k] = old[k]
    except Exception:
        pass
p.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"HARDWARE | CPU: {cpu_model} ({os.cpu_count()} cores) | RAM: {ram_gb} GB | GPU: {gpu_info}")
print(f"FRAMEWORKS | Torch {torch.__version__} (CUDA={torch.cuda.is_available()}) | JAX {jax.__version__} ({jax.devices()})")
PY

record_stage_time() {
  local key="$1" sec="$2"
  python - <<PY
import json
from pathlib import Path
p = Path("$META_JSON")
d = json.loads(p.read_text()) if p.exists() else {"stages_sec": {}}
d.setdefault("stages_sec", {})["$key"] = int("$sec")
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
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
      echo ">>> Copy GC10-DET từ Drive sang /content/GC10-DET để đọc ảnh nhanh..."
      cp -r "$cand" /content/GC10-DET
      rm -f datasets/GC10-DET
      ln -sfn "/content/GC10-DET" datasets/GC10-DET
      echo ">>> Linked datasets/GC10-DET -> /content/GC10-DET"
      return 0
    fi
  done
  echo "ERROR: Không tìm thấy GC10-DET tại /content/drive/MyDrive/Datasets/GC10-DET" >&2
  return 1
}

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in "$DRIVE_REPO"/features/*__colab_gc10_autoroi_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "features/$b" ] || { echo ">>> Đồng bộ features từ Drive: $b"; cp -r "$d" "features/$b"; }
  done
  for d in "$DRIVE_REPO"/runs/*__colab_gc10_autoroi_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "runs/$b" ] || { echo ">>> Đồng bộ run từ Drive: $b"; cp -r "$d" "runs/$b"; }
  done
}

backup_to_drive() {
  mkdir -p "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in features/*__colab_gc10_autoroi_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
  done
  for d in runs/*__colab_gc10_autoroi_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    rm -rf "$DRIVE_REPO/runs/$b.tmp"
    cp -r "$d" "$DRIVE_REPO/runs/$b.tmp"
    rm -rf "$DRIVE_REPO/runs/$b"
    mv "$DRIVE_REPO/runs/$b.tmp" "$DRIVE_REPO/runs/$b"
  done
}

run_r() {
  local r="$1"
  local ftag="colab_gc10_autoroi_ep${FT_EPOCHS}_r${r}"
  local feat
  feat=$(ls -d features/*__"${ftag}"/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
  if [ -z "$feat" ]; then
    link_gc10_dataset
    echo ">>> Phase 1 GC10 Unsupervised Auto-ROI: FT_EPOCHS=${FT_EPOCHS}, FT_SEED=${r} ($(date))..."
    local t0
    t0=$(date +%s)
    FT_EPOCHS="$FT_EPOCHS" FT_SEED="$r" python scripts/exp.py run "$ftag" --inputs splits/gc10_split.json -- \
      bash scripts/run_phase1_neu.sh --split splits/gc10_split.json --resize full --roi_auto
    local dt=$(($(date +%s) - t0))
    record_stage_time "phase1_r${r}" "$dt"
    feat=$(ls -d features/*__"${ftag}"/dino-vitb16-sl | sort | tail -n 1)
    python - <<PY
import json, re
from pathlib import Path
log = sorted(Path("runs").glob("*__${ftag}/console.log"))[-1].read_text(errors="replace")
p = Path("$META_JSON")
d = json.loads(p.read_text())
m = re.search(r"peak_vram_mb\s*=\s*([\d.]+)", log)
if m:
    d["peak_vram_mb"] = float(m.group(1))
m2 = re.search(r"--roi_auto localized.*", log)
if m2:
    d["auto_roi_diag"] = m2.group(0)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
    python scripts/feat_probe.py "$feat" || true
    backup_to_drive
  else
    echo ">>> Đã có Phase 1 ${ftag}: $feat (bỏ qua)"
  fi

  IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
  for s in "${SEEDS[@]}"; do
    local tag="${ftag}_s${s}"
    if ls runs/*__"${tag}"/snap/S1_14.pkl >/dev/null 2>&1; then
      echo ">>> Đã hoàn thành ${tag} (có snap/S1_14.pkl), bỏ qua."
      continue
    fi
    local t0
    t0=$(date +%s)
    local prev_dir=""
    prev_dir=$(ls -d runs/*__"${tag}" 2>/dev/null | sort | tail -n 1 || true)
    if [ -n "$prev_dir" ] && [ -d "$prev_dir/snap" ]; then
      echo ">>> Resume ${tag} từ ${prev_dir} ($(date))..."
      python scripts/exp.py run "$tag" --inputs "$feat" -- \
        python lab/sessions.py run --features "$feat" --seed "$s" --dataset gc10 --resume "$prev_dir"
      rm -rf "$prev_dir"
    else
      [ -n "$prev_dir" ] && rm -rf "$prev_dir"
      echo ">>> Tạo mới chuỗi ${tag} ($(date))..."
      python scripts/exp.py run "$tag" --inputs "$feat" -- \
        python lab/sessions.py run --features "$feat" --seed "$s" --dataset gc10
    fi
    record_stage_time "r${r}_s${s}" "$(($(date +%s) - t0))"
    backup_to_drive
  done
}

run_eval() {
  local t0
  t0=$(date +%s)
  echo ">>> Tổng hợp kết quả Bước A1b (FT_EPOCHS=${FT_EPOCHS}, fts=${FTS_CSV}, seeds=${SEEDS_CSV})..."
  local eval_txt
  eval_txt=$(python colab/eval_gc10_a1a.py --prefix "colab_gc10_autoroi_ep${FT_EPOCHS}" --fts "$FTS_CSV" --seeds "$SEEDS_CSV" --meta-json "$META_JSON" --out-json "$SUMMARY_JSON")
  record_stage_time "eval" "$(($(date +%s) - t0))"
  record_stage_time "total_wall" "$(($(date +%s) - TOTAL_T0))"
  {
    python - <<PY
import json
from pathlib import Path
d = json.loads(Path("$META_JSON").read_text())
print("=== THÔNG SỐ PHẦN CỨNG & THỜI GIAN CHẠY (BƯỚC A1b) ===")
print(f"CPU       : {d.get('cpu_model')} ({d.get('cpu_cores')} cores)")
print(f"RAM       : {d.get('ram_total_gb')} GB")
print(f"GPU       : {d.get('gpu')}")
if 'peak_vram_mb' in d:
    print(f"Peak VRAM : {d['peak_vram_mb']} MB")
if 'auto_roi_diag' in d:
    print(f"Auto-ROI  : {d['auto_roi_diag']}")
print(f"PyTorch   : {d.get('torch_version')} (CUDA={d.get('torch_cuda')})")
print(f"JAX       : {d.get('jax_version')} ({d.get('jax_devices')})")
print("Thời gian từng giai đoạn (giây):", d.get("stages_sec", {}))
print()
PY
    echo "$eval_txt"
  } | tee "$SUMMARY_OUT"
}

sync_from_drive

IFS=',' read -ra FTS <<< "$FTS_CSV"
case "$STAGE" in
  auto)
    for r in "${FTS[@]}"; do
      run_r "$r"
    done
    run_eval
    ;;
  eval)
    run_eval
    ;;
  *)
    echo "Stage không hợp lệ: $STAGE (chọn: auto | eval)" >&2
    exit 2
    ;;
esac
