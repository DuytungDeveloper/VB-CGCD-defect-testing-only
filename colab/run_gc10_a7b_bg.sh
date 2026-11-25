#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
FT_EPOCHS="${FT_EPOCHS:-20}"
PREFIX="${PREFIX:-colab_gc10_d3_yoloroi_ep${FT_EPOCHS}}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a7b_yolo_roi_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a7b_yolo_roi_summary.json}"
META_JSON="${META_JSON:-$WORK_DIR/gc10_a7b_yolo_roi_telemetry.json}"
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
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
if [ -n "${HF_TOKEN:-}" ]; then
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=== START GC10 STEP A7b CLASS-AGNOSTIC YOLOV8N(S0) + DINOV3 FUSION (stage=$STAGE) at $(date) ==="

python -c "import ultralytics" 2>/dev/null || pip install -q ultralytics

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
    "model": "YOLOv8n(S0 1-class) + DINOv3 Fusion",
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
        for k in ("torch_peak_vram_mb", "yolo_roi_diag"):
            if k in old:
                meta[k] = old[k]
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

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in "$DRIVE_REPO"/features/*__colab_gc10_dinov3_ep20_r0 "$DRIVE_REPO"/features/*__"${PREFIX}"*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "features/$b" ] || { echo ">>> Đồng bộ features từ Drive: $b"; cp -r "$d" "features/$b"; }
  done
  for d in "$DRIVE_REPO"/runs/*__"${PREFIX}"*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "runs/$b" ] || { echo ">>> Đồng bộ run từ Drive: $b"; cp -r "$d" "runs/$b"; }
  done
}

backup_to_drive() {
  mkdir -p "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in features/*__"${PREFIX}"*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
  done
  for d in runs/*__"${PREFIX}"*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ -d "$d/snap" ]; then
      rm -rf "$DRIVE_REPO/runs/$b"
      cp -r "$d" "$DRIVE_REPO/runs/$b"
    fi
  done
}

run_r() {
  local r="$1"
  local ftag="${PREFIX}_r${r}"
  local feat=""
  feat=$(ls -d features/*__"${ftag}"/dinov3-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)

  if [ -n "$feat" ] && [ -f "$feat/features-dinov3-vitb16-sl.npy" ]; then
    echo ">>> Đã có features ${ftag} tại ${feat}, bỏ qua bước YOLO+ROI."
  else
    link_gc10_dataset
    local a2b_root
    a2b_root=$(ls -d features/*__colab_gc10_dinov3_ep20_r"${r}" 2>/dev/null | sort | tail -n 1 || true)
    if [ -z "$a2b_root" ] || [ ! -f "$a2b_root/ft_ckpt.pt" ]; then
      echo "ERROR: Không tìm thấy features/*__colab_gc10_dinov3_ep20_r${r}/ft_ckpt.pt từ Bước A2b!" >&2
      exit 2
    fi
    local ts
    ts=$(date -u +%Y%m%d_%H%M%S)
    local out_root="features/${ts}__${ftag}"
    echo ">>> Train YOLOv8n(S0 1-class) + trích xuất DINOv3 ROI Fusion -> ${out_root} ($(date))..."
    local t0 hw_tmp mon_pid ext_log
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a7b_yolo_r${r}.json"
    ext_log="/tmp/a7b_extract_r${r}.log"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
    python colab/extract_gc10_a7b_yolo_roi.py \
      --a2b-feat-root "$a2b_root" \
      --out-root "$out_root" \
      --split splits/gc10_split.json \
      --alpha 0.5 \
      --yolo-epochs 30 \
      --batch-size 32 | tee "$ext_log"
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "yoloroi_r${r}" "$(($(date +%s) - t0))" "$hw_tmp"
    feat="${out_root}/dinov3-vitb16-sl"
    python - <<PY
import json, re
from pathlib import Path
log = Path("$ext_log").read_text(errors="replace")
p = Path("$META_JSON")
d = json.loads(p.read_text())
m = re.search(r"peak_vram_mb\s*=\s*([\d.]+)", log)
if m:
    d["torch_peak_vram_mb"] = float(m.group(1))
m2 = re.search(r"--yolo_s0_roi detected.*", log)
if m2:
    d["yolo_roi_diag"] = m2.group(0)
p.write_text(json.dumps(d, indent=2, ensure_ascii=False))
PY
    python scripts/feat_probe.py "$feat" || true
    backup_to_drive
  fi

  IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
  for s in "${SEEDS[@]}"; do
    local tag="${ftag}_s${s}"
    if ls runs/*__"${tag}"/snap/S1_14.pkl >/dev/null 2>&1; then
      echo ">>> Đã hoàn thành ${tag} (có snap/S1_14.pkl), bỏ qua."
      continue
    fi
    local t0 hw_tmp mon_pid
    t0=$(date +%s)
    hw_tmp="/tmp/hw_a7b_r${r}_s${s}.json"
    python colab/hw_monitor.py --out "$hw_tmp" &
    mon_pid=$!
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
    kill -INT "$mon_pid" 2>/dev/null || true
    wait "$mon_pid" 2>/dev/null || true
    record_stage "r${r}_s${s}" "$(($(date +%s) - t0))" "$hw_tmp"
    backup_to_drive
  done
}

run_eval() {
  local t0 hw_tmp mon_pid
  t0=$(date +%s)
  hw_tmp="/tmp/hw_a7b_eval.json"
  python colab/hw_monitor.py --out "$hw_tmp" &
  mon_pid=$!
  echo ">>> Tổng hợp kết quả Bước A7b (YOLOv8n(S0) + DINOv3 Fusion, fts=${FTS_CSV}, seeds=${SEEDS_CSV})..."
  local eval_txt
  eval_txt=$(python colab/eval_gc10_a2.py --prefix "$PREFIX" --model "yolov8n_s0_dinov3_fusion" --fts "$FTS_CSV" --seeds "$SEEDS_CSV" --meta-json "$META_JSON" --out-json "$SUMMARY_JSON")
  kill -INT "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  record_stage "eval" "$(($(date +%s) - t0))" "$hw_tmp"
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
print("=== THÔNG SỐ PHẦN CỨNG, PEAK & THỜI GIAN CHẠY (BƯỚC A7b - GC10 YOLOV8N(S0) + DINOV3 FUSION) ===")
print(f"CPU          : {d.get('cpu_model')} ({d.get('cpu_cores')} cores) | Peak CPU: {cpu_pk}%")
print(f"RAM          : Peak {ram_pk} GB / {d.get('ram_total_gb')} GB")
print(f"GPU          : {d.get('gpu')} | Peak GPU Util: {gpu_pk}% | Peak GPU VRAM: {vram_pk} MB")
if 'torch_peak_vram_mb' in d:
    print(f"Torch Peak   : {d['torch_peak_vram_mb']} MB")
if 'yolo_roi_diag' in d:
    print(f"YOLO-ROI     : {d['yolo_roi_diag']}")
print(f"PyTorch      : {d.get('torch_version')} (CUDA={d.get('torch_cuda')}) | JAX {d.get('jax_version')} ({d.get('jax_devices')})")
print(f"Tổng thời gian các công đoạn: {tot_sec} s ({tot_sec/60:.1f} phút)")
print("Chi tiết từng công đoạn:")
for k, v in stages.items():
    print(f"  - {k:<10}: {v.get('sec',0):4d}s | CPU peak {v.get('cpu_peak_pct',0):5.1f}% (avg {v.get('cpu_avg_pct',0):4.1f}%) | RAM peak {v.get('ram_peak_gb',0):5.2f} GB | GPU peak {v.get('gpu_util_peak_pct',0):5.1f}% | VRAM peak {v.get('gpu_vram_peak_mb',0):6.1f} MB")
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
