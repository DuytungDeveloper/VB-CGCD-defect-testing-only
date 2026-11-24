#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/gc10_a0_summary.txt}"
SUMMARY_JSON="${SUMMARY_JSON:-$WORK_DIR/gc10_a0_summary.json}"
EPOCHS_CSV="${EPOCHS_CSV:-10,20,30}"
FTS_CSV="${FTS_CSV:-0}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"

trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== START GC10 STEP A0 (stage=$STAGE) at $(date) ==="
python -c "import torch, jax; print('Torch CUDA:', torch.cuda.is_available(), '| JAX devices:', jax.devices())"

link_gc10_dataset() {
  mkdir -p datasets
  if [ -d "datasets/GC10-DET/1" ]; then
    return 0
  fi
  for cand in "/content/drive/MyDrive/Datasets/GC10-DET" "$WORK_DIR/GC10-DET" "$WORK_DIR/Datasets/GC10-DET"; do
    if [ -d "$cand/1" ]; then
      ln -sfn "$cand" datasets/GC10-DET
      echo ">>> Linked datasets/GC10-DET -> $cand"
      return 0
    fi
  done
  echo "ERROR: Không tìm thấy bộ ảnh GC10-DET (cần có thư mục 1..10) tại /content/drive/MyDrive/Datasets/GC10-DET hoặc $WORK_DIR/GC10-DET" >&2
  return 1
}

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in "$DRIVE_REPO"/features/*__colab_gc10_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ ! -d "features/$b" ]; then
      echo ">>> Đồng bộ features từ Drive: $b"
      cp -r "$d" "features/$b"
    fi
  done
  for d in "$DRIVE_REPO"/runs/*__colab_gc10_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ ! -d "runs/$b" ]; then
      echo ">>> Đồng bộ run từ Drive: $b"
      cp -r "$d" "runs/$b"
    fi
  done
}

backup_to_drive() {
  mkdir -p "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in features/*__colab_gc10_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
  done
  for d in runs/*__colab_gc10_ep*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    rm -rf "$DRIVE_REPO/runs/$b.tmp"
    cp -r "$d" "$DRIVE_REPO/runs/$b.tmp"
    rm -rf "$DRIVE_REPO/runs/$b"
    mv "$DRIVE_REPO/runs/$b.tmp" "$DRIVE_REPO/runs/$b"
  done
}

run_epoch() {
  local ep="$1"
  local r="$2"
  local ftag="colab_gc10_ep${ep}_r${r}"
  local feat
  feat=$(ls -d features/*__"${ftag}"/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
  if [ -z "$feat" ]; then
    link_gc10_dataset
    echo ">>> Phase 1 GC10: FT_EPOCHS=${ep}, FT_SEED=${r} ($(date))..."
    FT_EPOCHS="$ep" FT_SEED="$r" python scripts/exp.py run "$ftag" --inputs splits/gc10_split.json -- \
      bash scripts/run_phase1_neu.sh --split splits/gc10_split.json --resize full
    feat=$(ls -d features/*__"${ftag}"/dino-vitb16-sl | sort | tail -n 1)
    python scripts/feat_probe.py "$feat" || true
    backup_to_drive
  else
    echo ">>> Đã có Phase 1 ${ftag}: $feat (bỏ qua)"
  fi

  IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
  for s in "${SEEDS[@]}"; do
    local tag="colab_gc10_ep${ep}_r${r}_s${s}"
    if ls runs/*__"${tag}"/snap/S1_14.pkl >/dev/null 2>&1; then
      echo ">>> Đã hoàn thành ${tag} (có snap/S1_14.pkl), bỏ qua."
      continue
    fi
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
    backup_to_drive
  done
}

run_eval() {
  echo ">>> Tổng hợp kết quả Bước A0 (epochs=${EPOCHS_CSV}, fts=${FTS_CSV}, seeds=${SEEDS_CSV})..."
  python colab/eval_gc10_a0.py --epochs "$EPOCHS_CSV" --fts "$FTS_CSV" --seeds "$SEEDS_CSV" --out-json "$SUMMARY_JSON" | tee "$SUMMARY_OUT"
}

sync_from_drive

IFS=',' read -ra EPOCHS <<< "$EPOCHS_CSV"
IFS=',' read -ra FTS <<< "$FTS_CSV"

case "$STAGE" in
  auto)
    for ep in "${EPOCHS[@]}"; do
      for r in "${FTS[@]}"; do
        run_epoch "$ep" "$r"
      done
    done
    run_eval
    ;;
  ep10|ep20|ep30)
    ep="${STAGE#ep}"
    for r in "${FTS[@]}"; do
      run_epoch "$ep" "$r"
    done
    run_eval
    ;;
  eval)
    run_eval
    ;;
  *)
    echo "Stage không hợp lệ: $STAGE (chọn: auto | ep10 | ep20 | ep30 | eval)" >&2
    exit 2
    ;;
esac
