#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-auto}"
WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
DRIVE_REPO="${DRIVE_REPO:-$WORK_DIR/VB-CGCD-defect-testing-only}"
REPO_DIR="${REPO_DIR:-/content/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/e0915_summary.txt}"

trap 'rc=$?; echo "=== FINISHED (stage=$STAGE, exit=$rc) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== START E09.15 RUN (stage=$STAGE) at $(date) ==="
python -c "import jax; print('JAX devices:', jax.devices())"

sync_from_drive() {
  mkdir -p features runs "$DRIVE_REPO/features" "$DRIVE_REPO/runs"
  for d in "$DRIVE_REPO"/features/*__colab_cub_ft_l25_r0; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    if [ ! -d "features/$b" ]; then
      echo ">>> Đồng bộ features từ Drive: $b"
      cp -r "$d" "features/$b"
    fi
  done
  for d in "$DRIVE_REPO"/runs/*__colab_cub_ft_l25* "$DRIVE_REPO"/runs/*__colab_e09_15_ft_l25*; do
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
  for d in features/*__colab_cub_ft_l25_r0; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [ -d "$DRIVE_REPO/features/$b" ] || cp -r "$d" "$DRIVE_REPO/features/$b"
  done
  for d in runs/*__colab_*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    rm -rf "$DRIVE_REPO/runs/$b.tmp"
    cp -r "$d" "$DRIVE_REPO/runs/$b.tmp"
    rm -rf "$DRIVE_REPO/runs/$b"
    mv "$DRIVE_REPO/runs/$b.tmp" "$DRIVE_REPO/runs/$b"
  done
}

sync_from_drive

run_phase1() {
  local feat
  feat=$(ls -d features/*__colab_cub_ft_l25_r0/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
  if [ -n "$feat" ]; then
    echo ">>> Đã có Phase 1 features: $feat (bỏ qua)"
    return 0
  fi
  echo ">>> Đang tải CUB-200 & chạy Phase 1 (FT seed r0)..."
  FT_SEED=0 FT_EPOCHS=10 FT_BATCH_SIZE=32 FT_ACCUM=4 FT_LABELED_IDX=splits/cub_labeled_hf_idx.npy \
    python scripts/exp.py run colab_cub_ft_l25_r0 --inputs - -- bash -c \
    'O="features/$(basename "$EXP_RUN_DIR")"; mkdir -p "$O" && FT_CKPT_PATH="$O/ft_ckpt.pt" python -u feature_extractor/dino-cub200.py --finetuned --output_dir "$O" --batch-size 128 --seed "$FT_SEED"'
  backup_to_drive
}

run_seed() {
  local s="$1"
  local tag="colab_cub_ft_l25_r0_s${s}"
  local feat
  feat=$(ls -d features/*__colab_cub_ft_l25_r0/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
  if [ -z "$feat" ]; then
    echo "ERROR: Chưa có Phase 1 features. Hãy chạy stage phase1 trước." >&2
    return 1
  fi

  if ls runs/*__"${tag}"/snap/S5_14.pkl >/dev/null 2>&1; then
    echo ">>> Đã hoàn thành ${tag} (có snap/S5_14.pkl), bỏ qua."
    return 0
  fi

  local prev_dir=""
  prev_dir=$(ls -d runs/*__"${tag}" 2>/dev/null | sort | tail -n 1 || true)

  if [ -n "$prev_dir" ] && [ -d "$prev_dir/snap" ]; then
    echo ">>> Resume ${tag} từ ${prev_dir} ($(date))..."
    python scripts/exp.py run "$tag" --inputs "$feat" -- \
      python lab/sessions.py run --features "$feat" --seed "$s" --dataset cub200 --resume "$prev_dir"
    rm -rf "$prev_dir"
  else
    [ -n "$prev_dir" ] && rm -rf "$prev_dir"
    echo ">>> Tạo mới chuỗi ${tag} ($(date))..."
    python scripts/exp.py run "$tag" --inputs "$feat" -- \
      python lab/sessions.py run --features "$feat" --seed "$s" --dataset cub200
  fi
  backup_to_drive
}

commit_logs() {
  git config user.name 'Colab Runner'
  git config user.email 'colab@example.com'
  git add notes/LOG.md runs/INDEX.txt runs/*__colab_cub_ft_l25* 2>/dev/null || true
  git diff --cached --quiet || git commit -m "colab: luu log colab_cub_ft_l25_r0"
}

run_checks() {
  echo ">>> Chạy --checks-only..."
  python lab/score.py e0915 --prefix colab_cub_ft_l25 --fts 0 --checks-only
}

run_e0915() {
  if [ "$STAGE" = "auto" ] && ls runs/*__colab_e09_15_ft_l25/metrics.json >/dev/null 2>&1; then
    echo ">>> Đã có kết quả colab_e09_15_ft_l25/metrics.json, bỏ qua."
    return 0
  fi
  for d in runs/*__colab_e09_15_ft_l25*; do
    [ -d "$d" ] || continue
    [ -f "$d/metrics.json" ] || rm -rf "$d"
  done
  echo ">>> Chạy chính thức E09.15 ($(date))..."
  python scripts/exp.py run colab_e09_15_ft_l25 -- \
    python lab/score.py e0915 --prefix colab_cub_ft_l25 --fts 0
  backup_to_drive
}

run_summary() {
  echo ">>> Kết quả tổng hợp E09.15:"
  python lab/score.py e0915-summary | tee "$SUMMARY_OUT"
}

case "$STAGE" in
  auto)
    run_phase1
    for s in 0 1 2 3 4; do
      run_seed "$s"
    done
    commit_logs
    run_checks
    run_e0915
    run_summary
    ;;
  phase1) run_phase1 ;;
  s0|s1|s2|s3|s4)
    run_seed "${STAGE#s}"
    commit_logs
    ;;
  checks)
    commit_logs
    run_checks
    ;;
  e0915)
    commit_logs
    run_e0915
    run_summary
    ;;
  summary)
    run_summary
    ;;
  *)
    echo "Stage không hợp lệ: $STAGE (chọn: auto | phase1 | s0..s4 | checks | e0915 | summary)" >&2
    exit 2
    ;;
esac
