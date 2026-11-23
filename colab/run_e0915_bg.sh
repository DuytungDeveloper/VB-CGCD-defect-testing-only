#!/usr/bin/env bash
set -euo pipefail

WORK_DIR="${WORK_DIR:-/content/drive/MyDrive/VB_CGCD_Work}"
REPO_DIR="${REPO_DIR:-$WORK_DIR/VB-CGCD-defect-testing-only}"
SUMMARY_OUT="${SUMMARY_OUT:-$WORK_DIR/e0915_summary.txt}"

trap 'rc=$?; echo "=== FINISHED (exit=$rc) at $(date) ==="' EXIT

cd "$REPO_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "=== START E09.15 BACKGROUND RUN at $(date) ==="
python -c "import jax; print('JAX devices:', jax.devices())"

# 1. Phase 1 features (tái sử dụng nếu đã có trên Drive)
FEAT_DIR=$(ls -d features/*__colab_cub_ft_l25_r0/dino-vitb16-sl 2>/dev/null | sort | tail -n 1 || true)
if [ -z "$FEAT_DIR" ]; then
  echo ">>> Đang tải CUB-200 & chạy Phase 1 (FT seed r0)..."
  FT_SEED=0 FT_EPOCHS=10 FT_BATCH_SIZE=32 FT_ACCUM=4 FT_LABELED_IDX=splits/cub_labeled_hf_idx.npy \
    python scripts/exp.py run colab_cub_ft_l25_r0 --inputs - -- bash -c \
    'O="features/$(basename "$EXP_RUN_DIR")"; mkdir -p "$O" && FT_CKPT_PATH="$O/ft_ckpt.pt" python -u feature_extractor/dino-cub200.py --finetuned --output_dir "$O" --batch-size 128 --seed "$FT_SEED"'
  FEAT_DIR=$(ls -d features/*__colab_cub_ft_l25_r0/dino-vitb16-sl | sort | tail -n 1)
fi
echo "FEAT_DIR = $FEAT_DIR"

# 2. Dọn các run dở dang trước đó (chưa có snap/04.pkl)
for d in runs/*__colab_cub_ft_l25_r0_s*; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/snap/04.pkl" ]; then
    echo ">>> Xóa run dở dang: $d"
    rm -rf "$d"
  fi
done
for d in runs/*__colab_e09_15_ft_l25*; do
  [ -d "$d" ] || continue
  rm -rf "$d"
done

# 3. Tạo 5 chuỗi s0..s4 (tự động bỏ qua chuỗi đã hoàn thành)
for s in 0 1 2 3 4; do
  tag="colab_cub_ft_l25_r0_s${s}"
  if ls runs/*__"${tag}"/snap/04.pkl >/dev/null 2>&1; then
    echo ">>> Đã có ${tag}/snap/04.pkl, bỏ qua."
  else
    echo ">>> Tạo chuỗi ${tag} ($(date))..."
    python scripts/exp.py run "$tag" --inputs "$FEAT_DIR" -- \
      python lab/sessions.py run --features "$FEAT_DIR" --seed "$s" --dataset cub200
  fi
done

# 4. Commit log vào nhánh local để exp.py cho phép chạy bước đánh giá
git config user.name 'Colab Runner'
git config user.email 'colab@example.com'
git add notes/LOG.md runs/INDEX.txt runs/*__colab_cub_ft_l25* 2>/dev/null || true
git diff --cached --quiet || git commit -m "colab: luu log colab_cub_ft_l25_r0"

# 5. Kiểm tra nhanh
echo ">>> Chạy --checks-only..."
python lab/score.py e0915 --prefix colab_cub_ft_l25 --fts 0 --checks-only

# 6. Chạy chính thức E09.15
echo ">>> Chạy chính thức E09.15 ($(date))..."
python scripts/exp.py run colab_e09_15_ft_l25 -- \
  python lab/score.py e0915 --prefix colab_cub_ft_l25 --fts 0

# 7. Xuất bảng tổng hợp ra log và file trên Drive
echo ">>> Kết quả tổng hợp E09.15:"
python lab/score.py e0915-summary | tee "$SUMMARY_OUT"
