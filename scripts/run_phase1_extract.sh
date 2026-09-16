#!/usr/bin/env bash
# VB-CGCD phase 1 only: fine-tune the backbone, then extract features. Always call it through exp:
#   python scripts/exp.py run extract_c100 -- bash scripts/run_phase1_extract.sh
# 1) output folder = features/<run_id>/ (a new folder for every run, never overwritten)
# 2) stops if torch does not see the GPU
# 3) runs dino-cifar100.py with EVERY argument written out, so console.log holds the full config
# 4) checks the result with scripts/check_phase1.py (exit code 1 if any check fails)
# Extra arguments come last and override the defaults (argparse: last value wins).
set -u
if [ -z "${EXP_RUN_DIR:-}" ]; then
  echo "run_phase1_extract: ERROR run this through scripts/exp.py"; exit 2
fi
RUN_ID=$(basename "$EXP_RUN_DIR")
OUT="features/$RUN_ID"
if [ -e "$OUT" ]; then
  echo "run_phase1_extract: ERROR $OUT already exists"; exit 2
fi

GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" 2>&1 | tail -1)
echo "run_phase1_extract: torch device = $GPU"
case "$GPU" in
  NO_GPU|*Error*) echo "run_phase1_extract: ERROR torch does not see the GPU"; exit 3 ;;
esac

mkdir -p "$OUT"
export FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}"   # PATCH 2: 32 x 4 = effective batch 128 on 8 GB VRAM
export FT_ACCUM="${FT_ACCUM:-4}"
export FT_CKPT_PATH="$OUT/ft_ckpt.pt"         # PATCH 5
echo "run_phase1_extract: FT_BATCH_SIZE=$FT_BATCH_SIZE FT_ACCUM=$FT_ACCUM FT_CKPT_PATH=$FT_CKPT_PATH"

ARGS=(--finetuned --output_dir "$OUT"
      --model dino_vitb16 --device cuda --seed 42
      --batch-size 128                        # keep 128: merge_npy file order decides the labeled split
      --num-workers 8 --img_size 224 --labeled_classes 50)   # these 3 are parsed but never used

echo "run_phase1_extract: python -u feature_extractor/dino-cifar100.py ${ARGS[*]} $*"
python -u feature_extractor/dino-cifar100.py "${ARGS[@]}" "$@"
rc=$?
if [ $rc -ne 0 ]; then
  echo "run_phase1_extract: extraction failed (exit $rc), checks skipped"; exit $rc
fi

python scripts/check_phase1.py "$OUT"
exit $?