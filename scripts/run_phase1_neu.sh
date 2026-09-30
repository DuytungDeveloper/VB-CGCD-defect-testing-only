#!/usr/bin/env bash
# NEU-DET phase 1, always through exp:
#   python scripts/exp.py run neu_extract --inputs splits/neu_split.json -- bash scripts/run_phase1_neu.sh
set -u
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"   # model is cached; no network at run time
[ -z "${EXP_RUN_DIR:-}" ] && { echo "run_phase1_neu: ERROR run this through scripts/exp.py"; exit 2; }
OUT="features/$(basename "$EXP_RUN_DIR")"
[ -e "$OUT" ] && { echo "run_phase1_neu: ERROR $OUT already exists"; exit 2; }
GPU=$(python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')" 2>&1 | tail -1)
echo "run_phase1_neu: torch device = $GPU"
case "$GPU" in NO_GPU|*Error*) echo "run_phase1_neu: ERROR torch does not see the GPU"; exit 3 ;; esac
export FT_SEED="${FT_SEED:-42}" FT_EPOCHS="${FT_EPOCHS:-10}"
export FT_BATCH_SIZE="${FT_BATCH_SIZE:-32}" FT_ACCUM="${FT_ACCUM:-4}" FT_CKPT_PATH="$OUT/ft_ckpt.pt"
echo "run_phase1_neu: FT_SEED=$FT_SEED FT_EPOCHS=$FT_EPOCHS FT_BATCH_SIZE=$FT_BATCH_SIZE FT_ACCUM=$FT_ACCUM FT_LAMBDA=${FT_LAMBDA:-0}"
mkdir -p "$OUT"
python -u feature_extractor/dino-neudet.py --output_dir "$OUT" --seed "$FT_SEED" "$@"
