#!/usr/bin/env bash
# VB-CGCD only. Always call it through exp:
#   python scripts/exp.py run baseline_s1 -- bash scripts/run_vbcgcd.sh --seed 1
# 1) runs main.py with EVERY argument written out, so console.log holds the full config
# 2) extracts numbers into $EXP_RUN_DIR/metrics.json
# Extra arguments come last and override the defaults (argparse: last value wins).
set -u
RUN_ID=$(basename "${EXP_RUN_DIR:?run this through scripts/exp.py}")

ARGS=(--dataset cifar100 --data_dir datasets/cifar100 --load_mode t5
      --pretrained_model_name dino-vitb16-sl --base 50 --increment 10 --num_classes 100
      --seed 0 --trail_name "$RUN_ID"
      --clustering_alg gmm --classifier_alg mngmm --num_dim 384
      --with_early_stop --use_correct_scaling_factor
      --n_epochs 1000 --lr 4e-6 --batch-size 128
      --scaling-factor 1.2 --early_stop_ratio 0)

echo "run_vbcgcd: python -u main.py ${ARGS[*]} $*"
python -u main.py "${ARGS[@]}" "$@"
rc=$?
python scripts/extract_vbcgcd.py "$EXP_RUN_DIR"
exit $rc