#!/usr/bin/env bash
# VB-CGCD phase 2 only. Always call it through exp:
#   python scripts/exp.py run baseline_s1 -- bash scripts/run_vbcgcd.sh --seed 1
# 1) feature folder = first line of runs/inputs.txt, e.g. features/<extract_run_id>/dino-vitb16-sl
# 2) stops if JAX is not on the GPU (silent CPU fallback, see reproducibility-issues.md A1)
# 3) runs main.py with EVERY argument written out, so console.log holds the full config
# 4) links runs/<run>/artifacts -> logs/<run>_<minute>/ (checkpoints + TensorBoard)
# 5) extracts numbers into $EXP_RUN_DIR/metrics.json
# Extra arguments come last and override the defaults (argparse: last value wins).
set -u
RUN_ID=$(basename "${EXP_RUN_DIR:?run this through scripts/exp.py}")

FEAT=$(grep -v '^[[:space:]]*#' runs/inputs.txt 2>/dev/null | grep -v '^[[:space:]]*$' | head -1)
if [ -z "$FEAT" ]; then
  echo "run_vbcgcd: ERROR runs/inputs.txt is missing or empty"; exit 2
fi
MODEL=$(basename "$FEAT")        # dino-vitb16-sl
DATA_DIR=$(dirname "$FEAT")      # features/<extract_run_id>
if [ ! -f "$FEAT/features-$MODEL.npy" ]; then
  echo "run_vbcgcd: ERROR not found: $FEAT/features-$MODEL.npy"; exit 2
fi
echo "run_vbcgcd: features = $FEAT"

DEV=$(XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; print(jax.devices())" 2>&1 | tail -1)
echo "run_vbcgcd: jax devices = $DEV"
case "$DEV" in
  *Cuda*) ;;
  *) echo "run_vbcgcd: ERROR JAX is not on the GPU"; exit 3 ;;
esac

ARGS=(--dataset cifar100 --data_dir "$DATA_DIR" --load_mode t5
      --pretrained_model_name "$MODEL" --base 50 --increment 10 --num_classes 100
      --seed 0 --trail_name "$RUN_ID"
      --clustering_alg gmm --classifier_alg mngmm --num_dim 384
      --with_early_stop --use_correct_scaling_factor
      --n_epochs 1000 --lr 4e-6 --batch-size 128
      --scaling-factor 1.2 --early_stop_ratio 0)

echo "run_vbcgcd: python -u main.py ${ARGS[*]} $*"
python -u main.py "${ARGS[@]}" "$@"
rc=$?

LOGDIR=$(ls -d "logs/${RUN_ID}_"* 2>/dev/null | head -1)
if [ -n "$LOGDIR" ]; then
  ln -sfn "../../$LOGDIR" "$EXP_RUN_DIR/artifacts"
  echo "run_vbcgcd: artifacts -> $LOGDIR"
fi

python scripts/extract_vbcgcd.py "$EXP_RUN_DIR"
exit $rc