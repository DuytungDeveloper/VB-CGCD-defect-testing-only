#!/usr/bin/env bash
# Lambda axis at tau 0.2: extra cells for the Eq.3 dose-response (0 and 0.25 already exist).
# 500 imgs/class, 10 epochs, batch 16x8. One item = one phase-1 run + 5 phase-2 seeds
# (k-means, n_init 10). Phase-1 seeds from $SEEDS (default 42). Resumable: finished steps skipped.
# Usage: bash scripts/queue_grid.sh
set -u
cd "$(git rev-parse --show-toplevel)"
BACKUP=$(mktemp); cp runs/inputs.txt "$BACKUP"
trap 'cp "$BACKUP" runs/inputs.txt; echo "queue: runs/inputs.txt restored"' EXIT

CELLS=("0.5 0.07")

finished() {   # $1 = exact tag -> latest run dir with exit 0, or empty
  python - "$1" << 'PY'
import glob, json, sys
best = ""
for c in sorted(glob.glob(f"runs/*__{sys.argv[1]}/config.json")):
    try:
        if json.load(open(c)).get("exit_code") == 0:
            best = c.rsplit("/", 1)[0]
    except Exception:
        pass
print(best)
PY
}

for sd in ${SEEDS:-42}; do
  for cell in "${CELLS[@]}"; do
    set -- $cell; lam=$1; tau=$2
    L=$(echo "$lam" | tr -d '.'); T=$(echo "$tau" | tr -d '.')
    ext=c100_extract_con_l${L}_t${T}_sd$sd; grp=c100_con_l${L}_t${T}_sd$sd

    r=$(finished "$ext")
    if [ -z "$r" ]; then
      echo "queue: $(date '+%m-%d %H:%M') phase 1 $ext"
      FT_LAMBDA=$lam FT_TAU=$tau FT_EPOCHS=10 FT_BATCH_SIZE=16 FT_ACCUM=8 FT_SEED=$sd \
        python scripts/exp.py run "$ext" -- bash scripts/run_phase1_extract.sh
      r=$(finished "$ext")
      [ -z "$r" ] && { echo "queue: FAILED phase 1 $ext, moving on"; continue; }
    fi
    echo "features/$(basename "$r")/dino-vitb16-sl" > runs/inputs.txt
    for s in 0 1 2 3 4; do
      [ -n "$(finished "${grp}_s$s")" ] && continue
      echo "queue: $(date '+%m-%d %H:%M') phase 2 ${grp}_s$s"
      python scripts/exp.py run "${grp}_s$s" -- bash scripts/run_phase2_cl.sh --clustering_alg kmeans --seed "$s"
    done
  done
done
echo "queue: all items processed"
