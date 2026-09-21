#!/usr/bin/env bash
# 8-vs-8 test of Eq. 3: lambda 0 vs 0.25 (tau 0.2), 500 imgs/class, 10 epochs, batch 16x8.
# One item = one phase-1 run (FT_SEED) + 5 phase-2 seeds (k-means, n_init 10).
# Arms are interleaved. Finished steps are skipped, so re-run this script after any crash.
set -u
cd "$(git rev-parse --show-toplevel)"
BACKUP=$(mktemp); cp runs/inputs.txt "$BACKUP"
trap 'cp "$BACKUP" runs/inputs.txt; echo "queue: runs/inputs.txt restored"' EXIT

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

ITEMS=("0 45" "0.25 43" "0 46" "0.25 44" "0 47" "0.25 45" "0 48" "0.25 46" "0 49" "0.25 47" "0.25 48" "0.25 49")

for it in "${ITEMS[@]}"; do
  set -- $it; lam=$1; sd=$2
  if [ "$lam" = "0" ]; then ext=c100_extract_ft16_sd$sd; grp=c100_ft16sd$sd
  else                      ext=c100_extract_con_sd$sd;  grp=c100_consd$sd; fi

  r=$(finished "$ext")
  if [ -z "$r" ]; then
    echo "queue: $(date +%H:%M) phase 1 $ext (lambda $lam, FT_SEED $sd)"
    FT_LAMBDA=$lam FT_TAU=0.2 FT_EPOCHS=10 FT_BATCH_SIZE=16 FT_ACCUM=8 FT_SEED=$sd \
      python scripts/exp.py run "$ext" -- bash scripts/run_phase1_extract.sh
    r=$(finished "$ext")
    [ -z "$r" ] && { echo "queue: FAILED phase 1 $ext, moving on"; continue; }
  fi

  echo "features/$(basename "$r")/dino-vitb16-sl" > runs/inputs.txt
  for s in 0 1 2 3 4; do
    [ -n "$(finished "${grp}_s$s")" ] && continue
    echo "queue: $(date +%H:%M) phase 2 ${grp}_s$s"
    python scripts/exp.py run "${grp}_s$s" -- bash scripts/run_phase2_cl.sh --clustering_alg kmeans --seed "$s"
  done
done
echo "queue: all items processed"
