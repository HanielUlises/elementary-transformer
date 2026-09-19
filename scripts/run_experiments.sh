#!/usr/bin/env bash
# Reproduces the learning experiments reported in the README.
#
#   scripts/run_experiments.sh [DATA_DIR] [RUNS_DIR]
#
# Data generation runs on the CPU; training uses the default JAX device.
set -euo pipefail

DATA="${1:-data/k3q4}"
RUNS="${2:-runs/k3q4}"
SEEDS="${SEEDS:-0 1 2}"
WORKERS="${WORKERS:-8}"

if [ ! -f "$DATA/manifest.json" ]; then
    elementary-transformer generate --out "$DATA" --k 3 --q 4 \
        --train 20000 --val 2000 --test 1000 --workers "$WORKERS" --seed 0 --lean-verify
fi

mkdir -p "$RUNS"
[ -f "$RUNS/baselines.json" ] || elementary-transformer baselines --data "$DATA" --out "$RUNS/baselines.json" > /dev/null

for seed in $SEEDS; do
    [ -f "$RUNS/seed$seed/results.json" ] || \
        elementary-transformer train --data "$DATA" --out "$RUNS/seed$seed" --seed "$seed" | tee "$RUNS/seed$seed.log"
done

elementary-transformer report $(for seed in $SEEDS; do echo "$RUNS/seed$seed"; done) > "$RUNS/report.json"
echo "report written to $RUNS/report.json"
