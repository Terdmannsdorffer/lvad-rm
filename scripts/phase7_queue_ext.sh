#!/usr/bin/env bash
# Phase 7 extension of the PPO queue (run inside WSL from the repository root): agents that see pressures, IBU with the
# recurrent classifier at N = 10, and a smaller ladder at N = 20. Starts when the process given as first argument (the
# main queue) has ended.
#     bash scripts/phase7_queue_ext.sh <pid>
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
LOG=logs/p7_queue_ext.log
if [ $# -gt 0 ]; then while kill -0 "$1" 2>/dev/null; do sleep 60; done; fi
run() { echo "[$(date '+%F %T')] $*" >> $LOG; "$@" >> $LOG 2>&1 || echo "FAILED: $*" >> $LOG; }
train() { run python scripts/phase7_train.py --gamma 0.95 --ent 0.003 --beats 20e6 --rv 0.3 1 3 --regime unknown "$@"; }

for lam in 0.5 0.9 0; do train --lam $lam --kinds ibu_hist --seeds 0 1 2; done
train --kinds press press_mlp --seeds 0 1 2
run python scripts/phase7_eval.py
for seed in 0 1 2; do
  train --n 20 --kinds true window last --seeds $seed
  train --n 20 --lam 0.5 --kinds naive ibu_hist tdm filter last --seeds $seed
  run python scripts/phase7_eval.py
done
echo "[$(date '+%F %T')] QUEUE DONE" >> $LOG
