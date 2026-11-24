#!/usr/bin/env bash
# Phase 7 compute queue (run inside WSL from the repository root). Every stage skips what is already done;
# seeds are the outer loop, so that a complete (if noisy) ladder exists early.
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
run() { echo "[$(date '+%F %T')] $*" >> logs/p7_queue.log; "$@" >> logs/p7_queue.log 2>&1 || echo "FAILED: $*" >> logs/p7_queue.log; }
train() { run python scripts/phase7_train.py --gamma 0.95 --ent 0.003 --beats 20e6 --rv 0.3 1 3 "$@"; }

# same-controller comparison (no RL)
for lam in 0.5 0 0.9; do run python scripts/phase7_monitor.py --what estimators --n 10 --lam $lam; done
run python scripts/phase7_models.py --lams 0.5 0 0.9 --n 10 --eval-only

# PPO
for seed in 0 1 2; do
  train --regime unknown --kinds true window last none true_rnn --seeds $seed
  train --regime fixed --kinds true last --seeds $seed
  run python scripts/phase7_eval.py
  for lam in 0.5 0.9 0; do
    train --regime unknown --lam $lam --kinds naive ibu tdm filter window last --seeds $seed
    run python scripts/phase7_eval.py
  done
  # sensitivity to N (no RL), between seeds
  if [ $seed -lt 2 ]; then
    n=$([ $seed -eq 0 ] && echo 20 || echo 5)
    run python scripts/phase7_monitor.py --what reference --n $n
    [ -f results/phase7/models/lam0.5_N${n}_r2.pkl ] || run python scripts/phase7_models.py --lams 0.5 --n $n
    run python scripts/phase7_monitor.py --what estimators --n $n --lam 0.5
  fi
done
echo "[$(date '+%F %T')] QUEUE DONE" >> logs/p7_queue.log
