#!/usr/bin/env bash
# Phase 7, non-RL part of the queue: wait for the given process ids, then run the same-controller comparison.
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
for pid in "$@"; do while [ -d "/proc/$pid" ]; do sleep 15; done; done
run() { echo "[$(date '+%F %T')] $*" >> logs/p7_queue.log; "$@" >> logs/p7_queue.log 2>&1 || echo "FAILED: $*" >> logs/p7_queue.log; }
run python scripts/phase7_models.py --lams 0 0.5 0.9 --n 10 --eval-only
for lam in 0.5 0 0.9; do run python scripts/phase7_monitor.py --what estimators --n 10 --lam $lam; done
echo "[$(date '+%F %T')] NON-RL QUEUE DONE" >> logs/p7_queue.log
