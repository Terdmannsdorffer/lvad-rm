#!/usr/bin/env bash
# Phase 7: same-controller comparison with the ramp-test nominal speed (run inside WSL from the repository root).
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
run() { echo "[$(date '+%F %T')] $*" >> logs/p7_queue_ramp.log; "$@" >> logs/p7_queue_ramp.log 2>&1 || echo "FAILED: $*" >> logs/p7_queue_ramp.log; }
run python scripts/phase7_monitor.py --what reference --n 10 --nominal ramp
for lam in 0.5 0 0.9; do run python scripts/phase7_monitor.py --what estimators --n 10 --lam $lam --nominal ramp; done
echo "RAMP DONE" >> logs/p7_queue_ramp.log
