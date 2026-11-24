#!/usr/bin/env bash
# Phase 7, robustness of "calibrate or remember" (docs/phase7_decisions.md, section 7): queue with restarts (run inside WSL
# from the repository root). It can be started again at any time: finished work is skipped. results/phase7/PAUSE stops it
# as it does with phase7_run.sh (both queues can run at the same time; they share the GPU).
#     bash scripts/phase7_robust.sh
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
LOG=logs/p7_robust.log
PAUSE=results/phase7/PAUSE
STALE=${STALE:-5400}
exec 9>logs/p7_robust.lock
flock -n 9 || { echo "phase7_robust.sh is already running"; exit 1; }
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "START (pid $$)"

run() {
  local attempt=1 pid code
  while [ $attempt -le 5 ]; do
    if [ -f $PAUSE ]; then
      say "PAUSED, waiting for the removal of $PAUSE"
      while [ -f $PAUSE ]; do sleep 10; done
      say "RESUMED"
    fi
    say "$*"
    "$@" >> $LOG 2>&1 &
    pid=$!
    while kill -0 $pid 2>/dev/null; do
      sleep 10
      if [ $(($(date +%s) - $(stat -c %Y $LOG))) -gt "$STALE" ]; then
        say "HUNG, killed: $*"
        kill -9 $pid
      fi
    done
    wait $pid
    code=$?
    [ $code -eq 0 ] && return 0
    if [ $code -ne 75 ]; then
      say "FAILED (exit $code, attempt $attempt of 5): $*"
      attempt=$((attempt + 1))
      sleep 60
    fi
  done
  say "GAVE UP: $*"
  return 1
}
M=results/phase7/models
robust() { run python scripts/phase7_robust.py --part "$@"; }

robust mismatch shift
robust vote naive drift
for lam in 0.7 0.8 0.6 1 0.4 0.2; do
  [ -f $M/lam${lam}_N10_r2.pkl ] || run python scripts/phase7_models.py --lams $lam --skip-quality
  robust learned
done
for c in "0.9 10" "0.9 20" "0.5 10"; do
  set -- $c
  [ -f $M/lam$1_sb$2_N10_r2.pkl ] || run python scripts/phase7_models.py --lams $1 --sb $2 --skip-quality
  robust combine
done
for lam in 0.5 0.9 0; do
  [ -f $M/lam${lam}_N10_win.pkl ] || run python scripts/phase7_models.py --lams $lam --window
  robust window
done
for seed in 1 2; do
  for lam in 0.9 0.5; do
    [ -f $M/lam${lam}_N10_s${seed}_r2.pkl ] || run python scripts/phase7_models.py --lams $lam --seed $seed --skip-quality
  done
  robust seeds
done
say "QUEUE DONE"
