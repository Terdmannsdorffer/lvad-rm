#!/usr/bin/env bash
# Phase 7 compute queue with restarts (run inside WSL from the repository root). Replaces phase7_queue.sh and
# phase7_queue_ext.sh: the same stages in the same order. It can be started again at any time, after a crash, a reboot or
# a pause: finished work is skipped and a PPO run continues from its last checkpoint (about one minute of work is lost).
#     bash scripts/phase7_run.sh [PID]     wait for that process first (a stage of an older queue)
#     touch results/phase7/PAUSE           save and stop; the GPU is free when the log says PAUSED
#     rm results/phase7/PAUSE              go on
# A stage that fails is retried up to 5 times; one that writes nothing for 90 minutes is killed and retried.
set -u
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."
LOG=logs/p7_run.log
PAUSE=results/phase7/PAUSE
STALE=${STALE:-5400}
RETRY_WAIT=${RETRY_WAIT:-60}
exec 9>logs/p7_run.lock
flock -n 9 || { echo "phase7_run.sh is already running, or one of its stages is still finishing"; exit 1; }
say() { echo "[$(date '+%F %T')] $*" >> $LOG; }
say "START (pid $$)"
if [ $# -gt 0 ]; then
  say "waiting for process $1"
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi

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
      sleep "$RETRY_WAIT"
    fi
  done
  say "GAVE UP: $*"
  return 1
}
train() { run python scripts/phase7_train.py --gamma 0.95 --ent 0.003 --beats 20e6 --rv 0.3 1 3 "$@"; }

stages() {
  # main queue: seeds are the outer loop, so that a complete ladder exists early
  for seed in 0 1 2; do
    train --regime unknown --kinds true window last none true_rnn --seeds $seed
    train --regime fixed --kinds true last --seeds $seed
    run python scripts/phase7_eval.py
    for lam in 0.5 0.9 0; do
      train --regime unknown --lam $lam --kinds naive ibu tdm filter window last --seeds $seed
      run python scripts/phase7_eval.py
    done
    if [ $seed -lt 2 ]; then  # sensitivity to N (no RL)
      n=$([ $seed -eq 0 ] && echo 20 || echo 5)
      run python scripts/phase7_monitor.py --what reference --n $n
      if [ ! -f results/phase7/models/lam0.5_N${n}_r2.pkl ]; then
        run python scripts/phase7_models.py --lams 0.5 --n $n
      elif [ ! -f results/phase7/models/beliefs_lam0.5_N${n}.json ]; then
        run python scripts/phase7_models.py --lams 0.5 --n $n --eval-only
      fi
      run python scripts/phase7_monitor.py --what estimators --n $n --lam 0.5
    fi
  done
  # extension: IBU with the recurrent classifier, agents that see pressures, and a smaller ladder at N = 20
  for lam in 0.5 0.9 0; do train --regime unknown --lam $lam --kinds ibu_hist --seeds 0 1 2; done
  train --regime unknown --kinds press press_mlp --seeds 0 1 2
  run python scripts/phase7_eval.py
  for seed in 0 1 2; do
    train --regime unknown --n 20 --kinds true window last --seeds $seed
    train --regime unknown --n 20 --lam 0.5 --kinds naive ibu_hist tdm filter last --seeds $seed
    run python scripts/phase7_eval.py
  done
}

stages
say "QUEUE DONE"
