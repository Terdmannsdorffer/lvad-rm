#!/usr/bin/env bash
# Phase 7: second validation, after the ramp comparison has finished (run inside WSL from the repository root).
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
until grep -q "RAMP DONE" logs/p7_queue_ramp.log; do sleep 30; done
python scripts/phase7_revalidate.py >> logs/p7_reval.log 2>&1
echo "REVAL DONE" >> logs/p7_reval.log
