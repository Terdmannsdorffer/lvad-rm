"""Reproduce Phase 4 (run from the repository root, GPU): data, detectors, inference, belief agents.

    python scripts/run_phase4.py                 # everything (~4 h)
    python scripts/run_phase4.py --agents-only   # reuse data, detectors and TDMs
"""

from __future__ import annotations

import argparse
import subprocess
import sys

BELIEF_RUNS = [
    ("naive", "mlp"), ("ibu", "mlp"), ("tdm", "mlp"),
    ("naive", "logistic"), ("ibu", "logistic"), ("tdm", "logistic"),
    ("naive", "markov_e0.2_r0.9"), ("ibu", "markov_e0.2_r0.9"), ("tdm_det", "markov_e0.2_r0.9"), ("tdm", "markov_e0.2_r0.9"),
]
REFERENCE_RUNS = ["results/phase3/runs/oracle_s0", "results/phase3/runs/oracle_p_s0", "results/phase3/runs/rnn_s0"]


def run(*cmd):
    print("===", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not (args.agents_only or args.eval_only):
        run("scripts/collect_data.py")
        run("scripts/train_detectors.py")
        run("scripts/train_tdm.py")
    if not args.eval_only:
        for agent, det in BELIEF_RUNS:
            run("scripts/train_ppo.py", "--agent", agent, "--detector", det, "--seed", str(args.seed))
    dirs = REFERENCE_RUNS + [f"results/phase4/runs/{a}_{d}_s{args.seed}" for a, d in BELIEF_RUNS]
    run("scripts/eval_agents.py", "--phase", "4", "--dirs", *dirs)


if __name__ == "__main__":
    main()
