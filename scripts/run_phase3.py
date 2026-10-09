"""Reproduce Phase 3 (run from the repository root): train the agents and evaluate them (GPU).

About 11 minutes per agent and seed with 30 M training beats on an RTX 3090.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

AGENTS = ("oracle", "oracle_p", "rnn", "mlp", "fz", "fz4")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--agents", nargs="+", default=list(AGENTS))
    args = ap.parse_args()
    for seed in args.seeds:
        for agent in args.agents:
            print(f"=== train {agent} seed {seed}", flush=True)
            subprocess.run([sys.executable, "scripts/train_ppo.py", "--agent", agent, "--seed", str(seed)], check=True)
    subprocess.run([sys.executable, "scripts/eval_agents.py"], check=True)


if __name__ == "__main__":
    main()
