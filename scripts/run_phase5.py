"""Phase 5, main experiment: every agent with 5 seeds, evaluation and aggregation (GPU, ~8 h).

Resumable: runs whose params.msgpack already exists are skipped.

    python scripts/run_phase5.py [--seeds 0 1 2 3 4] [--eval-only]

Agents: RM oracle, pressure oracle, recurrent PPO without RM, and Naive / IBU / TDM beliefs from the
learned MLP AVO detector (Phase 4). Reward: base RM with N = 5 and r_back = -0.1.
Writes results/phase5/{runs/, eval.json, summary.md, aggregate.json, figures}.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RUNS = Path("results/phase5/runs")
AGENTS = [("oracle", None), ("oracle_p", None), ("rnn", None), ("naive", "mlp"), ("ibu", "mlp"), ("tdm", "mlp")]


def run_name(agent: str, det: str | None, seed: int) -> str:
    return f"{agent}{'' if det is None else '_' + det}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    dirs = []
    for seed in args.seeds:
        for agent, det in AGENTS:
            d = RUNS / run_name(agent, det, seed)
            dirs.append(str(d))
            if args.eval_only or (d / "params.msgpack").exists():
                continue
            cmd = ["scripts/train_ppo.py", "--agent", agent, "--seed", str(seed), "--out-dir", str(RUNS)]
            if det:
                cmd += ["--detector", det]
            print("=== train", " ".join(cmd[1:]), flush=True)
            subprocess.run([sys.executable, *cmd], check=True)
    print("=== evaluate", flush=True)
    subprocess.run([sys.executable, "scripts/eval_agents.py", "--phase", "5", "--dirs", *dirs], check=True)
    subprocess.run([sys.executable, "scripts/phase5_aggregate.py"], check=True)


if __name__ == "__main__":
    main()
