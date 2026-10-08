"""Reproduce every Phase 1 result and figure (run from the repository root).

Each step runs in its own process: the validation scripts use float64 on CPU, the
benchmark uses float32 on the GPU.
"""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    "scripts/validate_simaan.py",  # V1, V2, V3, V9, forms, literal pump
    "scripts/ramp_suction.py",  # V4, V5
    "scripts/aortic_closure.py",  # V8 and operating window
    "scripts/sweep_dt.py",  # step size vs Radau
    "scripts/bench_gpu.py",  # GPU throughput, float32 accuracy
    "scripts/phase1_report.py",  # results/phase1/summary.md
]


def main() -> None:
    for step in STEPS:
        print(f"=== {step}", flush=True)
        subprocess.run([sys.executable, step], check=True)


if __name__ == "__main__":
    main()
