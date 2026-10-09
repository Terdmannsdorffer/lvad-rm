"""Reproduce every Phase 2 result and figure (run from the repository root). Takes about one hour."""

from __future__ import annotations

import subprocess
import sys

STEPS = [
    ["scripts/calibrate_hf.py"],  # HF patient (float64, CPU); values are copied into lvadrm/patients.py
    ["scripts/hf_window.py"],  # operating window of the HF patient
    ["scripts/validate_controller.py"],  # V6, V6b, V6c
    ["scripts/task_difficulty.py", "--ranges", "default", "--protocols"],  # baselines (GPU)
    ["scripts/task_difficulty.py", "--ranges", "wide"],
    ["scripts/phase2_report.py"],  # results/phase2/summary.md
]


def main() -> None:
    for step in STEPS:
        print("===", " ".join(step), flush=True)
        subprocess.run([sys.executable, *step], check=True)


if __name__ == "__main__":
    main()
