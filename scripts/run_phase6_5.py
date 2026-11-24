"""Phase 6.5 pipeline (docs/phase6_5_plan.md), resumable: data, detector, inference models, agents.

    python scripts/run_phase6_5.py            # after the calibrations of scripts/phase6_5.py

Order: random-policy data -> oracle seed 0 -> oracle data -> per-beat detector (v2_cnn) -> inference
models (TDM, TDM-prop, IBU-hist) -> agent queue. Agents (env v2, N = 3, PPO without value clipping):
seeds 0-4 of oracle, ibu, tdm, rnn, ibu_hist, tdm_det; seeds 5-9 of oracle, ibu, tdm, rnn; then
seeds 0-4 of naive and oracle_p. Runs in results/phase6_5/runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_phase6 import py, train  # noqa: E402

OUT = Path("results/phase6_5")
RUNS = OUT / "runs"
DET = "v2_mlp"  # the best per-beat detector on the pipeline test data (AUC 0.922 vs 0.894 for the CNN)
N = int(json.loads((OUT / "calibration.json").read_text())["chosen"].get("n", 3))
EXTRA = ("--no-vf-clip", "--env", "v2")
TDM = {"tdm": OUT / "tdm" / f"{DET}_tdm_N{N}.msgpack", "tdm_det": OUT / "tdm" / f"{DET}_tdm_det_N{N}.msgpack",
       "ibu_hist": OUT / "tdm" / f"{DET}_ibu_hist.msgpack"}


def agent(name: str, seed: int) -> str:
    if name in ("oracle", "oracle_p", "rnn"):
        return train(name, seed, RUNS, rm_n=N, extra=EXTRA)
    return train(name, seed, RUNS, DET, str(TDM[name]) if name in TDM else None, rm_n=N, extra=EXTRA)


def main() -> None:
    cal = json.loads((OUT / "calibration.json").read_text())["chosen"]
    missing = {"starling", "vtot_lo", "flow_noise_scale"} - set(cal)
    assert not missing, f"calibration incomplete: {missing}"
    data = OUT / "data"
    if not (data / "test_random.npz").exists():
        py("scripts/phase6_5.py", "collect", "--source", "random")
    agent("oracle", 0)
    if not (data / "test_oracle_s0.npz").exists():
        py("scripts/phase6_5.py", "collect", "--source", "oracle_s0")
    if not (OUT / "detectors" / f"{DET}.msgpack").exists():
        py("scripts/phase6_5.py", "train-detector")
    if not all(p.exists() for p in TDM.values()):
        py("scripts/phase6_5.py", "train-inference", "--detector", DET, "--n", str(N))
    for s in range(5):
        for a in ("oracle", "ibu", "tdm", "rnn", "ibu_hist", "tdm_det"):
            agent(a, s)
    for s in range(5, 10):
        for a in ("oracle", "ibu", "tdm", "rnn"):
            agent(a, s)
    for s in range(5):
        for a in ("naive", "oracle_p"):
            agent(a, s)


if __name__ == "__main__":
    main()
