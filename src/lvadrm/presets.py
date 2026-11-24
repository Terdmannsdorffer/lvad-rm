"""Environment presets. ``v2``: the Phase 6.5 environment (docs/phase6_5_plan.md).

v2 differs from the Phases 3-6 environment in four ways, each an explicit EnvConfig option:
  - estimated pump flow as observation (obs_model = "estimated", flow_noise_scale calibrated so that
    the best learned AVO detector is clinical-level);
  - preload-sensitive (Starling-like) cardiac-output target in the reward;
  - RM horizon N = 3 (guarantees the clinical 1:3 aortic valve opening ratio);
  - training scenarios with Rs up to 1.6 x the patient's and Vtot down to 0.80 x (the wider ranges
    of the plan made most episodes infeasible; reward probe in docs/phase6_5_decisions.md).
Calibrated values live in results/phase6_5/calibration.json (written by scripts/phase6_5.py).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lvadrm.env import EnvConfig
from lvadrm.patients import HF_PATIENT
from lvadrm.reward_machine import RMConfig
from lvadrm.scenarios import ScenarioConfig

CALIBRATION = Path("results/phase6_5/calibration.json")
V2_DEFAULTS = {"flow_noise_scale": 1.0, "starling": (5.0, 10.0, 0.25, 3.5, 8.0), "vtot_lo": 0.80, "rs_hi": 1.6, "n": 3, "w_perf": 0.1, "r_back": -0.1}


def v2_calibration() -> dict:
    cal = dict(V2_DEFAULTS)
    if CALIBRATION.exists():
        cal.update(json.loads(CALIBRATION.read_text()).get("chosen", {}))
    return cal


def v2_scenarios(vtot_lo: float | None = None, rs_hi: float | None = None) -> ScenarioConfig:
    """Training scenarios: Rs (0.44, rs_hi) x the patient's, Vtot (vtot_lo, 1.10) x; the rest as in Phases 3-6.
    (Rs up to 2.5 x, WIDE_SCENARIOS, made most episodes infeasible: see docs/phase6_5_decisions.md.)"""
    cal = v2_calibration()
    lo = cal["vtot_lo"] if vtot_lo is None else vtot_lo
    hi = cal["rs_hi"] if rs_hi is None else rs_hi
    return ScenarioConfig(rs_scale=(0.4 / 0.9, hi), vtot_scale=(lo, 1.10))


def env_v2(episode_beats: int = 300, n: int | None = None, **overrides) -> EnvConfig:
    """``n`` defaults to the calibrated RM horizon; w_perf comes from the calibration (reward probe)."""
    cal = v2_calibration()
    cfg = EnvConfig(
        patient=HF_PATIENT,
        scenario=v2_scenarios(cal["vtot_lo"], cal["rs_hi"]),
        rm=RMConfig(n=int(cal["n"]) if n is None else n, w_perf=float(cal["w_perf"]), r_back=float(cal["r_back"])),
        episode_beats=episode_beats,
        obs_model="estimated",
        flow_noise_scale=float(cal["flow_noise_scale"]),
        perf_target="starling",
        starling=tuple(float(x) for x in cal["starling"]),
    )
    return replace(cfg, **overrides)


def env_for_preset(preset: str | None, episode_beats: int = 300, n: int = 5) -> EnvConfig:
    """The environment of a run's config.json: None / "v1" (Phases 3-6) or "v2" (Phase 6.5)."""
    if preset in (None, "v1"):
        return EnvConfig(patient=HF_PATIENT, episode_beats=episode_beats, rm=RMConfig(n=n))
    if preset == "v2":
        return env_v2(episode_beats=episode_beats, n=n)
    raise ValueError(f"unknown env preset {preset!r}")
