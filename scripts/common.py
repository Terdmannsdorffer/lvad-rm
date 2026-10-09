"""Shared settings for the Phase 1 scripts (run from the repository root)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RESULTS = Path("results/phase1")


def results_dir(phase: int = 1) -> Path:
    return Path(f"results/phase{phase}")

# Validation-grade discretization: N_sub = N_SEG * N_INNER substeps per beat.
N_SEG = 64
N_INNER = 160

# Heart used in the LVAD simulations of Simaan Figs. 12-17. HR = 60 is inferred from
# Fig. 16(b) (30 beats in 30 s). Emin = 0.06 (with its own Vtot fitted to Fig. 5) gives the
# best match of Fig. 15 in ramp_suction.py; Emax = 2, V0 = 10.
PUMP_HR = 60.0
PUMP_EMIN = 0.06


def pump_vtot() -> float:
    """Vtot calibrated on Simaan Fig. 5 (healthy, HR 75, no pump) for Emin = PUMP_EMIN."""
    from lvadrm.calibration import calibrate_healthy_vtot

    return calibrate_healthy_vtot(emin=PUMP_EMIN, n_seg=N_SEG, n_inner=N_INNER)[0]


def setup_cpu_x64() -> None:
    """Float64 on CPU for validation. Must run before JAX is first used."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax

    jax.config.update("jax_enable_x64", True)


def save_json(name: str, data: dict, phase: int = 1) -> Path:
    results_dir(phase).mkdir(parents=True, exist_ok=True)
    path = results_dir(phase) / name

    def default(o):
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(data, indent=2, default=default))
    return path


def load_json(name: str, phase: int = 1) -> dict:
    return json.loads((results_dir(phase) / name).read_text())


def savefig(fig, name: str, phase: int = 1) -> Path:
    results_dir(phase).mkdir(parents=True, exist_ok=True)
    path = results_dir(phase) / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
