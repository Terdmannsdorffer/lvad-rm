"""Constant-speed baselines.

(a) one fixed speed for the whole population (the calibrated operating point of the base patient);
(b) a per-episode speed that imitates a clinician's setting: the highest constant speed that still
    opens the aortic valve in the periodic regime of the episode's initial condition, reduced by a
    safety margin (assumption: 3 %).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from lvadrm.env import EnvConfig, EnvState, beat_params
from lvadrm.integrate import simulate_beats
from lvadrm.model import initial_state
from lvadrm.scenarios import VTOT

CLINICIAN_MARGIN = 0.03


@partial(jax.jit, static_argnames=("cfg", "n_bisect", "n_beats"))
def avc_speed(state: EnvState, cfg: EnvConfig, n_bisect: int = 12, n_beats: int = 20):
    """Highest constant speed in [omega_min, omega_max] with aortic valve opening in the periodic
    regime of the schedule's first row (bisection; returns omega_min if it never opens)."""
    v = state.sched[0]
    p = beat_params(cfg.patient, v)
    dtype = state.omega.dtype

    def opens(omega):
        y0 = initial_state(p, v[VTOT]).astype(dtype)
        _, stats, _ = simulate_beats(y0, p, jnp.full(n_beats, omega, dtype), n_seg=cfg.n_seg, n_inner=cfg.n_inner)
        return stats.v_aortic[-1] > cfg.eps_avo

    def body(_, lohi):
        lo, hi = lohi
        mid = 0.5 * (lo + hi)
        ok = opens(mid)
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

    lo, hi = jax.lax.fori_loop(0, n_bisect, body, (jnp.asarray(cfg.omega_min, dtype), jnp.asarray(cfg.omega_max, dtype)))
    return lo


def clinician_speed(state: EnvState, cfg: EnvConfig, margin: float = CLINICIAN_MARGIN):
    return jnp.maximum(avc_speed(state, cfg) * (1.0 - margin), cfg.omega_min)
