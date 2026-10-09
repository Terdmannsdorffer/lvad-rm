"""Minimum-pump-flow slope controller of Simaan et al. (2009), Section V, per-beat version.

The paper extracts the minimum pump flow of each cardiac cycle, fits a least-squares line to the
minima inside a moving 2.5 s window (3 minima at HR 60) and updates the speed every 0.005 s:
omega(k+1) = omega(k) + c * slope, with c = 0.76 rpm/(ml/s).

The slope can only change when a new cycle minimum becomes available, i.e. once per beat. Between
two minima the paper's update adds the same increment every 0.005 s, which is a staircase ramp
of rate c * slope / 0.005 rpm/s. ``speed_command`` returns that ramp for the next beat, to be run
with ``omega_hold = 0.005`` (exact staircase) by the integrator. The only approximation is that a
minimum is used from the end of its beat on, instead of from the instant it occurs.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

C_GAIN = 0.76  # rpm / (ml/s), as published
# The published constants give an effective gain K = c / dt = 152 rpm per ml/s, and the loop is a
# positive feedback (higher speed -> higher minimum flow -> positive slope) with loop gain
# K * dQmin/domega ~ 152 * 0.031 ~ 4.7 in our model: it runs away. Fig. 16 implies K ~ 29
# (1650 rpm for a 57 ml/s rise of the minimum). C_GAIN_FIG16 gives K = 30 (validate_controller.py).
C_GAIN_FIG16 = 0.15
# Even with K = 30 the loop collapses after entering suction (V6b). The paper notes that c must stay
# below the pump's slew rate; with a slew limit of 250 rpm/s (our assumption) the controller follows
# the suction speed. This stabilized version is the baseline used with the RL agents.
SLEW_RPM_S = 250.0
DT_UPDATE = 0.005  # s
WINDOW_S = 2.5  # s
BUFFER = 8


class MinFlowState(NamedTuple):
    t_min: jnp.ndarray  # [BUFFER] absolute times of past cycle minima (-inf if empty)
    q_min: jnp.ndarray  # [BUFFER] past cycle minima of the pump flow [ml/s]


def init(dtype=jnp.float32) -> MinFlowState:
    return MinFlowState(jnp.full(BUFFER, -jnp.inf, dtype), jnp.zeros(BUFFER, dtype))


def push_minimum(cs: MinFlowState, t_abs, q) -> MinFlowState:
    return MinFlowState(jnp.roll(cs.t_min, 1).at[0].set(t_abs), jnp.roll(cs.q_min, 1).at[0].set(q))


def slope(cs: MinFlowState, t_now, window: float = WINDOW_S):
    """Least-squares slope [ml/s per s] of the minima in (t_now - window, t_now]; 0 if fewer than 2."""
    w = (cs.t_min > t_now - window).astype(cs.q_min.dtype)
    n = w.sum()
    t = jnp.where(w > 0, cs.t_min, 0.0)
    tm = (w * t).sum() / jnp.maximum(n, 1.0)
    qm = (w * cs.q_min).sum() / jnp.maximum(n, 1.0)
    sxx = (w * (t - tm) ** 2).sum()
    sxy = (w * (t - tm) * (cs.q_min - qm)).sum()
    return jnp.where((n >= 2) & (sxx > 0), sxy / jnp.where(sxx > 0, sxx, 1.0), 0.0)


def speed_command(cs: MinFlowState, t_now, c: float = C_GAIN, dt_update: float = DT_UPDATE, slew=None):
    """Speed ramp rate [rpm/s] for the next beat (apply with omega_hold = dt_update), optionally
    limited to +-slew rpm/s."""
    rate = c * slope(cs, t_now) / dt_update
    return rate if slew is None else jnp.clip(rate, -slew, slew)


DP_REF = 75.0  # mmHg
DP_GAIN_RPM = 0.004 * 60.0 / (2.0 * jnp.pi)  # 0.004 rad/s per mmHg per update, in rpm


def deltap_rate(aop_mean, lvp_mean, dp_ref: float = DP_REF, dt_update: float = DT_UPDATE):
    """Constant-Delta-p controller (Giridharan and Skliar 2003), per-beat version: speed ramp rate
    [rpm/s] from the mean aorta-LV pressure difference of the last beat (privileged: uses pressures)."""
    return DP_GAIN_RPM * (dp_ref - (aop_mean - lvp_mean)) / dt_update
