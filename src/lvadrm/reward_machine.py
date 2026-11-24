"""Counting reward machine for periodic aortic valve opening (HANDOFF section 9).

States u_0 ... u_{N-1} count the beats since the last aortic valve opening (AVO):

    AVO                      -> u_0
    no AVO in u_k, k < N-1   -> u_{k+1}
    no AVO in u_{N-1}        -> violation: reward r_viol and back to u_0 (or terminal)
    SUC (any state)          -> r_suc, state unchanged
    BACK (any state)         -> r_back, state unchanged
    every beat (Markovian)   -> -w_perf |CO - CO_target| / CO_target - w_act |d omega| / omega

Propositions are encoded as bits of an integer sigma in [0, 16): AVO = 1, SUC = 2, BACK = 4,
LOW = 8. ``transition_table`` / ``violation_table`` give delta_u and the violation flag for
every (u, sigma), as needed by belief updates (IBU) in Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

PROPS = ("avo", "suc", "back", "low")
N_SIGMA = 2 ** len(PROPS)


@dataclass(frozen=True)
class RMConfig:
    n: int = 5  # beats allowed without an aortic valve opening
    r_viol: float = -1.0
    r_suc: float = -2.0
    r_back: float = -0.1  # was -0.5 in Phases 3-4: avoiding mild regurgitation outweighed avoiding violations
    w_perf: float = 0.1
    w_act: float = 0.01
    co_target: float = 5.0  # [l/min], Fernandez-Zapico et al. 2024
    terminal_on_violation: bool = False
    # Phase 7: stay in u_{N-1} after a violation and pay r_viol in every further closed beat. With the reset
    # variant a valve that never opens pays only r_viol / N per beat, so N changes constraint and price at once.
    saturate: bool = False


class RewardComponents(NamedTuple):
    viol: jax.Array
    suc: jax.Array
    back: jax.Array
    perf: jax.Array
    act: jax.Array

    def total(self):
        return self.viol + self.suc + self.back + self.perf + self.act


def sigma_from_labels(avo, suc, back, low):
    """Integer encoding of the propositions true in a beat."""
    return avo.astype(jnp.int32) + 2 * suc.astype(jnp.int32) + 4 * back.astype(jnp.int32) + 8 * low.astype(jnp.int32)


def rm_transition(u, avo, n: int, saturate: bool = False):
    """delta_u for the counting RM. Returns (u_next, violation)."""
    violation = (~avo) & (u == n - 1)
    if saturate:
        u_next = jnp.where(avo, 0, jnp.minimum(u + 1, n - 1))
    else:
        u_next = jnp.where(avo | violation, 0, u + 1)
    return u_next, violation


def rm_rewards(u, avo, suc, back, co, omega_prev, omega, cfg: RMConfig, co_target=None):
    """Next RM state and reward components for one beat. ``co_target`` overrides the fixed
    cfg.co_target (e.g. a Starling-like target that depends on the preload, Phase 6.5)."""
    u_next, violation = rm_transition(u, avo, cfg.n, cfg.saturate)
    dtype = jnp.result_type(co)
    target = cfg.co_target if co_target is None else co_target
    comps = RewardComponents(
        viol=cfg.r_viol * violation.astype(dtype),
        suc=cfg.r_suc * suc.astype(dtype),
        back=cfg.r_back * back.astype(dtype),
        perf=-cfg.w_perf * jnp.abs(co - target) / target,
        act=-cfg.w_act * jnp.abs(omega - omega_prev) / omega_prev,
    )
    return u_next, violation, comps


def transition_table(n: int) -> np.ndarray:
    """next_u[u, sigma] for all RM states and proposition combinations, shape [n, 16]."""
    u = np.arange(n)[:, None]
    avo = (np.arange(N_SIGMA)[None, :] & 1).astype(bool)
    violation = ~avo & (u == n - 1)
    return np.where(avo | violation, 0, u + 1).astype(np.int32)


def violation_table(n: int) -> np.ndarray:
    """violation[u, sigma], shape [n, 16]."""
    u = np.arange(n)[:, None]
    avo = (np.arange(N_SIGMA)[None, :] & 1).astype(bool)
    return ~avo & (u == n - 1)
