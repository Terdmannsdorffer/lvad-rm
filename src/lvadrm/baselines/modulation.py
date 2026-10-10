"""Intermittent speed-modulation baselines for periodic aortic valve opening.

open_loop   periodic speed drops: the speed is base_speed * (1 - drop) for one beat every `period`
            beats and base_speed otherwise (inspired by the speed-modulation features of commercial
            pumps, which lower the speed periodically without knowing the valve state).
rule        closed loop with an AVO detector: after `trigger` beats without a detected opening the
            speed is lowered by `step` per beat until an opening is detected, then it returns to the
            base speed. The base adapts slowly: it is lowered by `adapt` when an opening had to be
            forced and raised by `adapt` after `up_after` consecutive spontaneous openings.

base_speed starts at the speed given at reset (the clinician's setting in the evaluations) times
(1 + base). Parameters are arrays so that many configurations can be evaluated in one batch.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from lvadrm.detectors import Detector
from lvadrm.evaluation import Policy


class OpenLoopParams(NamedTuple):
    base: jax.Array
    drop: jax.Array
    period: jax.Array


class RuleParams(NamedTuple):
    base: jax.Array
    trigger: jax.Array
    step: jax.Array
    adapt: jax.Array
    up_after: jax.Array


class OLState(NamedTuple):
    k: jax.Array
    omega_b: jax.Array


class RuleState(NamedTuple):
    omega_b: jax.Array
    since: jax.Array  # beats since the last detected opening
    streak: jax.Array  # consecutive detected openings
    dropping: jax.Array  # whether the last command was a forced drop
    det: jax.Array | tuple  # detector state
    key: jax.Array


def open_loop_fns():
    def init(prm: OpenLoopParams, key, st, obs):
        return OLState(jnp.int32(0), st.omega * (1.0 + prm.base))

    def act(prm: OpenLoopParams, key, st, obs, ps: OLState):
        drop_now = ps.k % prm.period == prm.period - 1
        omega = jnp.where(drop_now, ps.omega_b * (1.0 - prm.drop), ps.omega_b)
        return omega, 0.0, 0.0, ps._replace(k=ps.k + 1)

    def observe(prm, ps, st, info, obs_next):
        return ps

    return init, act, observe


def rule_fns(detector: Detector | None):
    """``detector=None`` uses the true AVO label (an upper bound for the rule)."""

    def init(prm: RuleParams, key, st, obs):
        k_det, k = jax.random.split(key)
        det = detector.init(k_det) if detector is not None else jnp.zeros(())
        return RuleState(st.omega * (1.0 + prm.base), jnp.int32(0), jnp.int32(0), jnp.asarray(False), det, k)

    def act(prm: RuleParams, key, st, obs, ps: RuleState):
        dropping = ps.since >= prm.trigger
        omega = jnp.where(dropping, st.omega * (1.0 - prm.step), ps.omega_b)
        return omega, 0.0, 0.0, ps._replace(dropping=dropping)

    def observe(prm: RuleParams, ps: RuleState, st, info, obs_next):
        key, k = jax.random.split(ps.key)
        if detector is None:
            opened, det = info["labels"][0], ps.det
        else:
            _, opened, det = detector.step(k, ps.det, obs_next, info)
        forced = opened & ps.dropping
        omega_b = ps.omega_b * jnp.where(forced, 1.0 - prm.adapt, 1.0)
        streak = jnp.where(opened & ~ps.dropping, ps.streak + 1, 0)
        raise_now = streak >= prm.up_after
        omega_b = omega_b * jnp.where(raise_now, 1.0 + prm.adapt, 1.0)
        streak = jnp.where(raise_now, 0, streak)
        since = jnp.where(opened, 0, ps.since + 1)
        return RuleState(omega_b, since, streak, ps.dropping, det, key)

    return init, act, observe


def as_policy(fns, prm) -> Policy:
    """Bind one parameter set to get a ``lvadrm.evaluation.Policy``."""
    init, act, observe = fns
    return Policy(
        lambda key, st, obs: init(prm, key, st, obs),
        lambda key, st, obs, ps: act(prm, key, st, obs, ps),
        lambda ps, st, info, obs_next: observe(prm, ps, st, info, obs_next),
    )
