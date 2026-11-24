"""Simplex-style runtime assurance for learned speed controllers (Phase 6.5, block B4).

A decision module switches control from the learned policy (advanced controller) to a classical
fallback (baseline controller) and back (Sha 2001; Phan et al. 2020, Neural Simplex Architecture).
Triggers, from signals available at runtime:
  ood      the observation is far from the training data: Mahalanobis distance of the pump-flow
           features above a threshold (a quantile of the training distances) for ``k_ood`` beats;
  belief   the belief over the RM state is uncertain: entropy above a threshold (belief agents);
Once triggered, the fallback stays in control for at least ``dwell`` beats and until the triggers
clear. Both controllers keep their own state up to date at every beat, so the switch is bumpless
for the fallback (e.g. open-loop modulation around the clinician's speed set at reset).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from lvadrm.env import PUMP_FLOW_FEATURES
from lvadrm.evaluation import Policy


class OODModel(NamedTuple):
    mean: jnp.ndarray
    prec: jnp.ndarray  # inverse covariance
    threshold: float


def fit_ood(obs: np.ndarray, quantile: float = 0.995) -> OODModel:
    """Gaussian model of the pump-flow features (first PUMP_FLOW_FEATURES entries of the observation)."""
    x = obs.reshape(-1, obs.shape[-1])[:, :PUMP_FLOW_FEATURES].astype(np.float64)
    mu = x.mean(0)
    cov = np.cov(x, rowvar=False) + 1e-6 * np.eye(x.shape[1])
    prec = np.linalg.inv(cov)
    d = np.einsum("ij,jk,ik->i", x - mu, prec, x - mu)
    return OODModel(jnp.asarray(mu, jnp.float32), jnp.asarray(prec, jnp.float32), float(np.quantile(d, quantile)))


def mahalanobis(model: OODModel, obs):
    z = obs[..., :PUMP_FLOW_FEATURES] - model.mean
    return jnp.einsum("...i,ij,...j->...", z, model.prec, z)


class SimplexState(NamedTuple):
    agent: Any
    fallback: Any
    mode: jnp.ndarray  # 0 learned policy, 1 fallback
    dwell: jnp.ndarray  # beats left in fallback
    ood_run: jnp.ndarray  # consecutive out-of-distribution beats
    switches: jnp.ndarray  # number of hand-overs to the fallback
    fb_beats: jnp.ndarray  # beats under the fallback


def simplex_policy(agent: Policy, fallback: Policy, ood: OODModel | None = None, k_ood: int = 3,
                   belief_vec=None, entropy_thr: float | None = None, dwell: int = 30) -> Policy:
    """``belief_vec(agent_pstate) -> [N]`` gives the agent's belief (for the entropy trigger)."""

    def init(key, st, obs):
        z = jnp.int32(0)
        return SimplexState(agent.init(key, st, obs), fallback.init(key, st, obs), z, z, z, z, z)

    def act(key, st, obs, ps: SimplexState):
        wa, ra, ha, pa = agent.act(key, st, obs, ps.agent)
        wf, rf, hf, pf = fallback.act(key, st, obs, ps.fallback)
        use_fb = ps.mode == 1
        omega = jnp.where(use_fb, wf, wa)
        rate = jnp.where(use_fb, rf, ra)
        hold = jnp.where(use_fb, hf, ha)
        return omega, rate, hold, ps._replace(agent=pa, fallback=pf)

    def observe(ps: SimplexState, st, info, obs_next):
        pa = agent.observe(ps.agent, st, info, obs_next)
        pf = fallback.observe(ps.fallback, st, info, obs_next)
        trig = jnp.bool_(False)
        ood_run = ps.ood_run
        if ood is not None:
            ood_run = jnp.where(mahalanobis(ood, obs_next) > ood.threshold, ps.ood_run + 1, 0)
            trig = trig | (ood_run >= k_ood)
        if belief_vec is not None and entropy_thr is not None:
            b = belief_vec(pa)
            ent = -(b * jnp.log(jnp.clip(b, 1e-8, 1.0))).sum()
            trig = trig | (ent > entropy_thr)
        enter = trig & (ps.mode == 0)
        dwell_left = jnp.where(trig, dwell, jnp.maximum(ps.dwell - 1, 0))
        mode = jnp.where(trig | (dwell_left > 0), 1, 0).astype(jnp.int32)
        return SimplexState(pa, pf, mode, dwell_left.astype(jnp.int32), ood_run.astype(jnp.int32),
                            ps.switches + enter.astype(jnp.int32), ps.fb_beats + (ps.mode == 1).astype(jnp.int32))

    return Policy(init, act, observe)
