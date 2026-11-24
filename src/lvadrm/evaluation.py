"""Roll out speed-control policies on batches of scenarios and compute clinical metrics.

A policy is a triple of pure functions (vmapped over environments by ``rollout``):
    init(key, state, obs) -> pstate
    act(key, state, obs, pstate) -> (omega0, omega_rate, omega_hold, pstate)
    observe(pstate, state, info, obs_next) -> pstate   (after the beat; e.g. store a minimum, update a belief)
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.baselines import simaan_minflow as smf
from lvadrm.baselines.constant import CLINICIAN_MARGIN, avc_speed
from lvadrm.env import EnvConfig, reset_with_scenario, step_speed

INFO_KEYS = ("labels", "violation", "omega", "co", "qlv", "v_back", "lvp_min", "u", "tc", "lap_mean", "co_target")
N_LIST = (3, 5, 10)
SKIP_BEATS = 20


class Policy(NamedTuple):
    init: Callable
    act: Callable
    observe: Callable


def _no_observe(ps, state, info, obs_next):
    return ps


def classical_policy(name: str, cfg: EnvConfig) -> Policy:
    """const_pop, const_clinician (speed set at reset), simaan_minflow or random."""
    acts = jnp.asarray(cfg.actions)

    def init(key, state, obs):
        return smf.init(state.omega.dtype)

    if name in ("const_pop", "const_clinician"):
        return Policy(init, lambda key, st, obs, ps: (st.omega, 0.0, 0.0, ps), _no_observe)
    if name == "simaan_minflow":

        def act(key, st, obs, ps):
            rate = smf.speed_command(ps, st.t, c=smf.C_GAIN_FIG16, slew=smf.SLEW_RPM_S)
            return st.omega, rate, smf.DT_UPDATE, ps

        def observe(ps, st, info, obs_next):
            return smf.push_minimum(ps, st.t + info["t_qp_min"], info["qp_min"])

        return Policy(init, act, observe)
    if name == "random":

        def act(key, st, obs, ps):
            a = jax.random.randint(key, (), 0, len(cfg.actions))
            return st.omega * (1.0 + acts[a]), 0.0, 0.0, ps

        return Policy(init, act, _no_observe)
    raise ValueError(name)


def initial_speeds(name: str, scns, cfg: EnvConfig, key):
    """Initial speed per environment: the base operating point, or the clinician's setting."""
    n = scns.base.shape[0]
    omega0 = jnp.full(n, cfg.patient.omega)
    if name == "const_clinician":
        _, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(key, n), scns, cfg, omega0)
        omega0 = jnp.maximum(jax.vmap(avc_speed, in_axes=(0, None))(st, cfg) * (1.0 - CLINICIAN_MARGIN), cfg.omega_min)
    return omega0


def rollout(policy: Policy, scns, cfg: EnvConfig, key, omega0) -> dict[str, np.ndarray]:
    """Run ``cfg.episode_beats`` beats for a batch of scenarios. Returns info arrays [T, n] and rewards."""
    n = scns.base.shape[0]
    k_reset, k_roll, k_init = jax.random.split(key, 3)
    obs, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(k_reset, n), scns, cfg, omega0)

    def one_step(carry, k):
        st, obs, ps = carry
        ka, ks = jax.random.split(k)
        omega, rate, hold, ps = policy.act(ka, st, obs, ps)
        obs2, st2, r, _, info = step_speed(ks, st, omega, cfg, rate, hold)
        return (st2, obs2, policy.observe(ps, st, info, obs2)), (r, {k_: info[k_] for k_ in INFO_KEYS})

    @jax.jit
    def run(st, obs, k):
        ps = jax.vmap(policy.init)(jax.random.split(k_init, n), st, obs)
        step_v = jax.vmap(one_step)

        def body(carry, kk):
            return step_v(carry, jax.random.split(kk, n))

        _, (r, info) = jax.lax.scan(body, (st, obs, ps), jax.random.split(k, cfg.episode_beats))
        return r, info

    r, info = run(st, obs, k_roll)
    out = {k: np.asarray(v) for k, v in info.items()}
    out["reward"] = np.asarray(r)
    out["omega0"] = np.asarray(omega0)
    return out


def closure_runs(avo: np.ndarray, tc: np.ndarray):
    """Durations [s] of the runs of consecutive beats without aortic valve opening, per episode.
    avo, tc: [T, n_env]. Returns a list (one per episode) of arrays of run durations."""
    out = []
    for e in range(avo.shape[1]):
        closed = ~avo[:, e]
        edges = np.diff(np.concatenate([[0], closed.astype(np.int8), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        csum = np.concatenate([[0.0], np.cumsum(tc[:, e])])
        out.append(csum[ends] - csum[starts])
    return out


def clinical_metrics(o: dict, cfg: EnvConfig, lap_congestion: float = 20.0) -> dict[str, float]:
    """Time-based clinical metrics (Phase 6.5): opening ratio, closure durations, suction events per hour,
    congestion and actuator activity. Episode means; ``o`` from ``rollout`` (with tc, lap_mean, co_target)."""
    lab, tc = o["labels"], o["tc"]
    avo, suc, back = lab[..., 0], lab[..., 1], lab[..., 2]
    total_s = tc.sum(axis=0)  # per episode
    runs = closure_runs(avo, tc)
    max_closure = np.array([r.max() if len(r) else 0.0 for r in runs])
    m = {
        "opening_ratio": float(avo.mean()),
        "max_closure_s": float(max_closure.mean()),
        "max_closure_s_p95": float(np.percentile(max_closure, 95)),
    }
    for x in (10, 30, 60):
        frac = np.array([r[r > x].sum() for r in runs]) / total_s
        m[f"time_closed_gt{x}s"] = float(frac.mean())
        m[f"episodes_closed_gt{x}s"] = float((max_closure > x).mean())
    onsets = (suc[1:] & ~suc[:-1]).sum(axis=0) + suc[0]
    m["suction_per_hour"] = float((onsets / total_s * 3600.0).mean())
    m["suction_frac"] = float(suc.mean())
    m["back_frac"] = float(back.mean())
    m["congestion_frac"] = float((o["lap_mean"] > lap_congestion).mean())
    m["co_mean"] = float(o["co"].mean())
    m["co_abs_err"] = float(np.abs(o["co"] - o["co_target"]).mean())
    dw = np.abs(np.diff(o["omega"], axis=0)) / o["omega"][:-1]
    m["speed_changes_per_min"] = float(((dw > 0.005).sum(axis=0) / total_s * 60.0).mean())
    v = count_violations(avo, cfg.rm.n)
    m[f"violations_per_hour_N{cfg.rm.n}"] = float((v.sum(axis=0) / total_s * 3600.0).mean())
    m[f"return_mean_N{cfg.rm.n}"] = float(o["reward"].sum(axis=0).mean())
    return m


def count_violations(avo: np.ndarray, n: int) -> np.ndarray:
    """Violations of the counting RM with N = n for AVO sequences [T, n_env]."""
    u = np.zeros(avo.shape[1], dtype=int)
    viol = np.zeros(avo.shape, dtype=bool)
    for t in range(avo.shape[0]):
        v = ~avo[t] & (u == n - 1)
        viol[t] = v
        u = np.where(avo[t] | v, 0, u + 1)
    return viol


def metrics(o: dict, cfg: EnvConfig, n_list=N_LIST) -> dict[str, float]:
    lab = o["labels"]  # [T, n, 4]
    avo = lab[..., 0]
    m = {
        "avo_frac": float(avo.mean()),
        "suc_frac": float(lab[..., 1].mean()),
        "back_frac": float(lab[..., 2].mean()),
        "low_frac": float(lab[..., 3].mean()),
        "episodes_with_suction": float((lab[..., 1].any(axis=0)).mean()),
        "co_mean": float(o["co"].mean()),
        "co_abs_err_mean": float(np.abs(o["co"] - cfg.rm.co_target).mean()),
        "omega_mean": float(o["omega"].mean()),
        f"return_mean_N{cfg.rm.n}": float(o["reward"].sum(axis=0).mean()),
        f"return_std_N{cfg.rm.n}": float(o["reward"].sum(axis=0).std()),
    }
    for n in n_list:
        v = count_violations(avo, n)
        m[f"N{n}_episodes_with_violation"] = float(v.any(axis=0).mean())
        m[f"N{n}_violations_per_100_beats"] = float(100.0 * v.mean())
        # excluding the initial transient (every episode starts at the base patient's speed)
        m[f"N{n}_episodes_with_violation_after{SKIP_BEATS}"] = float(v[SKIP_BEATS:].any(axis=0).mean())
        m[f"N{n}_violations_per_100_beats_after{SKIP_BEATS}"] = float(100.0 * v[SKIP_BEATS:].mean())
    return m
