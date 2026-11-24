"""Shared pieces of the Phase 7 scripts: rollouts of parameterized controllers, data, metrics, saved models."""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.env import reset_with_scenario, step_speed
from lvadrm.estimators import FilterModel, LambdaDetector
from lvadrm.scenarios import sample_scenario

OUT = Path("results/phase7")
SKIP = 30  # beats discarded at the start of every evaluation episode (common initial speed)
PAUSE = OUT / "PAUSE"  # while this file exists, the stages save their state and exit (scripts/phase7_run.sh waits)
EXIT_PAUSED = 75


def write_atomic(path, data):
    """Write to a temporary file and rename it, so that an interruption never leaves a truncated file."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode())
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # the file is open on the Windows side
            time.sleep(0.5)
    os.replace(tmp, path)


def pause_point():
    """Leave (exit code EXIT_PAUSED) if a pause was requested; call it where everything finished is on disk."""
    if PAUSE.exists():
        print("PAUSED", flush=True)
        raise SystemExit(EXIT_PAUSED)


def scenarios(key, cfg, n, beats=None):
    beats = cfg.episode_beats if beats is None else beats
    return jax.vmap(lambda k: sample_scenario(k, cfg.patient, cfg.scenario, beats))(jax.random.split(key, n))


def start_speeds(key, cfg, n):
    """Common start: the base speed of the population +- 5 % (it must not reveal the patient's threshold)."""
    return cfg.patient.omega * jax.random.uniform(key, (n,), minval=0.95, maxval=1.05)


def rollout(fns, prm, scns, cfg, key, omega0, keep=("avo", "back", "co", "omega", "tc"), belief_vec=None):
    """Roll out (init, act, observe) with one parameter set and one scenario per environment."""
    init, act, observe = fns
    n = scns.base.shape[0]
    k_reset, k_roll, k_init = jax.random.split(key, 3)
    obs, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(k_reset, n), scns, cfg, omega0)

    def one_step(carry, k, p):
        st, obs, ps = carry
        ka, ks = jax.random.split(k)
        omega, rate, hold, ps = act(p, ka, st, obs, ps)
        obs2, st2, r, _, info = step_speed(ks, st, omega, cfg, rate, hold)
        ps2 = observe(p, ps, st, info, obs2)
        rec = {"avo": info["labels"][0], "suc": info["labels"][1], "back": info["labels"][2], "co": info["co"], "omega": info["omega"],
               "tc": info["tc"], "t_open": info["t_open"], "u_next": info["u_next"], "obs": obs2, "params": info["params"],
               "reward": r, "qp": info["qp_wave"], "viol": info["violation"]}
        rec = {k_: rec[k_] for k_ in keep}
        if belief_vec is not None:
            rec["belief"] = belief_vec(ps2)
        return (st2, obs2, ps2), rec

    @jax.jit
    def run(st, obs, k):
        ps = jax.vmap(init)(prm, jax.random.split(k_init, n), st, obs)

        def body(carry, kk):
            return jax.vmap(one_step)(carry, jax.random.split(kk, n), prm)

        return jax.lax.scan(body, (st, obs, ps), jax.random.split(k, cfg.episode_beats))[1]

    return {k: np.asarray(v) for k, v in run(st, obs, k_roll).items()}


def violations(avo, n):
    """Saturating counter: True in every closed beat that follows at least n - 1 closed beats. avo: [T, E]."""
    u = np.zeros(avo.shape[1], int)
    v = np.zeros(avo.shape, bool)
    for t in range(avo.shape[0]):
        v[t] = ~avo[t] & (u >= n - 1)
        u = np.where(avo[t], 0, np.minimum(u + 1, n - 1))
    return v


def outcome(o, n, co_target=5.0, skip=SKIP):
    """Per-episode outcomes of the two axes and the side effects (arrays [E])."""
    v = violations(o["avo"], n)[skip:]
    co = o["co"][skip:]
    return dict(viol=v.mean(0), dev=np.abs(co - co_target).mean(0) / co_target, co=co.mean(0), open=o["avo"][skip:].mean(0),
                back=o["back"][skip:].mean(0), omega=o["omega"][skip:].mean(0))


def auc(score, y):
    order = np.argsort(score, kind="mergesort")
    s, yy = np.asarray(score)[order], np.asarray(y)[order]
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    n1, n0 = yy.sum(), (~yy).sum()
    return float((ranks[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_calibration():
    return json.loads((OUT / "calibration.json").read_text())


def detector(lam, cal=None, perfect=False):
    cal = load_calibration() if cal is None else cal
    if perfect:
        return LambdaDetector(perfect=True, p_pos=1.0, p_neg=0.0)
    return LambdaDetector(tau=cal["tau"], sigma=cal["sigma"], lam=float(lam), p_pos=cal["p_pos"], p_neg=cal["p_neg"])


def load_filter(cal=None):
    cal = load_calibration() if cal is None else cal
    f = cal["filter"]
    return FilterModel(**{k: (np.asarray(v) if isinstance(v, list) else v) for k, v in f.items()})


def model_path(lam, n, rnd=2, sb=None, seed=0):
    """``sb``: residual SD [ms] of the bias of a calibrated detector (None: not calibrated); ``seed``: training seed."""
    return OUT / "models" / f"lam{lam:g}{'' if sb is None else f'_sb{sb:g}'}_N{n}{f'_s{seed}' if seed else ''}_r{rnd}.pkl"


def window_path(lam, n, sb=None, seed=0):
    p = model_path(lam, n, 2, sb, seed)
    return p.with_name(p.name.replace("_r2", "_win"))


def load_models(lam, n, rnd=2, sb=None, seed=0):
    m = pickle.loads(model_path(lam, n, rnd, sb, seed).read_bytes())
    if window_path(lam, n, sb, seed).exists():
        m |= pickle.loads(window_path(lam, n, sb, seed).read_bytes())
    return m | {"filter": load_filter()}
