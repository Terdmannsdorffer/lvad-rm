"""Intermittent speed-modulation baselines: tune on validation scenarios, evaluate on the test set (GPU).

Families (lvadrm/baselines/modulation.py): open-loop periodic speed drops, and a closed-loop rule
with the MLP AVO detector in the loop (plus the same rule with the true AVO label as an upper bound).
Every configuration of a grid is evaluated in one batch on the same 128 validation scenarios
(seed 777, clinician's initial speed); the best configuration of each family by RM return (N = 5,
r_back = -0.1) is then evaluated exactly like the agents (2048 test scenarios, seed 12345, and the
three protocols) and merged into results/phase6/eval.json under mod_open_loop, mod_rule_mlp and
mod_rule_oracle. Writes results/phase6/modulation_tuning.json.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.agents.belief import load_detector
from lvadrm.baselines.modulation import OpenLoopParams, RuleParams, as_policy, open_loop_fns, rule_fns
from lvadrm.env import EnvConfig, reset_with_scenario, step_speed
from lvadrm.evaluation import INFO_KEYS, initial_speeds, metrics, rollout
from lvadrm.patients import HF_PATIENT
from lvadrm.scenarios import ScenarioConfig, protocol_batch, sample_scenario

P6 = Path("results/phase6")
N_VAL, VAL_SEED, TEST_SEED, N_TEST = 128, 777, 12345, 2048
GRIDS = {
    "open_loop": (OpenLoopParams, {"base": [-0.03, 0.0, 0.05, 0.10], "drop": [0.05, 0.10, 0.15, 0.20], "period": [4, 5]}),
    "rule": (RuleParams, {"base": [0.0, 0.05, 0.10], "trigger": [2, 3], "step": [0.03, 0.05, 0.10], "adapt": [0.0, 0.01], "up_after": [20]}),
}


def configs(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*grid.values())]


def batched_rollout(fns, prm, scns, cfg, key, omega0):
    """Rollout where every environment has its own parameter set (leading axis of ``prm``)."""
    init, act, observe = fns
    n = scns.base.shape[0]
    k_reset, k_init, k_roll = jax.random.split(key, 3)
    obs, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(k_reset, n), scns, cfg, omega0)

    def one(p, carry, k):
        st, obs, ps = carry
        ka, ks = jax.random.split(k)
        omega, rate, hold, ps = act(p, ka, st, obs, ps)
        obs2, st2, r, _, info = step_speed(ks, st, omega, cfg, rate, hold)
        return (st2, obs2, observe(p, ps, st, info, obs2)), (r, {k_: info[k_] for k_ in INFO_KEYS})

    @jax.jit
    def run(prm, st, obs):
        ps = jax.vmap(init)(prm, jax.random.split(k_init, n), st, obs)
        step_v = jax.vmap(one)
        return jax.lax.scan(lambda c, kk: step_v(prm, c, jax.random.split(kk, n)), (st, obs, ps), jax.random.split(k_roll, cfg.episode_beats))[1]

    r, info = run(prm, st, obs)
    out = {k: np.asarray(v) for k, v in info.items()}
    out["reward"] = np.asarray(r)
    return out


def tune(family, fns, cfg, scns_val, omega_val, key):
    cls, grid = GRIDS[family]
    cfgs = configs(grid)
    C = len(cfgs)
    prm = cls(**{f: jnp.repeat(jnp.asarray([c[f] for c in cfgs], jnp.float32 if f not in ("period", "trigger", "up_after") else jnp.int32), N_VAL)
                 for f in cls._fields})
    scns = jax.tree.map(lambda a: jnp.tile(a, (C,) + (1,) * (a.ndim - 1)), scns_val)
    o = batched_rollout(fns, prm, scns, cfg, key, jnp.tile(omega_val, C))
    rows = []
    for i, c in enumerate(cfgs):
        sl = slice(i * N_VAL, (i + 1) * N_VAL)
        m = metrics({k: v[:, sl] for k, v in o.items()}, cfg)
        rows.append({"config": c, **{k: m[k] for k in ("return_mean_N5", "N5_violations_per_100_beats", "avo_frac", "back_frac", "co_mean")}})
    best = max(rows, key=lambda r: r["return_mean_N5"])
    print(f"{family}: best {best['config']} return {best['return_mean_N5']:.2f} viol {best['N5_violations_per_100_beats']:.2f} "
          f"AVO {best['avo_frac']:.2f} CO {best['co_mean']:.2f}", flush=True)
    return best, rows


def scalar_params(cls, c: dict):
    return cls(**{f: jnp.asarray(c[f], jnp.int32 if f in ("period", "trigger", "up_after") else jnp.float32) for f in cls._fields})


def main() -> None:
    cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300)
    det = load_detector("mlp")
    fams = {"open_loop": open_loop_fns(), "rule": rule_fns(det)}

    k_scn, k_run = jax.random.split(jax.random.PRNGKey(VAL_SEED))
    scns_val = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, ScenarioConfig(), 300))(jax.random.split(k_scn, N_VAL))
    omega_val = initial_speeds("const_clinician", scns_val, cfg, k_run)
    tuning, best = {}, {}
    for fam, fns in fams.items():
        best[fam], rows = tune(fam, fns, cfg, scns_val, omega_val, k_run)
        tuning[fam] = {"best": best[fam], "all": rows}
    common.save_json("modulation_tuning.json", tuning, phase=6)

    # test set and protocols, exactly like eval_agents.py
    k_scn, k_run = jax.random.split(jax.random.PRNGKey(TEST_SEED))
    scns = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, ScenarioConfig(), 300))(jax.random.split(k_scn, N_TEST))
    names, proto_scns = protocol_batch(HF_PATIENT)
    cfg_long = dataclasses.replace(cfg, episode_beats=1206)
    w_test = initial_speeds("const_clinician", scns, cfg, k_run)
    w_proto = initial_speeds("const_clinician", proto_scns, cfg_long, k_run)
    policies = {
        "mod_open_loop": as_policy(fams["open_loop"], scalar_params(OpenLoopParams, best["open_loop"]["config"])),
        "mod_rule_mlp": as_policy(fams["rule"], scalar_params(RuleParams, best["rule"]["config"])),
        "mod_rule_oracle": as_policy(rule_fns(None), scalar_params(RuleParams, best["rule"]["config"])),
    }
    ev = json.loads((P6 / "eval.json").read_text())
    for label, pol in policies.items():
        o = rollout(pol, scns, cfg, k_run, w_test)
        ev["random_scenarios"][label] = metrics(o, cfg)
        ol = rollout(pol, proto_scns, cfg_long, k_run, w_proto)
        for i, proto in enumerate(names):
            oi = {k: (v[:, i : i + 1] if v.ndim >= 2 else v[i : i + 1]) for k, v in ol.items()}
            ev["protocols"][proto][label] = metrics(oi, cfg_long)
        m = ev["random_scenarios"][label]
        print(f"{label:16s} test: viol {m['N5_violations_per_100_beats']:.3f}/100, episodes {m['N5_episodes_with_violation']:.0%}, "
              f"AVO {m['avo_frac']:.2f}, BACK {m['back_frac']:.3f}, CO {m['co_mean']:.2f}, return {m['return_mean_N5']:.1f} | "
              + " ".join(f"{p} {ev['protocols'][p][label]['N5_violations_per_100_beats']:.2f}" for p in names), flush=True)
    (P6 / "eval.json").write_text(json.dumps(ev, indent=2))
    print("merged into results/phase6/eval.json")


if __name__ == "__main__":
    main()
