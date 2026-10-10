"""Phase 6 data for the TDM: wide scenarios and on-policy data (GPU, float32).

Sources (each 512 / 64 / 128 episodes for train / val / test, 300 beats):
  wide_random       random actions, wide scenarios (Rs up to 2.5 x the patient's, beyond training)
  wide_oracle       stochastic RM oracle (Phase 5, seed 0), wide scenarios
  onpolicy_default  stochastic TDM agent (Phase 5, seed 0), default scenarios
  onpolicy_wide     stochastic TDM agent, wide scenarios
The Phase 4 data (default scenarios, random + oracle policies) is reused as is.

Writes results/phase6/data/{split}_{source}.npz (not tracked by git).
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eval_agents import load_agent
from lvadrm.agents.ppo import greedy_policy
from lvadrm.env import EnvConfig, reset_with_scenario, step_speed
from lvadrm.evaluation import Policy, classical_policy
from lvadrm.patients import HF_PATIENT
from lvadrm.scenarios import WIDE_SCENARIOS, ScenarioConfig, sample_scenario

OUT = Path("results/phase6/data")
RUNS = Path("results/phase5/runs")
SOURCES = {
    "wide_random": ("random", WIDE_SCENARIOS),
    "wide_oracle": ("oracle_s0", WIDE_SCENARIOS),
    "onpolicy_default": ("tdm_mlp_s0", ScenarioConfig()),
    "onpolicy_wide": ("tdm_mlp_s0", WIDE_SCENARIOS),
}
COUNTS = {"train": 512, "val": 64, "test": 128}
SEEDS = {"train": 1000, "val": 2000, "test": 3000}


def make_policy(name: str, cfg: EnvConfig) -> Policy:
    if name == "random":
        return classical_policy("random", cfg)
    spec, _, net, params, hidden, belief = load_agent(RUNS / name)
    return greedy_policy(spec, cfg, net, params, hidden, stochastic=True, belief=belief)


def collect(policy: Policy, scns, cfg: EnvConfig, key) -> dict[str, np.ndarray]:
    n = scns.base.shape[0]
    k_w, k_reset, k_init, k_roll = jax.random.split(key, 4)
    omega0 = cfg.patient.omega * jax.random.uniform(k_w, (n,), minval=0.9, maxval=1.1)
    obs, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(k_reset, n), scns, cfg, omega0)

    def one_step(carry, k):
        st, obs, ps = carry
        ka, ks = jax.random.split(k)
        omega, rate, hold, ps = policy.act(ka, st, obs, ps)
        obs2, st2, _, _, info = step_speed(ks, st, omega, cfg, rate, hold)
        rec = {"obs": obs2, "wave": info["qp_wave"], "labels": info["labels"], "u_next": info["u_next"],
               "v_aortic": info["v_aortic"], "omega": info["omega"]}
        return (st2, obs2, policy.observe(ps, st, info, obs2)), rec

    @jax.jit
    def run(st, obs):
        ps = jax.vmap(policy.init)(jax.random.split(k_init, n), st, obs)
        step_v = jax.vmap(one_step)
        return jax.lax.scan(lambda c, kk: step_v(c, jax.random.split(kk, n)), (st, obs, ps), jax.random.split(k_roll, cfg.episode_beats))[1]

    rec = run(st, obs)
    return {k: np.swapaxes(np.asarray(v), 0, 1) for k, v in rec.items()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source, (pol_name, scn_cfg) in SOURCES.items():
        cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300, scenario=scn_cfg)
        policy = make_policy(pol_name, cfg)
        for split, n in COUNTS.items():
            path = OUT / f"{split}_{source}.npz"
            if path.exists():
                continue
            t0 = time.perf_counter()
            k_scn, k_run = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(SEEDS[split]), list(SOURCES).index(source)))
            scns = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, scn_cfg, 300))(jax.random.split(k_scn, n))
            data = collect(policy, scns, cfg, k_run)
            data["labels"] = data["labels"].astype(bool)
            data["u_next"] = data["u_next"].astype(np.int8)
            data["policy"] = np.full(n, list(SOURCES).index(source) + 2, dtype=np.int8)
            np.savez_compressed(path, **data)
            print(f"{source:17s} {split:5s}: {n} episodes, AVO {data['labels'][..., 0].mean():.3f}, {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
