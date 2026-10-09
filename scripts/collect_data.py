"""Collect labelled beats for the AVO detectors and the TDM (GPU, float32).

Half of the episodes use uniformly random actions (broad coverage, many violations) and half the
trained RM oracle with stochastic actions (states near the operating region). Each episode has 300
beats from a random scenario (default ranges). Per beat we store what the agent can observe after the
beat (observation, measured pump-flow waveform, action) and the ground truth (AVO and other labels,
RM state after the beat, ejected volume).

Writes results/phase4/data/{train,val,test}.npz (not tracked by git).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lvadrm.agents.ppo import PPOConfig, make_network
from lvadrm.agents.specs import SPECS, agent_obs
from lvadrm.env import EnvConfig, reset, step
from lvadrm.patients import HF_PATIENT

OUT = Path("results/phase4/data")
ORACLE_RUN = Path("results/phase3/runs/oracle_s0")
SPLITS = {"train": (1024, 1024, 100), "val": (128, 128, 200), "test": (256, 256, 300)}  # (random, oracle, seed)


def load_oracle(cfg):
    spec = SPECS["oracle"]
    conf = json.loads((ORACLE_RUN / "config.json").read_text())
    net = make_network(spec, PPOConfig(hidden=conf["ppo"]["hidden"]))
    obs, st = reset(jax.random.PRNGKey(0), cfg)
    template = net.init(jax.random.PRNGKey(0), agent_obs(spec, obs, st, cfg))
    return spec, net, serialization.from_bytes(template, (ORACLE_RUN / "params.msgpack").read_bytes())


def collect(policy: str, n: int, key, cfg, oracle):
    spec, net, params = oracle
    k_reset, k_roll = jax.random.split(key)
    obs, st = jax.vmap(reset, in_axes=(0, None))(jax.random.split(k_reset, n), cfg)
    acts = jnp.asarray(cfg.actions)
    step_v = jax.vmap(step, in_axes=(0, 0, 0, None))

    @jax.jit
    def run(obs, st, key):
        def body(carry, k):
            obs, st = carry
            ka, ks = jax.random.split(k)
            if policy == "random":
                a = jax.random.randint(ka, (n,), 0, len(cfg.actions))
            else:
                logits, _ = net.apply(params, agent_obs(spec, obs, st, cfg))
                a = jax.random.categorical(ka, logits)
            obs2, st2, _, _, info = step_v(jax.random.split(ks, n), st, a, cfg)
            rec = {"obs": obs2, "wave": info["qp_wave"], "labels": info["labels"], "u_next": info["u_next"],
                   "v_aortic": info["v_aortic"], "omega": info["omega"], "action": acts[a]}
            return (obs2, st2), rec

        return jax.lax.scan(body, (obs, st), jax.random.split(key, cfg.episode_beats))[1]

    rec = run(obs, st, k_roll)
    return {k: np.swapaxes(np.asarray(v), 0, 1) for k, v in rec.items()}  # [n, T, ...]


def main() -> None:
    cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300)
    oracle = load_oracle(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    for split, (n_rand, n_orc, seed) in SPLITS.items():
        t0 = time.perf_counter()
        k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
        parts = [collect("random", n_rand, k1, cfg, oracle), collect("oracle", n_orc, k2, cfg, oracle)]
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        data["policy"] = np.array([0] * n_rand + [1] * n_orc, dtype=np.int8)
        data["labels"] = data["labels"].astype(bool)
        data["u_next"] = data["u_next"].astype(np.int8)
        np.savez_compressed(OUT / f"{split}.npz", **data)
        avo = data["labels"][..., 0]
        print(f"{split}: {avo.shape[0]} episodes x {avo.shape[1]} beats, AVO {avo.mean():.3f} "
              f"(random {avo[:n_rand].mean():.3f}, oracle {avo[n_rand:].mean():.3f}), {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
