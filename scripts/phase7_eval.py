"""Phase 7, stage 6: evaluate the PPO agents on the test scenarios of the monitor controllers (GPU).

Greedy policies, 512 scenarios x 400 beats, same scenarios and noise seeds for every agent. Agents that share
the estimator (regime, lambda, kind, N) are evaluated in one batch.

    python scripts/phase7_eval.py [--runs runs]
Writes results/phase7/agents.json (one entry per run; finished entries are kept).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from phase7_common import OUT, outcome, pause_point, rollout, scenarios, start_speeds, write_atomic
from phase7_monitor import BEATS, N_TEST, TEST_SEED
from phase7_train import belief_for, spec_for

from lvadrm.agents.ppo import PPOConfig, make_network
from lvadrm.agents.specs import agent_obs, env_config_for
from lvadrm.env import reset
from lvadrm.presets import env_v3

HIDDEN = PPOConfig().hidden


def agent_fns(spec, cfg, net, belief):
    acts = jnp.asarray(cfg.actions)

    def init(params, key, st, obs):
        k1, k2 = jax.random.split(key)
        return jnp.zeros(HIDDEN), (None if belief is None else belief.init(k1)), k2

    def act(params, key, st, obs, ps):
        h, bs, kd = ps
        x = agent_obs(spec, obs, st, cfg, None if belief is None else belief.vector(bs))
        if spec.recurrent:
            h, logits, _ = net.apply(params, h, x)
        else:
            logits, _ = net.apply(params, x)
        return st.omega * (1.0 + acts[jnp.argmax(logits)]), 0.0, 0.0, (h, bs, kd)

    def observe(params, ps, st, info, obs2):
        h, bs, kd = ps
        if belief is not None:
            kd, k = jax.random.split(kd)
            bs = belief.update(k, bs, obs2, info)
        return h, bs, kd

    return init, act, observe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--name", default="agents")
    ap.add_argument("--batch", type=int, default=9)
    a = ap.parse_args()
    path = OUT / f"{a.name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    groups = defaultdict(list)
    for d in sorted((OUT / a.runs).glob("*/config.json")):
        if d.parent.name not in res and (d.parent / "params.msgpack").exists():
            c = json.loads(d.read_text())
            groups[(c["regime"], c["lam"], c["kind"], c["n"])].append((d.parent, c))
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(TEST_SEED), 3)
    for (regime, lam, kind, n), runs in groups.items():
        spec, belief = spec_for(kind), belief_for(kind, lam, n)
        cfg = env_config_for(spec, env_v3(n, regime, episode_beats=BEATS))
        net = make_network(spec, PPOConfig())
        obs, st = reset(jax.random.PRNGKey(0), cfg)
        x = agent_obs(spec, obs, st, cfg, None if belief is None else belief.vector(belief.init(jax.random.PRNGKey(0))))
        tmpl = net.init(jax.random.PRNGKey(0), jnp.zeros(HIDDEN), x) if spec.recurrent else net.init(jax.random.PRNGKey(0), x)
        scn, w0 = scenarios(k1, cfg, N_TEST), start_speeds(k2, cfg, N_TEST)
        for i in range(0, len(runs), a.batch):
            pause_point()
            t0 = time.perf_counter()
            part = runs[i : i + a.batch]
            ps = [serialization.from_bytes(tmpl, (Path(d) / "params.msgpack").read_bytes()) for d, _ in part]
            m = len(ps)
            prm = jax.tree.map(lambda *z: jnp.repeat(jnp.stack([jnp.asarray(q) for q in z]), N_TEST, 0), *ps)
            s = jax.tree.map(lambda q: jnp.tile(q, (m,) + (1,) * (q.ndim - 1)), scn)  # noqa: B023
            o = outcome(rollout(agent_fns(spec, cfg, net, belief), prm, s, cfg, k3, jnp.tile(w0, m)), n)
            for j, (d, c) in enumerate(part):
                e = {k: v.reshape(m, N_TEST)[j] for k, v in o.items()}
                w5 = float(np.sort(e["viol"])[-N_TEST // 20 :].mean())
                res[d.name] = c | {k: float(v.mean()) for k, v in e.items()} | {"viol_worst5": w5, "viol_episodes": e["viol"].round(4).tolist()}
                print(f"{d.name:44s} viol {100 * res[d.name]['viol']:6.2f}%  dev {res[d.name]['dev']:.3f}  CO {res[d.name]['co']:.2f}  open {res[d.name]['open']:.2f}  "
                      f"back {res[d.name]['back']:.3f}  worst5 {100 * w5:.0f}%", flush=True)
            write_atomic(path, json.dumps(res))
            print(f"  [{time.perf_counter() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
