"""Phase 6.5: full-length (30 M beats) oracle probe for one v2 variant; prints the learning curve.

    python scripts/p65_probe_full.py --name s2_b01 --scale 2.0 --r-back -0.1 --n 3

The oracle with noise scale 4 drifted after ~10 M beats from a good regime (return -0.034, 10 %
backflow) to a low-speed regime (return -0.116, 86 % backflow, 4.0 l/min). This probe checks which
setting avoids the drift. Appends to results/phase6_5/probe_full.json.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from lvadrm.agents.ppo import PPOConfig, make_ppo
from lvadrm.agents.specs import SPECS, env_config_for
from lvadrm.presets import env_v2

OUT = Path("results/phase6_5/probe_full.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--scale", type=float, required=True)
    ap.add_argument("--r-back", type=float, default=-0.1)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    spec = SPECS["oracle"]
    base = env_v2(n=args.n, flow_noise_scale=args.scale)
    cfg_env = env_config_for(spec, replace(base, rm=replace(base.rm, r_back=args.r_back)))
    cfg = PPOConfig(vf_clip=None)
    init, update, _ = make_ppo(spec, cfg_env, cfg)
    ts = init(jax.random.PRNGKey(args.seed))
    t0, log, curve = time.perf_counter(), [], []
    for i in range(cfg.n_updates):
        ts, m = update(ts)
        log.append({k: float(v) for k, v in m.items()})
        if i % 12 == 11 or i == cfg.n_updates - 1:
            last = log[-12:]
            r = {k: float(np.mean([x[k] for x in last])) for k in ("reward", "violations_per_100", "avo", "back", "co")}
            r["beats_M"] = (i + 1) * cfg.n_envs * cfg.n_steps / 1e6
            curve.append(r)
            print(f"{args.name} {r['beats_M']:5.1f}M reward {r['reward']:+.4f} viol {r['violations_per_100']:.2f} AVO {r['avo']:.3f} "
                  f"BACK {r['back']:.3f} CO {r['co']:.2f} [{time.perf_counter() - t0:.0f} s]", flush=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    res[args.name] = {"args": vars(args), "curve": curve}
    OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
