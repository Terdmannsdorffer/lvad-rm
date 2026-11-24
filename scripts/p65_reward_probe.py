"""Phase 6.5: short oracle runs to choose the RM weights of the v2 environment (8 M beats each).

    python scripts/p65_reward_probe.py

The first v2 oracle (N = 3, w_perf = 0.1) settled at ~86 % opening, ~50 % backflow and 3.5 l/min:
with N = 3 a violation costs up to 1/3 per beat, far more than the output deficit (~0.03), so in the
wide hypertensive scenarios the agent sacrificed perfusion to keep the valve opening. This probe
compares N in {3, 5} and w_perf in {0.1, 1.0}. Writes results/phase6_5/reward_probe.json.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from lvadrm.agents.ppo import PPOConfig, make_ppo
from lvadrm.agents.specs import SPECS, env_config_for
from lvadrm.presets import env_v2

OUT = Path("results/phase6_5/reward_probe.json")
VARIANTS = [(3, 0.1), (3, 1.0), (5, 0.1), (5, 1.0)]
# second round: scenario distributions (name, rs_scale upper bound x patient, vtot lower bound, w_perf), N = 3
SCENARIO_VARIANTS = [("default", 1.4 / 0.9, 0.85, 0.1), ("rs2.0", 2.0, 0.85, 0.1), ("rs2.0_w0.3", 2.0, 0.85, 0.3), ("rs1.6_v0.80", 1.6, 0.80, 0.1)]


def main() -> None:
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    spec = SPECS["oracle"]
    runs = [(f"N{n}_w{w}", n, w, None) for n, w in VARIANTS]
    runs += [(f"N3_{name}", 3, w, (rs_hi, v_lo)) for name, rs_hi, v_lo, w in SCENARIO_VARIANTS]
    for key, n, w, scn in runs:
        if key in res:
            continue
        base = env_v2(n=n)
        if scn is not None:
            base = replace(base, scenario=replace(base.scenario, rs_scale=(0.4 / 0.9, scn[0]), vtot_scale=(scn[1], 1.10)))
        cfg_env = env_config_for(spec, replace(base, rm=replace(base.rm, w_perf=w)))
        cfg = PPOConfig(total_beats=8_000_000, vf_clip=None)
        init, update, _ = make_ppo(spec, cfg_env, cfg)
        ts = init(jax.random.PRNGKey(0))
        t0 = time.perf_counter()
        log = []
        for _ in range(cfg.n_updates):
            ts, m = update(ts)
            log.append({k: float(v) for k, v in m.items()})
        last = log[-6:]
        r = {k: float(np.mean([x[k] for x in last])) for k in ("reward", "violations_per_100", "avo", "back", "low", "co", "omega")}
        res[key] = r
        print(f"{key:10s} reward {r['reward']:+.4f} viol/100 {r['violations_per_100']:.2f} AVO {r['avo']:.2f} BACK {r['back']:.3f} "
              f"LOW {r['low']:.2f} CO {r['co']:.2f} omega {r['omega']:.0f} [{time.perf_counter() - t0:.0f} s]", flush=True)
        OUT.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
