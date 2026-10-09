"""Why do belief agents fail in the hypertension protocol? (GPU, single environment, ~12 min)

Runs the Ferreira hypertension protocol (Rs x 1.8 from 200 s, 50 s ramp) with seed-0/1 agents of
Phase 5 and records, beat by beat, the true AVO, the MLP detector output and the agent's belief.
Reports for the hypertension phase (t > 260 s): AVO fraction, mean speed, BACK fraction, detector
false-positive rate on closed beats, belief mass on u = 0 when the valve stayed closed, and belief on
the violation-risk state u = 4 when the true state is 4.

Writes results/phase5/hypertension_diagnosis.json.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import numpy as np

from eval_agents import load_agent
from lvadrm.agents.belief import load_detector
from lvadrm.agents.ppo import greedy_policy
from lvadrm.env import reset_with_scenario, step_speed
from lvadrm.evaluation import initial_speeds
from lvadrm.patients import HF_PATIENT
from lvadrm.scenarios import ferreira_hypertension

RUNS = Path("results/phase5/runs")
AGENTS = ("tdm_mlp_s0", "ibu_mlp_s0", "naive_mlp_s0", "oracle_s0", "rnn_s1")
T_HYPER = 260.0


def main() -> None:
    det = load_detector("mlp")
    scn, _ = ferreira_hypertension(HF_PATIENT)
    out = {}
    for name in AGENTS:
        spec, cfg, net, params, hidden, belief = load_agent(RUNS / name)
        cfg = dataclasses.replace(cfg, episode_beats=1206)
        pol = greedy_policy(spec, cfg, net, params, hidden, belief=belief)
        w0 = initial_speeds("const_clinician", jax.tree.map(lambda a: a[None], scn), dataclasses.replace(cfg, obs_mode="pump"), jax.random.PRNGKey(0))[0]
        obs, st = reset_with_scenario(jax.random.PRNGKey(1), scn, cfg, w0)
        ps = pol.init(jax.random.PRNGKey(2), st, obs)

        @jax.jit
        def one(key, st, obs, ps):
            ka, ks = jax.random.split(key)
            omega, rate, hold, ps = pol.act(ka, st, obs, ps)
            obs2, st2, _, _, info = step_speed(ks, st, omega, cfg, rate, hold)
            p, _, _ = det.step(ks, jnp.zeros(()), obs2, info)
            ps2 = pol.observe(ps, st, info, obs2)
            bvec = belief.vector(ps2[1]) if belief is not None else jax.nn.one_hot(info["u_next"], cfg.rm.n)
            return obs2, st2, ps2, (info["labels"][0], info["u_next"], p, bvec, info["omega"], st.t, info["v_back"])

        rec = []
        for k in jax.random.split(jax.random.PRNGKey(3), cfg.episode_beats):
            obs, st, ps, o = one(k, st, obs, ps)
            rec.append([np.asarray(x) for x in o])
        avo, u, p, b, w, t, vb = (np.array([r[i] for r in rec]) for i in range(7))
        hyp = t > T_HYPER
        closed = hyp & ~avo
        risk = hyp & (u == cfg.rm.n - 1)
        out[name] = {
            "uses_belief": belief is not None,
            "avo_frac": float(avo[hyp].mean()),
            "omega_mean": float(w[hyp].mean()),
            "back_frac": float(np.mean(vb[hyp] > cfg.back_ml)),
            "detector_mean_p_when_closed": float(p[closed].mean()) if closed.any() else None,
            "detector_fp_rate_when_closed": float(np.mean(p[closed] > 0.5)) if closed.any() else None,
            "belief_u0_when_closed": float(b[closed][:, 0].mean()) if closed.any() and belief is not None else None,
            "belief_risk_when_at_risk": float(b[risk][:, -1].mean()) if risk.any() and belief is not None else None,
        }
        print(name, out[name], flush=True)
    common.save_json("hypertension_diagnosis.json", out, phase=5)


if __name__ == "__main__":
    main()
