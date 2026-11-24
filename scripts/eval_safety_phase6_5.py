"""Phase 6.5, block B4: Simplex runtime assurance around the belief agents (GPU).

    python scripts/eval_safety_phase6_5.py

Wrapped agents: TDM and IBU (v2_mlp detector), seeds 0-4. Decision module (lvadrm.safety):
  - out-of-distribution observation: Mahalanobis distance of the pump-flow features above the
    0.995 quantile of the training data (random + oracle episodes) for 3 consecutive beats;
  - uncertain belief: entropy of the RM-state belief > 0.8 nats (maximum ln 3 = 1.10 for N = 3);
  fallback: open-loop speed modulation around the clinician's speed (Phase 6 parameters); dwell 30 beats.
Test sets (clinician's initial speed): random (512 x 1200 beats, in distribution), the three stress
families (128 each, 900 beats) and an out-of-distribution set (512 x 900 beats): Rs 1.8-2.5 x the
patient's, Emax 0.5-0.7 x, Vtot 0.70-0.80 x, up to 3 events within these ranges.

P3 (pre-registered; margin fixed here before running): TDM + Simplex is non-inferior to the
clinician's constant speed on the out-of-distribution set, % of time in closures > 10 s, margin
delta = 2 percentage points (one-sided Welch test of mean difference < delta, alpha 0.05), and
superior in distribution. Writes results/phase6_5/eval_safety.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from eval_agents import load_agent
from lvadrm.agents.ppo import greedy_policy
from lvadrm.baselines.modulation import OpenLoopParams, RuleParams, as_policy, open_loop_fns, rule_fns
from lvadrm.env import reset_with_scenario, step_speed
from lvadrm.evaluation import INFO_KEYS, classical_policy, clinical_metrics, initial_speeds
from lvadrm.patients import HF_PATIENT
from lvadrm.presets import env_v2
from lvadrm.safety import fit_ood, simplex_policy
from lvadrm.scenarios import ScenarioConfig, sample_scenario, stress_family

OUT = Path("results/phase6_5")
RUNS = OUT / "runs"
SEED = 54321
FAMILIES = ("hypertension", "exercise", "hypovolemia")
OOD_SCN = ScenarioConfig(rs_scale=(1.8, 2.5), emax_scale=(0.5, 0.7), vtot_scale=(0.70, 0.80))
ENTROPY_THR, K_OOD, DWELL, DELTA = 0.8, 3, 30, 0.02


def rollout_with_state(policy, scns, cfg, key, omega0):
    """lvadrm.evaluation.rollout that also returns the final policy state (to count fallback beats)."""
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
        (_, _, ps), (r, info) = jax.lax.scan(lambda c, kk: step_v(c, jax.random.split(kk, n)), (st, obs, ps), jax.random.split(k, cfg.episode_beats))
        return r, info, ps

    r, info, ps = run(st, obs, k_roll)
    out = {k: np.asarray(v) for k, v in info.items()}
    out["reward"] = np.asarray(r)
    return out, ps


def test_sets():
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(SEED), 3)
    cfg_r, cfg_f = env_v2(episode_beats=1200), env_v2(episode_beats=900)
    rand = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, cfg_r.scenario, 1200))(jax.random.split(k1, 512))
    fam = jax.tree.map(lambda *x: jnp.concatenate(x), *[stress_family(k, HF_PATIENT, kind, 128) for k, kind in zip(jax.random.split(k2, 3), FAMILIES)])
    ood = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, OOD_SCN, 900))(jax.random.split(k3, 512))
    return {"random": (cfg_r, rand), "families": (cfg_f, fam), "ood": (cfg_f, ood)}


def evaluate(make_policy, sets, key, only=None) -> dict:
    out = {}
    for name, (cfg, scns) in sets.items():
        if only and name not in only:
            continue
        pol = make_policy(cfg)
        o, ps = rollout_with_state(pol, scns, cfg, key, initial_speeds("const_clinician", scns, cfg, key))
        fb = float(np.asarray(ps.fb_beats).sum() / o["tc"].size) if hasattr(ps, "fb_beats") else 0.0
        sw = float(np.asarray(ps.switches).sum() / (o["tc"].sum() / 3600.0)) if hasattr(ps, "switches") else 0.0
        if name == "families":
            for i, kind in enumerate(FAMILIES):
                sl = slice(i * 128, (i + 1) * 128)
                out[kind] = clinical_metrics({k: (v[:, sl] if v.ndim >= 2 else v[sl]) for k, v in o.items()}, cfg)
        else:
            out[name] = clinical_metrics(o, cfg) | {"fallback_frac": fb, "switches_per_hour": sw}
    return out


def main() -> None:
    path = OUT / "eval_safety.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    obs = np.concatenate([np.load(OUT / "data" / f"train_{s}.npz")["obs"] for s in ("random", "oracle_s0")])
    ood = fit_ood(obs)
    sets = test_sets()
    key = jax.random.PRNGKey(SEED + 1)
    ol = OpenLoopParams(base=jnp.float32(0.0), drop=jnp.float32(0.10), period=jnp.int32(5))
    RULE = RuleParams(base=jnp.float32(0.0), trigger=jnp.int32(2), step=jnp.float32(0.05), adapt=jnp.float32(0.01), up_after=jnp.int32(20))  # noqa: N806
    todo = {
        "const_clinician": (lambda cfg: classical_policy("const_clinician", cfg), None),
        "mod_open_loop": (lambda cfg: as_policy(open_loop_fns(), ol), None),
        "mod_rule_oracle": (lambda cfg: as_policy(rule_fns(None), RULE), None),
    }
    for agent in ("tdm_v2_mlp", "ibu_v2_mlp"):
        for s in range(5):
            d = RUNS / f"{agent}_s{s}"
            spec, _, net, params, hidden, belief = load_agent(d)

            def plain(cfg, spec=spec, net=net, params=params, hidden=hidden, belief=belief):
                from dataclasses import replace
                return greedy_policy(spec, replace(cfg, obs_mode=spec.obs_mode, actions=spec.actions), net, params, hidden, belief=belief)

            def wrapped(cfg, plain=plain, belief=belief):
                return simplex_policy(plain(cfg), as_policy(open_loop_fns(), ol), ood=ood, k_ood=K_OOD,
                                      belief_vec=lambda pa: belief.vector(pa[1]), entropy_thr=ENTROPY_THR, dwell=DWELL)

            todo[f"{d.name}"] = (plain, ["ood"])  # random and families are already in eval.json
            todo[f"{d.name}+simplex"] = (wrapped, None)
    for label, (fn, only) in todo.items():
        if label in res:
            continue
        t0 = time.perf_counter()
        res[label] = evaluate(fn, sets, key, only)
        r = res[label]
        line = " | ".join(f"{k}: >10s {v['time_closed_gt10s']:.2%} back {v['back_frac']:.2f}" + (f" fb {v['fallback_frac']:.2f}" if "fallback_frac" in v else "")
                          for k, v in r.items() if k in ("random", "hypertension", "ood"))
        print(f"{label:22s} {line} [{time.perf_counter() - t0:.0f} s]", flush=True)
        path.write_text(json.dumps(res, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
