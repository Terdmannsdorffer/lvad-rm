"""Phase 6.5 evaluation (GPU): clinical metrics on long episodes and randomized stress families.

    python scripts/eval_phase6_5.py --dirs results/phase6_5/runs/oracle_s0 ... [--baselines] [--name eval]

Test sets (all start at the clinician's speed):
  random        1024 scenarios of the v2 training distribution (seed 12345), 1200 beats (~15 min)
  hypertension  256 randomized scenarios, 900 beats (lvadrm.scenarios.stress_family)
  exercise      256, 900 beats
  hypovolemia   256, 900 beats
Metrics: lvadrm.evaluation.clinical_metrics, per episode set. Merges into results/phase6_5/<name>.json
(existing entries are kept, so agents can be evaluated in several calls).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from eval_agents import load_agent
from lvadrm.agents.belief import load_detector
from lvadrm.agents.ppo import greedy_policy
from lvadrm.baselines.modulation import OpenLoopParams, RuleParams, as_policy, open_loop_fns, rule_fns
from lvadrm.evaluation import classical_policy, clinical_metrics, initial_speeds, rollout
from lvadrm.patients import HF_PATIENT
from lvadrm.presets import env_v2
from lvadrm.scenarios import sample_scenario, stress_family

OUT = Path("results/phase6_5")
N_RANDOM, T_RANDOM, N_FAMILY, T_FAMILY, SEED = 1024, 1200, 256, 900, 12345
FAMILIES = ("hypertension", "exercise", "hypovolemia")


def test_sets(n_random=N_RANDOM, t_random=T_RANDOM, n_family=N_FAMILY, t_family=T_FAMILY):
    k_scn, k_fam = jax.random.split(jax.random.PRNGKey(SEED))
    cfg_r = env_v2(episode_beats=t_random)
    rand = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, cfg_r.scenario, t_random))(jax.random.split(k_scn, n_random))
    fams = [stress_family(k, HF_PATIENT, kind, n_family) for k, kind in zip(jax.random.split(k_fam, 3), FAMILIES)]
    fam = jax.tree.map(lambda *x: jnp.concatenate(x), *fams)
    return (cfg_r, rand), (env_v2(episode_beats=t_family), fam), n_family


def evaluate(policy_fn, sets, key, cfg_tf=None) -> dict:
    """``cfg_tf`` adapts the environment to the agent (observation mode and action set, e.g. oracle_p);
    initial speeds always come from the clinician's rule on the pump-signal environment."""
    (cfg_r, rand), (cfg_f, fam), n_family = sets
    tf = cfg_tf or (lambda c: c)
    out = {}
    o = rollout(policy_fn(tf(cfg_r)), rand, tf(cfg_r), key, initial_speeds("const_clinician", rand, cfg_r, key))
    out["random"] = clinical_metrics(o, cfg_r)
    o = rollout(policy_fn(tf(cfg_f)), fam, tf(cfg_f), key, initial_speeds("const_clinician", fam, cfg_f, key))
    for i, kind in enumerate(FAMILIES):
        sl = slice(i * n_family, (i + 1) * n_family)
        out[kind] = clinical_metrics({k: (v[:, sl] if v.ndim >= 2 else v[sl]) for k, v in o.items()}, cfg_f)
    return out


def baselines(detector: str):
    """Classical baselines. Modulation parameters: the Phase 6 tuning (on v1; not retuned for v2)."""
    ol = OpenLoopParams(base=jnp.float32(0.0), drop=jnp.float32(0.10), period=jnp.int32(5))
    rp = RuleParams(base=jnp.float32(0.0), trigger=jnp.int32(2), step=jnp.float32(0.05), adapt=jnp.float32(0.01), up_after=jnp.int32(20))
    det = load_detector(detector)
    return {
        "const_clinician": lambda cfg: classical_policy("const_clinician", cfg),
        "simaan_minflow": lambda cfg: classical_policy("simaan_minflow", cfg),
        "mod_open_loop": lambda cfg: as_policy(open_loop_fns(), ol),
        f"mod_rule_{detector}": lambda cfg: as_policy(rule_fns(det), rp),
        "mod_rule_oracle": lambda cfg: as_policy(rule_fns(None), rp),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--detector", default="v2_mlp")
    ap.add_argument("--name", default="eval")
    ap.add_argument("--smoke", action="store_true", help="tiny test sets (checks the pipeline only)")
    ap.add_argument("--watch", default=None, help="run directory to poll: evaluate every finished run, until --expect runs are done")
    ap.add_argument("--expect", type=int, default=60)
    ap.add_argument("--poll", type=int, default=300, help="seconds between polls in --watch mode")
    args = ap.parse_args()
    path = OUT / f"{args.name}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    sets = test_sets(16, 60, 8, 60) if args.smoke else test_sets()
    key = jax.random.PRNGKey(SEED + 1)
    todo = dict(baselines(args.detector)) if args.baselines else {}

    def run_all(todo):
        for label, fn in todo.items():
            if label in res:
                continue
            t0 = time.perf_counter()
            fn, tf = fn if isinstance(fn, tuple) else (fn, None)
            res[label] = evaluate(fn, sets, key, tf)
            report(label, t0)
            path.write_text(json.dumps(res, indent=2))

    def agent_fn(d):
        spec, _, net, params, hidden, belief = load_agent(d)
        return ((lambda cfg, spec=spec, net=net, params=params, hidden=hidden, belief=belief: greedy_policy(spec, cfg, net, params, hidden, belief=belief)),
                (lambda cfg, spec=spec: dataclasses.replace(cfg, obs_mode=spec.obs_mode, actions=spec.actions)))

    def report(label, t0):
        r = res[label]
        print(f"{label:24s} random: opening {r['random']['opening_ratio']:.2f}, closed>10s {r['random']['time_closed_gt10s']:.3%}, "
              f"max closure {r['random']['max_closure_s']:.1f}s, viol/h {next(v for k, v in r['random'].items() if k.startswith('violations_per_hour')):.1f}, "
              f"suction/h {r['random']['suction_per_hour']:.2f}, CO err {r['random']['co_abs_err']:.2f} | hypertension closed>10s "
              f"{r['hypertension']['time_closed_gt10s']:.3%} | hypovolemia suction/h {r['hypovolemia']['suction_per_hour']:.2f} "
              f"[{time.perf_counter() - t0:.0f} s]", flush=True)

    for d in map(Path, args.dirs):
        todo[d.name] = agent_fn(d)
    run_all(todo)
    if args.watch:
        watch = Path(args.watch)
        while True:
            done = sorted(p.parent for p in watch.glob("*/params.msgpack"))
            run_all({d.name: agent_fn(d) for d in done if d.name not in res})
            if len(done) >= args.expect:
                break
            time.sleep(args.poll)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
