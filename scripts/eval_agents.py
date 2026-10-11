"""Evaluate trained agents and the classical baselines on held-out scenarios and the protocols (GPU).

Agents act greedily (argmax of the policy). Every policy starts each episode at the clinician's
setting (0.97 x the highest constant speed that opens the aortic valve at t = 0), except the
population constant speed, which by definition runs at the base patient's operating point. Metrics
use the true labels and the base RM (N = 5).

Writes results/phase3/eval.json, results/phase3/summary.md and figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization  # noqa: E402

from lvadrm.agents.belief import make_belief  # noqa: E402
from lvadrm.agents.ppo import PPOConfig, greedy_policy, make_network  # noqa: E402
from lvadrm.agents.specs import SPECS, agent_obs, env_config_for  # noqa: E402
from lvadrm.env import EnvConfig, reset  # noqa: E402
from lvadrm.evaluation import classical_policy, initial_speeds, metrics, rollout  # noqa: E402
from lvadrm.patients import HF_PATIENT  # noqa: E402
from lvadrm.reward_machine import RMConfig  # noqa: E402
from lvadrm.scenarios import WIDE_SCENARIOS, ScenarioConfig, protocol_batch, sample_scenario  # noqa: E402

RUNS = Path("results/phase3/runs")
N_TEST = 2048
TEST_SEED = 12345
BASELINES = ("const_pop", "const_clinician", "simaan_minflow")
TEST_SCENARIOS = {
    "default": ScenarioConfig(),
    "wide": WIDE_SCENARIOS,  # Rs up to 2.5 x the patient's (training: 1.56 x)
    "lowemax": ScenarioConfig(emax_scale=(0.3, 0.5)),  # contractility below the training range (0.5-1.5 x)
}


def load_agent(run_dir: Path):
    conf = json.loads((run_dir / "config.json").read_text())
    spec = SPECS[conf["agent"]["name"]]
    ppo = PPOConfig(hidden=conf["ppo"]["hidden"])
    env_cfg = env_config_for(spec, EnvConfig(patient=HF_PATIENT, episode_beats=300, rm=RMConfig(n=conf["rm_n"])))
    net = make_network(spec, ppo)
    obs, st = reset(jax.random.PRNGKey(0), env_cfg)
    x = agent_obs(spec, obs, st, env_cfg)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros(ppo.hidden), x) if spec.recurrent else net.init(jax.random.PRNGKey(0), x)
    params = serialization.from_bytes(template, (run_dir / "params.msgpack").read_bytes())
    belief = make_belief(spec.belief, conf["rm_n"], conf.get("detector"), conf.get("tdm_path")) if spec.belief != "true" else None
    return spec, env_cfg, net, params, ppo.hidden, belief


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run directory names in results/phase3/runs (default: all)")
    ap.add_argument("--dirs", nargs="*", default=None, help="explicit run directories (overrides --runs)")
    ap.add_argument("--phase", type=int, default=3, help="where to write eval.json / summary.md")
    ap.add_argument("--scenarios", choices=sorted(TEST_SCENARIOS), default="default")
    ap.add_argument("--no-protocols", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--name", default="eval", help="output stem (eval -> eval.json, summary.md)")
    args = ap.parse_args()
    suffix = "" if args.name == "eval" else "_" + args.name.removeprefix("eval_")
    if args.dirs:
        run_dirs = [Path(d) for d in args.dirs if (Path(d) / "params.msgpack").exists()]
    else:
        run_dirs = sorted(d for d in RUNS.iterdir() if (d / "params.msgpack").exists())
        if args.runs:
            run_dirs = [d for d in run_dirs if d.name in args.runs]

    base_cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300)
    k_scn, k_run = jax.random.split(jax.random.PRNGKey(TEST_SEED))
    scn_cfg = TEST_SCENARIOS[args.scenarios]
    scns = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, scn_cfg, 300))(jax.random.split(k_scn, N_TEST))
    names, proto_scns = protocol_batch(HF_PATIENT)
    out = {"n_test": N_TEST, "test_seed": TEST_SEED, "initial_speed": "clinician (const_pop: base)",
           "scenarios": args.scenarios, "random_scenarios": {}, "protocols": {}}
    base_long = dataclasses.replace(base_cfg, episode_beats=1206)
    starts = {
        "clinician": (initial_speeds("const_clinician", scns, base_cfg, k_run),
                      None if args.no_protocols else initial_speeds("const_clinician", proto_scns, base_long, k_run)),
        "base": (jnp.full(N_TEST, HF_PATIENT.omega), jnp.full(len(names), HF_PATIENT.omega)),
    }

    def evaluate(label, policy_fn, cfg, start="clinician"):
        o = rollout(policy_fn(cfg), scns, cfg, k_run, starts[start][0])
        out["random_scenarios"][label] = metrics(o, cfg)
        if args.no_protocols:
            m = out["random_scenarios"][label]
            print(f"{label:28s} viol N5 {m['N5_violations_per_100_beats']:.3f}/100 AVO {m['avo_frac']:.2f} CO {m['co_mean']:.2f}", flush=True)
            return
        cfg_long = dataclasses.replace(cfg, episode_beats=1206)
        ol = rollout(policy_fn(cfg_long), proto_scns, cfg_long, k_run, starts[start][1])
        for i, proto in enumerate(names):
            oi = {k: (v[:, i : i + 1] if v.ndim >= 2 else v[i : i + 1]) for k, v in ol.items()}
            out["protocols"].setdefault(proto, {})[label] = metrics(oi, cfg_long)
        m = out["random_scenarios"][label]
        print(f"{label:22s} viol N5 {m['N5_episodes_with_violation']:.0%} ({m['N5_violations_per_100_beats']:.2f}/100)  "
              f"AVO {m['avo_frac']:.2f} BACK {m['back_frac']:.2f} SUC {m['suc_frac']:.4f} LOW {m['low_frac']:.2f} "
              f"CO {m['co_mean']:.2f} return (N={cfg.rm.n}) {m[f'return_mean_N{cfg.rm.n}']:.1f}", flush=True)

    for b in () if args.no_baselines else BASELINES:
        evaluate(b, lambda cfg, b=b: classical_policy(b, cfg), base_cfg, "base" if b == "const_pop" else "clinician")
    for d in run_dirs:
        spec, env_cfg, net, params, hidden, belief = load_agent(d)
        evaluate(d.name, lambda cfg, spec=spec, net=net, params=params, hidden=hidden, belief=belief:
                 greedy_policy(spec, cfg, net, params, hidden, belief=belief), env_cfg)
    common.save_json(f"{args.name}.json", out, phase=args.phase)
    if args.name != "eval" or args.scenarios != "default":
        print(f"wrote results/phase{args.phase}/{args.name}.json")
        return

    # learning curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for d in run_dirs:
        log = json.loads((d / "train_log.json").read_text())
        beats = np.array([x["beats"] for x in log]) / 1e6
        for a, key in zip(ax, ("r_env", "violations_per_100", "back")):
            y = np.convolve([x[key] for x in log], np.ones(12) / 12, mode="valid")
            a.plot(beats[11:], y, label=d.name)
    for a, t in zip(ax, ("RM reward per beat", "violations per 100 beats (N = 5)", "BACK fraction")):
        a.set(xlabel="training beats (M)", title=t)
    ax[0].legend(fontsize=7)
    common.savefig(fig, "learning_curves.png", phase=args.phase)
    plt.close(fig)

    # summary table
    L = [f"# Fase {args.phase}: agentes PPO contra líneas base", "",
         f"{N_TEST} escenarios de prueba no vistos (semilla {TEST_SEED}), 300 latidos, RM con N = 5. Agentes con política greedy. "
         "Todas las políticas parten en la velocidad del clínico, salvo const_pop (velocidad poblacional).", "",
         "| Política | Episodios con violación (N=5) | Violaciones / 100 latidos | Ídem desde el latido 20 | AVO | BACK | SUC | LOW | Gasto medio | Retorno RM |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, m in out["random_scenarios"].items():
        L.append(f"| {k} | {m['N5_episodes_with_violation']:.0%} | {m['N5_violations_per_100_beats']:.2f} | "
                 f"{m['N5_episodes_with_violation_after20']:.0%}, {m['N5_violations_per_100_beats_after20']:.2f} | {m['avo_frac']:.0%} | "
                 f"{m['back_frac']:.0%} | {m['suc_frac']:.2%} | {m['low_frac']:.0%} | {m['co_mean']:.2f} | "
                 f"{m['return_mean_N5']:.1f} ± {m['return_std_N5']:.1f} |")
    L += ["", "## Protocolos (1206 latidos): violaciones / 100 latidos (N = 5), BACK y gasto", "",
          "| Política | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for label in out["random_scenarios"]:
        cells = [f"{out['protocols'][p][label]['N5_violations_per_100_beats']:.1f} (BACK {out['protocols'][p][label]['back_frac']:.0%}, "
                 f"gasto {out['protocols'][p][label]['co_mean']:.1f})" for p in names]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    (common.results_dir(args.phase) / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote results/phase{args.phase}/eval.json and summary.md")


if __name__ == "__main__":
    main()
