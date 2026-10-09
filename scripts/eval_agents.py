"""Evaluate trained agents and the classical baselines on held-out scenarios and the protocols (GPU).

Agents act greedily (argmax of the policy). All policies start every episode at the base patient's
operating speed except the clinician baseline. Metrics use the true labels and the base RM (N = 5).

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

from lvadrm.agents.ppo import PPOConfig, greedy_policy, make_network  # noqa: E402
from lvadrm.agents.specs import SPECS, agent_obs, env_config_for  # noqa: E402
from lvadrm.env import EnvConfig, reset  # noqa: E402
from lvadrm.evaluation import classical_policy, initial_speeds, metrics, rollout  # noqa: E402
from lvadrm.patients import HF_PATIENT  # noqa: E402
from lvadrm.reward_machine import RMConfig  # noqa: E402
from lvadrm.scenarios import ScenarioConfig, protocol_batch, sample_scenario  # noqa: E402

RUNS = Path("results/phase3/runs")
N_TEST = 2048
TEST_SEED = 12345
BASELINES = ("const_pop", "const_clinician", "simaan_minflow")


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
    return spec, env_cfg, net, params, ppo.hidden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run directory names (default: all)")
    args = ap.parse_args()
    run_dirs = sorted(d for d in RUNS.iterdir() if (d / "params.msgpack").exists())
    if args.runs:
        run_dirs = [d for d in run_dirs if d.name in args.runs]

    base_cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300)
    k_scn, k_run = jax.random.split(jax.random.PRNGKey(TEST_SEED))
    scns = jax.vmap(lambda k: sample_scenario(k, HF_PATIENT, ScenarioConfig(), 300))(jax.random.split(k_scn, N_TEST))
    names, proto_scns = protocol_batch(HF_PATIENT)
    out = {"n_test": N_TEST, "test_seed": TEST_SEED, "random_scenarios": {}, "protocols": {}}

    def evaluate(label, policy_fn, cfg, omega0_fn):
        o = rollout(policy_fn(cfg), scns, cfg, k_run, omega0_fn(scns, cfg))
        out["random_scenarios"][label] = metrics(o, cfg)
        cfg_long = dataclasses.replace(cfg, episode_beats=1206)
        ol = rollout(policy_fn(cfg_long), proto_scns, cfg_long, k_run, omega0_fn(proto_scns, cfg_long))
        for i, proto in enumerate(names):
            oi = {k: (v[:, i : i + 1] if v.ndim >= 2 else v[i : i + 1]) for k, v in ol.items()}
            out["protocols"].setdefault(proto, {})[label] = metrics(oi, cfg_long)
        m = out["random_scenarios"][label]
        print(f"{label:22s} viol N5 {m['N5_episodes_with_violation']:.0%} ({m['N5_violations_per_100_beats']:.2f}/100)  "
              f"AVO {m['avo_frac']:.2f} BACK {m['back_frac']:.2f} SUC {m['suc_frac']:.4f} LOW {m['low_frac']:.2f} "
              f"CO {m['co_mean']:.2f} return {m['return_mean_N5']:.1f}", flush=True)

    for b in BASELINES:
        evaluate(b, lambda cfg, b=b: classical_policy(b, cfg), base_cfg,
                 lambda s, cfg, b=b: initial_speeds(b, s, cfg, k_run))
    for d in run_dirs:
        spec, env_cfg, net, params, hidden = load_agent(d)
        evaluate(d.name, lambda cfg, spec=spec, net=net, params=params, hidden=hidden: greedy_policy(spec, cfg, net, params, hidden),
                 env_cfg, lambda s, cfg: jnp.full(s.base.shape[0], cfg.patient.omega))
    common.save_json("eval.json", out, phase=3)

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
    common.savefig(fig, "learning_curves.png", phase=3)
    plt.close(fig)

    # summary table
    L = ["# Fase 3: agentes PPO contra líneas base", "",
         f"{N_TEST} escenarios de prueba no vistos (semilla {TEST_SEED}), 300 latidos, RM con N = 5. Agentes con política greedy.", "",
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
    (common.results_dir(3) / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote results/phase3/eval.json and summary.md")


if __name__ == "__main__":
    main()
