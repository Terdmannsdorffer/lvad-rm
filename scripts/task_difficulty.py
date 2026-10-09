"""Is the task non-trivial? Classical baselines on random scenarios and evaluation protocols (GPU, float32).

Policies (HANDOFF section 11):
  const_pop        one constant speed for everyone (base patient operating point)
  const_clinician  per-episode constant speed: 0.97 x highest speed that opens the aortic valve at t = 0
  simaan_minflow   minimum-flow slope controller, c = 0.15, slew 250 rpm/s (validate_controller.py)
  random           uniformly random actions (reference)

Metrics per policy: episodes with at least one RM violation and violations per 100 beats for
N = 3, 5, 10 (computed from the true AVO sequence), fraction of beats with AVO / SUC / BACK / LOW,
mean cardiac output and mean return of the base RM (N = 5).

Writes results/phase2/task_difficulty[_wide].json, results/phase2/task_difficulty_protocols.json and figures.
"""

from __future__ import annotations

import argparse
import time

import common
import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lvadrm.env import EnvConfig  # noqa: E402
from lvadrm.evaluation import N_LIST, classical_policy, initial_speeds, metrics, rollout  # noqa: E402
from lvadrm.patients import HF_PATIENT  # noqa: E402
from lvadrm.scenarios import WIDE_SCENARIOS, ScenarioConfig, protocol_batch, sample_scenario  # noqa: E402

N_EPISODES = 2048
POLICIES = ("const_pop", "const_clinician", "simaan_minflow", "random")


def run_policy(name, scns, cfg, key):
    return rollout(classical_policy(name, cfg), scns, cfg, key, initial_speeds(name, scns, cfg, key))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranges", choices=("default", "wide"), default="default")
    ap.add_argument("--protocols", action="store_true", help="also run the long evaluation protocols")
    args = ap.parse_args()
    scn_cfg = WIDE_SCENARIOS if args.ranges == "wide" else ScenarioConfig()
    suffix = "" if args.ranges == "default" else "_wide"
    cfg = EnvConfig(patient=HF_PATIENT, episode_beats=300, scenario=scn_cfg)
    key = jax.random.PRNGKey(0)
    k_scn, k_run = jax.random.split(key)
    scns = jax.vmap(lambda k: sample_scenario(k, cfg.patient, cfg.scenario, cfg.episode_beats))(jax.random.split(k_scn, N_EPISODES))
    out = {"n_episodes": N_EPISODES, "episode_beats": cfg.episode_beats, "ranges": args.ranges,
           "scenario_config": scn_cfg.__dict__, "random_scenarios": {}}
    for name in POLICIES:
        t0 = time.perf_counter()
        o = run_policy(name, scns, cfg, k_run)
        out["random_scenarios"][name] = metrics(o, cfg) | {"seconds": time.perf_counter() - t0}
        print(name, {k: round(v, 3) for k, v in out["random_scenarios"][name].items()}, flush=True)
    common.save_json(f"task_difficulty{suffix}.json", out, phase=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(POLICIES))
    for j, n in enumerate(N_LIST):
        ax[0].bar(x + (j - 1) * 0.25, [out["random_scenarios"][p][f"N{n}_episodes_with_violation"] for p in POLICIES], 0.25, label=f"N = {n}")
    ax[0].set(xticks=x, xticklabels=POLICIES, ylabel="episodes with >= 1 violation", title=f"{N_EPISODES} random scenarios, 300 beats ({args.ranges} ranges)")
    ax[0].legend(fontsize=8)
    for j, key_ in enumerate(("suc_frac", "back_frac", "low_frac")):
        ax[1].bar(x + (j - 1) * 0.25, [out["random_scenarios"][p][key_] for p in POLICIES], 0.25, label=key_)
    ax[1].set(xticks=x, xticklabels=POLICIES, ylabel="fraction of beats")
    ax[1].legend(fontsize=8)
    common.savefig(fig, f"task_difficulty{suffix}.png", phase=2)
    plt.close(fig)
    print(f"wrote results/phase2/task_difficulty{suffix}.json")
    if args.protocols:
        run_protocols(k_run)


def run_protocols(key) -> None:
    """Deterministic evaluation protocols (long episodes; independent of the sampling ranges)."""
    cfg_long = EnvConfig(patient=HF_PATIENT, episode_beats=1206)
    names, scn_b = protocol_batch(HF_PATIENT)
    out, traces = {"episode_beats": cfg_long.episode_beats, "protocols": {}}, {}
    for name in POLICIES[:3]:
        o = run_policy(name, scn_b, cfg_long, key)
        for i, proto in enumerate(names):
            oi = {k: (v[:, i : i + 1] if v.ndim >= 2 else v[i : i + 1]) for k, v in o.items()}
            out["protocols"].setdefault(proto, {})[name] = metrics(oi, cfg_long)
        traces[name] = o
        print("protocols", name, {p: round(out["protocols"][p][name]["N5_violations_per_100_beats"], 2) for p in names})
    common.save_json("task_difficulty_protocols.json", out, phase=2)

    i = names.index("fz_sequential")
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for name, o in traces.items():
        ax[0].plot(o["omega"][:, i], lw=0.8, label=name)
        ax[1].plot(o["co"][:, i], lw=0.8)
        ax[2].plot(o["labels"][:, i, 0].astype(float) + 0.05 * POLICIES.index(name), ".", ms=2)
    ax[0].set(ylabel="speed (rpm)")
    ax[0].legend(fontsize=8)
    ax[1].set(ylabel="CO (l/min)")
    ax[2].set(ylabel="AVO", xlabel="beat")
    fig.suptitle("Sequential protocol (Vtot, HR, Rs, Emax up/down), HF patient")
    common.savefig(fig, "protocol_fz_sequential.png", phase=2)
    plt.close(fig)
    print("wrote results/phase2/task_difficulty_protocols.json")


if __name__ == "__main__":
    main()
