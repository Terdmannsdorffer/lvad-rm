"""Train one PPO agent (GPU, float32).

    python scripts/train_ppo.py --agent oracle --seed 0 [--beats 30e6] [--n-envs 8192] [--n-steps 25]
    python scripts/train_ppo.py --agent tdm --detector mlp --seed 0        # belief agents (Phase 4)

Writes results/phase3/runs/<agent>[_<detector>]_s<seed>/{params.msgpack, config.json, train_log.json}
(belief agents go to results/phase4/runs).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from lvadrm.agents.belief import make_belief
from lvadrm.agents.ppo import PPOConfig, make_ppo
from lvadrm.agents.specs import SPECS, env_config_for
from lvadrm.env import EnvConfig
from lvadrm.patients import HF_PATIENT
from lvadrm.reward_machine import RMConfig

RUNS = Path("results/phase3/runs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=sorted(SPECS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beats", type=float, default=30e6)
    ap.add_argument("--n-envs", type=int, default=8192)
    ap.add_argument("--n-steps", type=int, default=25)
    ap.add_argument("--rm-n", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--detector", default=None, help="AVO detector for naive / ibu / tdm agents")
    ap.add_argument("--out-dir", default=None, help="directory for the run folder (default: results/phase3|4/runs)")
    ap.add_argument("--tdm-path", default=None, help="TDM parameters for tdm agents (default: Phase 4 file of the detector)")
    ap.add_argument("--no-vf-clip", action="store_true", help="turn off PPO value clipping")
    args = ap.parse_args()

    spec = SPECS[args.agent]
    belief = None
    runs = RUNS
    if spec.belief != "true":
        assert args.detector, "belief agents need --detector"
        belief = make_belief(spec.belief, args.rm_n, args.detector, args.tdm_path)
        runs = Path("results/phase4/runs")
        args.tag = f"_{args.detector}{args.tag}"
    env_cfg = env_config_for(spec, EnvConfig(patient=HF_PATIENT, episode_beats=300, rm=RMConfig(n=args.rm_n)))
    cfg = PPOConfig(n_envs=args.n_envs, n_steps=args.n_steps, total_beats=int(args.beats),
                    vf_clip=None if args.no_vf_clip else 0.2)
    runs = Path(args.out_dir) if args.out_dir else runs
    run_dir = runs / f"{args.agent}{args.tag}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "agent": dataclasses.asdict(spec), "ppo": dataclasses.asdict(cfg), "seed": args.seed, "rm_n": args.rm_n,
        "detector": args.detector,
        "tdm_path": args.tdm_path,
        "env": {"episode_beats": env_cfg.episode_beats, "snr_db": env_cfg.snr_db, "back_ml": env_cfg.back_ml,
                "omega_min": env_cfg.omega_min, "omega_max": env_cfg.omega_max, "rm": dataclasses.asdict(env_cfg.rm)},
    }, indent=2))

    init, update, _ = make_ppo(spec, env_cfg, cfg, belief)
    t0 = time.perf_counter()
    ts = init(jax.random.PRNGKey(args.seed))
    log = []
    n_mb = cfg.n_minibatches_recurrent if spec.recurrent else cfg.n_minibatches
    print(f"{run_dir.name}: {cfg.n_updates} updates x {cfg.n_envs * cfg.n_steps} beats, {n_mb} minibatches", flush=True)
    for i in range(cfg.n_updates):
        ts, m = update(ts)
        m = {k: float(v) for k, v in m.items()} | {"update": i + 1, "beats": (i + 1) * cfg.n_envs * cfg.n_steps,
                                                   "seconds": time.perf_counter() - t0}
        log.append(m)
        if i % 12 == 11 or i == cfg.n_updates - 1:  # once per episode (300 / 25 updates)
            last = log[-12:]
            print(f"[{m['seconds']:6.0f}s] beats {m['beats'] / 1e6:6.2f}M  reward {np.mean([x['reward'] for x in last]):+.4f}  "
                  f"viol/100 {np.mean([x['violations_per_100'] for x in last]):5.2f}  AVO {np.mean([x['avo'] for x in last]):.3f}  "
                  f"BACK {np.mean([x['back'] for x in last]):.3f}  SUC {np.mean([x['suc'] for x in last]):.4f}  "
                  f"CO {np.mean([x['co'] for x in last]):.2f}  ent {m['entropy']:.3f}  kl {m['approx_kl']:.4f}", flush=True)
    (run_dir / "params.msgpack").write_bytes(serialization.to_bytes(ts.params))
    (run_dir / "train_log.json").write_text(json.dumps(log))
    print("wrote", run_dir)


if __name__ == "__main__":
    main()
