"""Phase 7, stage 5: PPO agents of the information ladder (GPU). Runs a queue and skips finished runs.

    python scripts/phase7_train.py --regime unknown --kinds true window last none --rv 0.3 1 3 --seeds 0 1 2      # true labels
    python scripts/phase7_train.py --regime unknown --lam 0.5 --kinds naive ibu tdm filter --rv 0.3 1 3 --seeds 0 1 2

Kinds: estimators of lvadrm.estimators (their vector, with the current detection, is appended to the observation),
"last" (recurrent policy that sees the current detection, no reward machine), "none" (recurrent policy without any
detection), "press" and "press_mlp" (recurrent and memoryless policies that also see the end-diastolic pressure and
the valve pressure difference of the last two beats, no detector: the invasive reference) and "<kind>_rnn" (recurrent policy with that estimator: same memory and information as "last", plus the
reward-machine structure). Without --lam the detection is the true label. All agents share the reduced observation.
Writes results/phase7/runs/<run>/{params.msgpack, config.json, train_log.json}.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time

import jax
import numpy as np
from flax import serialization
from phase7_common import OUT, detector, load_filter, load_models

from lvadrm.agents.ppo import PPOConfig, make_ppo
from lvadrm.agents.specs import FIVE_ACTIONS, AgentSpec, env_config_for
from lvadrm.estimators import make_estimator
from lvadrm.presets import env_v3

RECURRENT = ("last", "none", "press")
PRESSURE = ("press", "press_mlp")  # no detector: the pressures show the state of the valve


def spec_for(kind):
    obs_mode = "reduced_pressure" if kind in PRESSURE else "reduced"
    return AgentSpec(kind, obs_mode, False, "rm", FIVE_ACTIONS, kind in RECURRENT or kind.endswith("_rnn"))


def belief_for(kind, lam, n, models=None):
    if kind == "none" or kind in PRESSURE:
        return None
    kind = kind.removesuffix("_rnn")
    det = detector(0.0, perfect=True) if lam is None else detector(lam)
    if models is None and lam is not None:
        needs = kind in ("ibu", "ibu_hist", "tdm", "filter")
        models = load_models(lam, n) if needs and kind != "filter" else {"filter": load_filter()} if needs else None
    return make_estimator(kind, det, n, models)


def run_name(regime, lam, kind, n, rv, seed):
    lam_s = "true" if lam is None else f"lam{lam:g}"
    return f"{regime}_{lam_s}_{kind}_N{n}_rv{rv:g}_s{seed}"


def train(regime, lam, kind, n, rv, seed, beats, out, tag="", **ppo_kw):
    run = out / (run_name(regime, lam, kind, n, rv, seed) + tag)
    if (run / "params.msgpack").exists():
        return
    run.mkdir(parents=True, exist_ok=True)
    cfg = env_config_for(spec_for(kind), env_v3(n, regime, rv))
    ppo = PPOConfig(total_beats=int(beats), vf_clip=None, **ppo_kw)
    init, update, _ = make_ppo(spec_for(kind), cfg, ppo, belief_for(kind, lam, n))
    t0, log = time.perf_counter(), []
    ts = init(jax.random.PRNGKey(seed))
    for i in range(ppo.n_updates):
        ts, m = update(ts)
        log.append({k: float(v) for k, v in m.items()} | {"beats": (i + 1) * ppo.n_envs * ppo.n_steps, "seconds": time.perf_counter() - t0})
        if i % 24 == 23 or i == ppo.n_updates - 1:
            s = {k: np.mean([x[k] for x in log[-12:]]) for k in ("reward", "violations_per_100", "avo", "back", "co", "omega", "entropy")}
            print(f"{run.name} [{log[-1]['seconds']:5.0f}s] {log[-1]['beats'] / 1e6:5.1f}M  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()), flush=True)
    (run / "params.msgpack").write_bytes(serialization.to_bytes(ts.params))
    (run / "train_log.json").write_text(json.dumps(log))
    (run / "config.json").write_text(json.dumps(dict(regime=regime, lam=lam, kind=kind, n=n, rv=rv, seed=seed, beats=beats, ppo=ppo_kw)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default="unknown")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--kinds", nargs="+", required=True)
    ap.add_argument("--rv", type=float, nargs="+", default=[1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--beats", type=float, default=30e6)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--ent", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    kw = {k: v for k, v in (("ent_coef", a.ent), ("gamma", a.gamma)) if v is not None}
    for seed, rv, kind in itertools.product(a.seeds, a.rv, a.kinds):
        train(a.regime, a.lam, kind, a.n, rv, seed, a.beats, OUT / a.out, a.tag, **kw)


if __name__ == "__main__":
    main()
