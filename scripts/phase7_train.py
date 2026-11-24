"""Phase 7, stage 5: PPO agents of the information ladder (GPU). Runs a queue and skips finished runs.

    python scripts/phase7_train.py --regime unknown --kinds true window last none --rv 0.3 1 3 --seeds 0 1 2      # true labels
    python scripts/phase7_train.py --regime unknown --lam 0.5 --kinds naive ibu tdm filter --rv 0.3 1 3 --seeds 0 1 2

Kinds: estimators of lvadrm.estimators (their vector, with the current detection, is appended to the observation),
"last" (recurrent policy that sees the current detection, no reward machine), "none" (recurrent policy without any
detection), "press" and "press_mlp" (recurrent and memoryless policies that also see the end-diastolic pressure and
the valve pressure difference of the last two beats, no detector: the invasive reference) and "<kind>_rnn" (recurrent policy with that estimator: same memory and information as "last", plus the
reward-machine structure). Without --lam the detection is the true label. All agents share the reduced observation.
Writes results/phase7/runs/<run>/{params.msgpack, config.json, train_log.json}.

The complete training state goes to results/phase7/ckpt/<run>.npz every CKPT_EVERY updates (about one minute) and when a
pause is requested (file results/phase7/PAUSE); a run that finds its checkpoint continues from it.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import time
import zipfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from phase7_common import OUT, PAUSE, detector, load_filter, load_models, pause_point, write_atomic

from lvadrm.agents.ppo import PPOConfig, make_ppo
from lvadrm.agents.specs import FIVE_ACTIONS, AgentSpec, env_config_for
from lvadrm.estimators import make_estimator
from lvadrm.presets import env_v3

RECURRENT = ("last", "none", "press")
CKPT, CKPT_EVERY = OUT / "ckpt", 12
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


def save_checkpoint(path, ts, log, meta):
    buf = io.BytesIO()
    leaves = {f"a{i:04d}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(ts))}
    np.savez(buf, meta=np.array(json.dumps(meta, sort_keys=True)), log=np.array(json.dumps(log)), **leaves)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, buf.getvalue())


def load_checkpoint(path, template, meta):
    """(training state, log) of a checkpoint of the same run, or None. ``template`` gives the structure."""
    if not path.exists():
        return None
    try:
        with np.load(path) as z:
            if str(z["meta"]) != json.dumps(meta, sort_keys=True):
                raise ValueError("it belongs to another configuration")
            leaves, tree = jax.tree.flatten(template)
            new = [z[f"a{i:04d}"] for i in range(len(leaves))]
            if f"a{len(leaves):04d}" in z.files or any(a.shape != b.shape or a.dtype != b.dtype for a, b in zip(new, leaves)):
                raise ValueError("the structure of the state changed")
            return jax.tree.unflatten(tree, [jnp.asarray(a) for a in new]), json.loads(str(z["log"]))
    except (ValueError, KeyError, OSError, EOFError, zipfile.BadZipFile) as e:
        print(f"{path.name}: checkpoint ignored, {e}", flush=True)
        return None


def train(regime, lam, kind, n, rv, seed, beats, out, tag="", **ppo_kw):
    run = out / (run_name(regime, lam, kind, n, rv, seed) + tag)
    if (run / "params.msgpack").exists():
        return
    pause_point()
    run.mkdir(parents=True, exist_ok=True)
    cfg = env_config_for(spec_for(kind), env_v3(n, regime, rv))
    ppo = PPOConfig(total_beats=int(beats), vf_clip=None, **ppo_kw)
    init, update, _ = make_ppo(spec_for(kind), cfg, ppo, belief_for(kind, lam, n))
    meta = dict(regime=regime, lam=lam, kind=kind, n=n, rv=rv, seed=seed, beats=beats, ppo=ppo_kw)
    ckpt = CKPT / f"{out.name}_{run.name}.npz"
    ts, log = init(jax.random.PRNGKey(seed)), []
    saved = load_checkpoint(ckpt, ts, meta)
    if saved is not None:
        ts, log = saved
        print(f"{run.name}: continues from update {len(log)} of {ppo.n_updates}", flush=True)
    t0 = time.perf_counter() - (log[-1]["seconds"] if log else 0.0)
    for i in range(len(log), ppo.n_updates):
        ts, m = update(ts)
        log.append({k: float(v) for k, v in m.items()} | {"beats": (i + 1) * ppo.n_envs * ppo.n_steps, "seconds": time.perf_counter() - t0})
        if i % 24 == 23 or i == ppo.n_updates - 1:
            s = {k: np.mean([x[k] for x in log[-12:]]) for k in ("reward", "violations_per_100", "avo", "back", "co", "omega", "entropy")}
            print(f"{run.name} [{log[-1]['seconds']:5.0f}s] {log[-1]['beats'] / 1e6:5.1f}M  " + "  ".join(f"{k} {v:.3f}" for k, v in s.items()), flush=True)
        if i < ppo.n_updates - 1 and (i % CKPT_EVERY == CKPT_EVERY - 1 or PAUSE.exists()):
            save_checkpoint(ckpt, ts, log, meta)
            pause_point()
    write_atomic(run / "train_log.json", json.dumps(log))
    write_atomic(run / "config.json", json.dumps(meta))
    write_atomic(run / "params.msgpack", serialization.to_bytes(ts.params))  # last: it marks the run as finished
    ckpt.unlink(missing_ok=True)


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
