"""Phase 7, stage 4 (no RL): the information ladder with one controller class and different estimators (GPU).

Every detector-based policy is the same monitor controller (lvadrm.estimators.monitor_fns): track a flow set point and
drop the speed in 5 % steps while P(counter >= m) > p. Only the estimator of the counter changes. Configurations
(q_set, m, p) are chosen on validation scenarios (Pareto set of mean violation rate and mean deviation of the cardiac
output) and reported on test scenarios; all policies share scenarios and noise seeds.

    python scripts/phase7_monitor.py --what reference --n 10          # constant, blind, true counter; fixed and unknown patients
    python scripts/phase7_monitor.py --what estimators --n 10 --lam 0.5
Writes results/phase7/monitor/<what>_N<n>[_lam<lam>].json (test outcomes per configuration, with per-episode violations).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from phase7_common import OUT, detector, load_filter, load_models, model_path, outcome, pause_point, rollout, scenarios, start_speeds, write_atomic

from lvadrm.baselines.constant import avc_speed
from lvadrm.env import reset_with_scenario
from lvadrm.estimators import BlindParams, MonitorParams, blind_fns, make_estimator, monitor_fns
from lvadrm.presets import env_v3

BEATS = 400
Q_SET = tuple(np.round(np.arange(3.8, 6.21, 0.2), 1))
FACTORS = tuple(np.round(np.arange(1.0, 1.46, 0.05), 2))
P_SOFT = (0.03, 0.1, 0.2, 0.4, 0.7)
N_VAL, N_TEST = 64, 512
VAL_SEED, TEST_SEED = 777, 12345


def const_fns():
    def init(prm, key, st, obs):
        return jnp.zeros(())

    def act(prm, key, st, obs, ps):
        return prm, 0.0, 0.0, ps

    return init, act, lambda prm, ps, st, info, obs2: ps


def grid(*axes):
    return np.stack([a.ravel() for a in np.meshgrid(*[np.asarray(a, float) for a in axes], indexing="ij")], -1)


def pareto(viol, dev, k=14):
    """Indices of the non-dominated configurations (minimize both), thinned to at most k."""
    order = np.lexsort((dev, viol))
    front, best = [], np.inf
    for i in order:
        if dev[i] < best - 1e-6:
            front.append(i)
            best = dev[i]
    if len(front) > k:
        front = [front[j] for j in np.unique(np.linspace(0, len(front) - 1, k).round().astype(int))]
    return front


def run_configs(fns, to_prm, configs, cfg, n, seed, n_scn, omega_const=None, chunk=9000, ref=None):
    """Outcomes [n_config, n_scn] of every configuration on the same scenarios and noise."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    scn, w0 = scenarios(k1, cfg, n_scn), start_speeds(k2, cfg, n_scn)
    per = max(1, chunk // n_scn)
    out = []
    for i in range(0, len(configs), per):
        c = configs[i : i + per]
        pad = np.concatenate([c, np.repeat(c[:1], per - len(c), 0)]) if len(configs) > per else c
        m = len(pad)
        if omega_const is not None:
            prm = jnp.asarray((pad[:, :1] * omega_const[None]).ravel(), jnp.float32)
        else:
            prm = to_prm(np.repeat(pad, n_scn, 0)) if ref is None else to_prm(np.repeat(pad, n_scn, 0), jnp.tile(ref, m))
        s = jax.tree.map(lambda a: jnp.tile(a, (m,) + (1,) * (a.ndim - 1)), scn)  # noqa: B023
        w = jnp.tile(w0, m) if omega_const is None else prm
        o = outcome(rollout(fns, prm, s, cfg, k3, w), n)
        out.append({k: v.reshape(m, n_scn)[: len(c)] for k, v in o.items()})
    return {k: np.concatenate([x[k] for x in out]) for k in out[0]}


def select_and_test(name, fns, to_prm, configs, cfg, n, res, omega_val=None, omega_test=None, ref_val=None, ref_test=None):
    t0 = time.perf_counter()
    val = run_configs(fns, to_prm, configs, cfg, n, VAL_SEED, N_VAL, omega_val, ref=ref_val)
    front = pareto(val["viol"].mean(1), val["dev"].mean(1))
    test = run_configs(fns, to_prm, configs[front], cfg, n, TEST_SEED, N_TEST, omega_test, ref=ref_test)
    res[name] = {"configs": configs[front].tolist(), "val_viol": val["viol"].mean(1)[front].tolist(), "val_dev": val["dev"].mean(1)[front].tolist()}
    res[name] |= {k: v.mean(1).tolist() for k, v in test.items()}
    res[name]["viol_episodes"] = test["viol"].round(4).tolist()
    res[name]["dev_episodes"] = test["dev"].round(4).tolist()
    worst = np.sort(test["viol"], 1)[:, -N_TEST // 20 :].mean(1)
    res[name]["viol_worst5"] = worst.tolist()
    print(f"{name:12s} [{time.perf_counter() - t0:.0f}s] " + "  ".join(
        f"({100 * v:.2f}%, {d:.3f}, w5 {100 * w:.0f}%)" for v, d, w in zip(res[name]["viol"], res[name]["dev"], worst)), flush=True)


def threshold_speeds(cfg, seed, n_scn):
    """omega_AVO of every scenario at the start of the episode (information of a clinician's ramp test)."""
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(seed), 3)
    scn, w0 = scenarios(k1, cfg, n_scn), start_speeds(k2, cfg, n_scn)
    _, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(k3, n_scn), scn, cfg, w0)
    return jax.vmap(avc_speed, in_axes=(0, None))(st, cfg)


def mon(c, ref=None):
    """c: (flow set point or speed factor, m, p); ``ref``: speed the factor refers to (None: flow tracking)."""
    f = jnp.asarray(c[:, 0], jnp.float32)
    return MonitorParams(f, jnp.asarray(c[:, 1], jnp.int32), jnp.asarray(c[:, 2], jnp.float32), f * ref if ref is not None else jnp.zeros_like(f))


def bli(c, ref=None):
    f = jnp.asarray(c[:, 0], jnp.float32)
    return BlindParams(f, jnp.asarray(c[:, 1], jnp.int32), jnp.asarray(c[:, 2], jnp.int32), f * ref if ref is not None else jnp.zeros_like(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["reference", "estimators"], required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--nominal", choices=["flow", "ramp"], default="flow",
                    help="nominal speed: flow tracking, or a factor of the omega_AVO measured at the start (ramp test)")
    args = ap.parse_args()
    n = args.n
    (OUT / "monitor").mkdir(parents=True, exist_ok=True)
    ramp = args.nominal == "ramp"
    sfx = "_ramp" if ramp else ""
    first = FACTORS if ramp else Q_SET
    hard = grid(first, range(1, n), (0.5,))
    soft = grid(first, sorted({max(1, round(0.4 * n)), max(1, round(0.6 * n)), max(1, round(0.8 * n))}), P_SOFT)
    if args.what == "reference":
        path = OUT / "monitor" / f"reference_N{n}{sfx}.json"
        res = json.loads(path.read_text()) if path.exists() else {}
        for regime in ("fixed", "unknown"):
            cfg = env_v3(n, regime, episode_beats=BEATS)
            r = res.setdefault(regime, {})
            if all(k in r for k in ("constant", "blind", "true")):
                continue
            kw = dict(ref_val=threshold_speeds(cfg, VAL_SEED, N_VAL), ref_test=threshold_speeds(cfg, TEST_SEED, N_TEST)) if ramp else {}
            f = grid(np.r_[np.arange(0.85, 1.0, 0.03), np.arange(1.0, 1.4, 0.04)])
            if "constant" not in r:
                pause_point()
                select_and_test("constant", const_fns(), None, f, cfg, n, r, threshold_speeds(cfg, VAL_SEED, N_VAL), threshold_speeds(cfg, TEST_SEED, N_TEST))
                write_atomic(path, json.dumps(res))
            blind = grid(first if ramp else Q_SET[::2], sorted({4, 6, 8, 9, 10, 11, 12, 14, 18, 24}), range(1, 7))
            blind = blind[(blind[:, 2] < blind[:, 1] - 1) & (blind[:, 1] <= 2 * n)]
            if "blind" not in r:
                pause_point()
                select_and_test("blind", blind_fns(cfg), bli, blind, cfg, n, r, **kw)
                write_atomic(path, json.dumps(res))
            pause_point()
            det = detector(0.0, perfect=True)
            select_and_test("true", monitor_fns(make_estimator("true", det, n), cfg, n), mon, hard, cfg, n, r, **kw)
            write_atomic(path, json.dumps(res))
        return
    path = OUT / "monitor" / f"estimators_N{n}_lam{args.lam:g}{sfx}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    cfg = env_v3(n, "unknown", episode_beats=BEATS)
    det = detector(args.lam)
    models = load_models(args.lam, n) if model_path(args.lam, n).exists() else {"filter": load_filter()}
    kw = dict(ref_val=threshold_speeds(cfg, VAL_SEED, N_VAL), ref_test=threshold_speeds(cfg, TEST_SEED, N_TEST)) if ramp else {}
    kinds = args.kinds or (["naive", "ibu_d", "filter"] + (["ibu", "ibu_hist", "tdm"] if "tdm" in models else []))
    for kind in kinds:
        if kind in res:
            continue
        pause_point()
        est = make_estimator(kind.split("@")[0], det, n, models, **({"lam": float(kind.split("@")[1])} if "@" in kind else {}))
        select_and_test(kind, monitor_fns(est, cfg, n), mon, hard if kind == "naive" else soft, cfg, n, res, **kw)
        write_atomic(path, json.dumps(res))


if __name__ == "__main__":
    main()
