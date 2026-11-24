"""Phase 6.5 (docs/phase6_5_plan.md): calibration of the v2 environment, data, inference models.

    python scripts/phase6_5.py calibrate-starling   # steady states vs speed -> Starling target
    python scripts/phase6_5.py calibrate-suction    # Vtot range for which suction is reachable
    python scripts/phase6_5.py calibrate-noise      # flow_noise_scale for a clinical-level AVO detector
    python scripts/phase6_5.py collect --source random|oracle
    python scripts/phase6_5.py train-detector
    python scripts/phase6_5.py train-inference

Results in results/phase6_5/ (calibration.json, detectors/, tdm/, data/ untracked).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lvadrm.baselines.constant import CLINICIAN_MARGIN, avc_speed  # noqa: E402
from lvadrm.env import reset_with_scenario, step_speed  # noqa: E402
from lvadrm.patients import HF_PATIENT  # noqa: E402
from lvadrm.presets import CALIBRATION, env_v2, v2_scenarios  # noqa: E402
from lvadrm.scenarios import make_scenario, sample_scenario  # noqa: E402

OUT = Path("results/phase6_5")
P = HF_PATIENT


def save_calibration(section: str, data: dict, chosen: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cal = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {"chosen": {}}
    cal[section] = data
    if chosen:
        cal["chosen"].update(chosen)
    CALIBRATION.write_text(json.dumps(cal, indent=2))


def steady_states(conditions: dict[str, dict], speeds: np.ndarray, cfg, n_beats: int = 40) -> dict[str, np.ndarray]:
    """Last-beat hemodynamics at constant speed and constant parameters, for every (condition, speed)."""
    names = list(conditions)
    scns = [make_scenario({**{"rs": P.rs, "hr": P.hr, "emax": P.emax, "vtot": P.vtot}, **conditions[c]}, []) for c in names]
    scn = jax.tree.map(lambda *x: jnp.stack(x), *scns)
    C, S = len(names), len(speeds)
    scn = jax.tree.map(lambda a: jnp.repeat(a, S, axis=0), scn)
    omega = jnp.tile(jnp.asarray(speeds, jnp.float32), C)
    cfg = replace(cfg, episode_beats=n_beats)

    @jax.jit
    def run(scn, omega):
        keys = jax.random.split(jax.random.PRNGKey(0), omega.shape[0])
        _, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(keys, scn, cfg, omega)

        def body(st, k):
            _, st2, _, _, info = jax.vmap(step_speed, in_axes=(0, 0, 0, None))(jax.random.split(k, omega.shape[0]), st, omega, cfg)
            return st2, {k_: info[k_] for k_ in ("co", "edp", "labels", "lap_mean", "aop_mean", "lvp_min")}

        _, rec = jax.lax.scan(body, st, jax.random.split(jax.random.PRNGKey(1), n_beats))
        return jax.tree.map(lambda a: a[-5:].mean(0) if a.dtype != jnp.bool_ else a[-5:].any(0), rec)

    rec = {k: np.asarray(v).reshape(C, S, *np.asarray(v).shape[1:]) for k, v in run(scn, omega).items()}
    rec["names"] = names
    return rec


def calibrate_starling(_args) -> None:
    cfg = env_v2(obs_model="measured", perf_target="fixed")
    conditions = {
        "rest": {},
        "light_exercise": {"rs": 0.8 * P.rs, "hr": 90.0},
        "exercise": {"rs": 0.6 * P.rs, "hr": 135.0},
        "hypertension": {"rs": 1.8 * P.rs},
        "hypovolemia_0.85": {"vtot": 0.85 * P.vtot},
        "hypovolemia_0.75": {"vtot": 0.75 * P.vtot},
        "low_contractility": {"emax": 0.7 * P.emax},
    }
    speeds = np.arange(6000.0, 16001.0, 250.0)
    r = steady_states(conditions, speeds, cfg)
    # clinician speed at rest (as in the evaluation)
    scn0 = make_scenario({"rs": P.rs, "hr": P.hr, "emax": P.emax, "vtot": P.vtot}, [])
    _, st0 = reset_with_scenario(jax.random.PRNGKey(0), scn0, cfg, P.omega)
    w_cl = float(max(avc_speed(st0, cfg) * (1.0 - CLINICIAN_MARGIN), cfg.omega_min))
    i_cl = int(np.argmin(np.abs(speeds - w_cl)))
    rest = r["names"].index("rest")
    edp_rest, co_cl = float(r["edp"][rest, i_cl]), float(r["co"][rest, i_cl])
    print(f"clinician speed at rest {w_cl:.0f} rpm: CO {co_cl:.2f} l/min, EDP {edp_rest:.1f} mmHg")
    for c, name in enumerate(r["names"]):
        avo, suc = r["labels"][c, :, 0], r["labels"][c, :, 1]
        w_close = speeds[np.argmax(~avo)] if (~avo).any() else np.nan
        w_suc = speeds[np.argmax(suc)] if suc.any() else np.nan
        print(f"{name:18s} CO {r['co'][c, 0]:.2f}-{r['co'][c, -1]:.2f}  EDP {r['edp'][c, 0]:.1f}-{r['edp'][c, -1]:.1f}  "
              f"AV closes {w_close:.0f}  suction {w_suc:.0f}")

    rows = []
    for slope in (0.1, 0.2, 0.3, 0.5, 0.8, 1.2):
        eq = {}
        for c, name in enumerate(r["names"]):
            tgt = np.clip(5.0 + slope * (r["edp"][c] - edp_rest), 3.5, 8.0)
            i = int(np.argmin(np.abs(r["co"][c] - tgt)))
            eq[name] = {"speed": float(speeds[i]), "co": float(r["co"][c, i]), "target": float(tgt[i]),
                        "avo": bool(r["labels"][c, i, 0]), "suc": bool(r["labels"][c, i, 1])}
        rows.append({"slope": slope, "equilibria": eq})
        print(f"slope {slope:4.2f}: " + " | ".join(f"{k[:8]} {v['co']:.2f}@{v['speed']:.0f}{'S' if v['suc'] else ''}{'' if v['avo'] else 'c'}"
                                                 for k, v in eq.items()))
    # choose the smallest slope that raises the strenuous-exercise equilibrium by >= 15 % without suction
    ok = [x for x in rows if x["equilibria"]["exercise"]["co"] >= 1.15 * x["equilibria"]["rest"]["co"]
          and not any(v["suc"] for v in x["equilibria"].values())]
    pick = ok[0] if ok else rows[-1]
    starling = [5.0, edp_rest, pick["slope"], 3.5, 8.0]
    print(f"chosen Starling target: co_rest 5.0, edp_rest {edp_rest:.2f}, slope {pick['slope']}, clip [3.5, 8.0]"
          + ("" if ok else "  (no slope met the criterion: largest used)"))
    save_calibration("starling", {"speeds": speeds.tolist(), "names": r["names"], "co": r["co"].tolist(), "edp": r["edp"].tolist(),
                                  "clinician_speed": w_cl, "edp_rest": edp_rest, "candidates": rows, "criterion_met": bool(ok)},
                     {"starling": starling})


def calibrate_suction(_args) -> None:
    """Fraction of sampled scenarios (base parameters) that reach suction at omega_max."""
    cfg = env_v2(obs_model="measured", perf_target="fixed")
    res = {}
    for lo in (0.85, 0.80, 0.75, 0.70, 0.65):
        sc = v2_scenarios(lo)
        keys = jax.random.split(jax.random.PRNGKey(5), 512)
        scns = jax.vmap(lambda k: sample_scenario(k, P, sc, 40))(keys)
        scns = scns._replace(ev_active=jnp.zeros_like(scns.ev_active))  # base parameters only
        c40 = replace(cfg, episode_beats=40, scenario=sc)

        @jax.jit
        def run(scns):
            n = scns.base.shape[0]
            w = jnp.full(n, cfg.omega_max)
            _, st = jax.vmap(reset_with_scenario, in_axes=(0, 0, None, 0))(jax.random.split(jax.random.PRNGKey(0), n), scns, c40, w)

            def body(st, k):
                _, st2, _, _, info = jax.vmap(step_speed, in_axes=(0, 0, 0, None))(jax.random.split(k, n), st, w, c40)
                return st2, info["labels"][:, 1]

            return jax.lax.scan(body, st, jax.random.split(jax.random.PRNGKey(1), 40))[1].any(0)

        frac = float(np.asarray(run(scns)).mean())
        res[str(lo)] = frac
        print(f"Vtot lower bound {lo:.2f}: suction reachable at {cfg.omega_max:.0f} rpm in {frac:.0%} of scenarios")
    target = [lo for lo in (0.85, 0.80, 0.75, 0.70, 0.65) if 0.20 <= res[str(lo)] <= 0.40]
    lo = target[0] if target else min((0.85, 0.80, 0.75, 0.70, 0.65), key=lambda x: abs(res[str(x)] - 0.3))
    print(f"chosen Vtot range: ({lo:.2f}, 1.10)")
    save_calibration("suction", {"reachable_fraction": res}, {"vtot_lo": lo})


# ------------------------------------------------------------------------------------------- data
DATA = OUT / "data"
DET = OUT / "detectors"
TDMD = OUT / "tdm"
RUNS = OUT / "runs"
COUNTS = {"train": 512, "val": 64, "test": 128}
SEEDS = {"train": 1100, "val": 2100, "test": 3100}


def episodes(policy, cfg, n: int, seed: int) -> dict:
    from collect_data_v2 import collect

    k_scn, k_run = jax.random.split(jax.random.PRNGKey(seed))
    scns = jax.vmap(lambda k: sample_scenario(k, P, cfg.scenario, cfg.episode_beats))(jax.random.split(k_scn, n))
    d = collect(policy, scns, cfg, k_run)
    d["avo"] = d["labels"][..., 0]
    return d


def make_policy(source: str, cfg):
    from eval_agents import load_agent
    from lvadrm.agents.ppo import greedy_policy
    from lvadrm.evaluation import classical_policy

    if source == "random":
        return classical_policy("random", cfg)
    spec, _, net, params, hidden, belief = load_agent(RUNS / source)
    return greedy_policy(spec, cfg, net, params, hidden, stochastic=True, belief=belief)


def calibrate_noise(args) -> None:
    """Best learned AVO detector (CNN on waveform + features) vs flow_noise_scale, random-policy data."""
    from train_detectors import detector_metrics, train_learned

    rows = []
    for s in args.scales:
        cfg = env_v2(flow_noise_scale=s)
        pol = make_policy("random", cfg)
        tr, te = episodes(pol, cfg, args.n_train, 11), episodes(pol, cfg, 128, 12)
        _, params, predict, _ = train_learned("cnn", tr, te, epochs=12)
        p = np.asarray(predict(params, jnp.asarray(te["obs"]), jnp.asarray(te["wave"])))
        m = detector_metrics(p, te["avo"].astype(bool))
        rows.append({"scale": s, **m, "avo_rate": float(te["avo"].mean())})
        print(f"scale {s:4.1f}: AUC {m['auc']:.3f} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
              f"P(err | prev err) {m['p_error_given_prev_error']:.2f} (AVO rate {te['avo'].mean():.2f})", flush=True)
    inside = [r for r in rows if 0.93 <= r["auc"] <= 0.97]
    pick = min(inside or rows, key=lambda r: abs(r["auc"] - 0.95))
    print(f"chosen flow_noise_scale {pick['scale']} (AUC {pick['auc']:.3f})" + ("" if inside else "  (none in [0.93, 0.97])"))
    save_calibration("noise", {"rows": rows}, {"flow_noise_scale": pick["scale"]})


def collect_source(args) -> None:
    cfg = env_v2()
    pol = make_policy(args.source, cfg)
    DATA.mkdir(parents=True, exist_ok=True)
    for split, n in COUNTS.items():
        path = DATA / f"{split}_{args.source}.npz"
        if path.exists():
            continue
        t0 = time.perf_counter()
        d = episodes(pol, cfg, n, SEEDS[split] + (0 if args.source == "random" else 7))
        np.savez_compressed(path, **d)
        print(f"{path}: {n} episodes, AVO {d['avo'].mean():.2f}, SUC {d['labels'][..., 1].mean():.4f} [{time.perf_counter() - t0:.0f} s]", flush=True)


def load_data(sources=("random", "oracle_s0")) -> dict:
    out = {}
    for split in COUNTS:
        ds = [dict(np.load(DATA / f"{split}_{s}.npz")) for s in sources if (DATA / f"{split}_{s}.npz").exists()]
        out[split] = {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}
    return out


def train_detector(_args) -> None:
    from flax import serialization
    from train_detectors import detector_metrics, train_learned

    data = load_data()
    DET.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in ("cnn", "mlp"):
        _, params, predict, _ = train_learned(name, data["train"], data["val"], epochs=12)
        (DET / f"v2_{name}.msgpack").write_bytes(serialization.to_bytes(params))
        te = data["test"]
        p = np.asarray(predict(params, jnp.asarray(te["obs"]), jnp.asarray(te["wave"])))
        res[name] = detector_metrics(p, te["avo"].astype(bool))
        m = res[name]
        print(f"v2_{name}: AUC {m['auc']:.3f} sens {m['sensitivity']:.3f} spec {m['specificity']:.3f} "
              f"P(err | prev err) {m['p_error_given_prev_error']:.2f}", flush=True)
    (DET / "metrics.json").write_text(json.dumps(res, indent=2))


def train_inference(args) -> None:
    """TDM (full), TDM-prop (detector output + non-flow observations) and the recurrent AVO model of
    IBU-hist, all on the same data and the same per-beat detector outputs. Belief quality of every variant."""
    from flax import serialization
    from lvadrm.agents.belief import load_detector
    from lvadrm.inference import LL_FLOOR, ibu_update, initial_belief, naive_update, rm_state_sequence, tdm_input, tdm_sequence_logits, train_tdm

    n = args.n
    data = load_data()
    det = load_detector(args.detector)
    p = {}
    for split, d in data.items():
        E, T = d["avo"].shape
        info = {"qp_wave": jnp.asarray(d["wave"].reshape(E * T, -1))}
        pp, _, _ = jax.vmap(lambda o, w: det.step(jax.random.PRNGKey(0), jnp.zeros(()), o, {"qp_wave": w}))(
            jnp.asarray(d["obs"].reshape(E * T, -1)), info["qp_wave"])
        p[split] = np.asarray(pp).reshape(E, T)
    u = {s: rm_state_sequence(d["avo"].astype(bool), n) for s, d in data.items()}
    avo = {s: d["avo"].astype(np.int32) for s, d in data.items()}
    x = {f: {s: np.asarray(tdm_input(jnp.asarray(p[s]), jnp.asarray(data[s]["obs"]), f)) for s in data} for f in ("full", "detector")}
    TDMD.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, feats, labels, k in (("tdm", "full", u, n), ("tdm_det", "detector", u, n), ("ibu_hist", "full", avo, 2)):
        stem = f"{args.detector}_{name}_N{n}" if name != "ibu_hist" else f"{args.detector}_ibu_hist"
        model, params, hist = train_tdm(x[feats]["train"], labels["train"], x[feats]["val"], labels["val"], k, epochs=args.epochs, log=print)
        (TDMD / f"{stem}.msgpack").write_bytes(serialization.to_bytes(params))
        models[name] = (model, params, feats)
        print(f"{stem}: best val LL {max(h['val_ll'] for h in hist):.4f}", flush=True)

    def beliefs(kind, s):
        pt = jnp.asarray(p[s])
        E = pt.shape[0]
        if kind in ("naive", "ibu"):
            def body(c, ptt):
                uh, b = c
                uh, b = naive_update(uh, ptt > 0.5, n), ibu_update(b, ptt, n)
                return (uh, b), (jax.nn.one_hot(uh, n), b)
            _, (bn, bi) = jax.lax.scan(body, (jnp.zeros(E, jnp.int32), jnp.broadcast_to(initial_belief(n), (E, n))), pt.T)
            return np.swapaxes(np.asarray(bn if kind == "naive" else bi), 0, 1)
        model, params, feats = models[kind]
        logits = tdm_sequence_logits(model, params, jnp.swapaxes(jnp.asarray(x[feats][s]), 0, 1))
        if kind != "ibu_hist":
            return np.swapaxes(np.asarray(jax.nn.softmax(logits)), 0, 1)
        pa = jax.nn.softmax(logits)[..., 1]
        _, b = jax.lax.scan(lambda b, q: (ibu_update(b, q, n), ibu_update(b, q, n)), jnp.broadcast_to(initial_belief(n), (E, n)), pa)
        return np.swapaxes(np.asarray(b), 0, 1)

    res = {}
    for kind in ("naive", "ibu", "ibu_hist", "tdm_det", "tdm"):
        b = beliefs(kind, "test")
        ut = u["test"]
        pb = np.take_along_axis(b, ut[..., None], axis=-1)[..., 0]
        res[kind] = {"log_likelihood": float(np.log(np.maximum(pb, LL_FLOOR)).mean()), "accuracy": float((b.argmax(-1) == ut).mean()),
                     "mass_on_u0_when_closed": float(b[..., 0][ut >= 1].mean())}
        print(f"belief {kind:8s}: LL {res[kind]['log_likelihood']:.3f} acc {res[kind]['accuracy']:.3f} "
              f"u0 when closed {res[kind]['mass_on_u0_when_closed']:.3f}", flush=True)
    (TDMD / f"{args.detector}_N{n}_beliefs.json").write_text(json.dumps(res, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate-starling")
    sub.add_parser("calibrate-suction")
    c = sub.add_parser("calibrate-noise")
    c.add_argument("--scales", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0])
    c.add_argument("--n-train", type=int, default=512, help="training episodes (as in the pipeline)")
    c = sub.add_parser("collect")
    c.add_argument("--source", default="random", help="random or a run name in results/phase6_5/runs")
    sub.add_parser("train-detector")
    c = sub.add_parser("train-inference")
    c.add_argument("--detector", default="v2_mlp")
    c.add_argument("--n", type=int, default=3)
    c.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    t0 = time.perf_counter()
    {"calibrate-starling": calibrate_starling, "calibrate-suction": calibrate_suction, "calibrate-noise": calibrate_noise,
     "collect": collect_source, "train-detector": train_detector, "train-inference": train_inference}[args.cmd](args)
    print(f"[{time.perf_counter() - t0:.0f} s]")


if __name__ == "__main__":
    main()
