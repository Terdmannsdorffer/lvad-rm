"""RM-state inference with each AVO detector: Naive, IBU and TDM (GPU).

For every detector (learned and synthetic) the per-beat outputs p are computed on the collected data;
Naive and IBU run on them, and a TDM (encoder + GRU) is trained on the training split with the true
RM states as targets (input: detector p and its logit, plus the agent's observation).

Test metrics (N = 5): mean log-likelihood of the true RM state (probabilities floored at 0.01),
accuracy of the most likely state, belief mass on "just opened" (u = 0) when the valve has in fact
been closed for at least one beat (overestimation of opening, H2), and the calibration of the
violation-risk state u = N - 1.
Writes results/phase4/tdm/{<detector>.msgpack, inference.json} and a figure.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from flax import serialization  # noqa: E402

from lvadrm.detectors import LEARNED, SyntheticDetector  # noqa: E402
from lvadrm.inference import LL_FLOOR, ibu_update, initial_belief, naive_update, tdm_input, tdm_sequence_logits, train_tdm  # noqa: E402
from train_detectors import DATA, synthetic_detections  # noqa: E402

N_RM = 5
METHODS = ("naive", "ibu", "tdm_det", "tdm")
OUT = Path("results/phase4/tdm")
DET_DIR = Path("results/phase4/detectors")


def load(split):
    d = dict(np.load(DATA / f"{split}.npz"))
    d["avo"] = d["labels"][..., 0]
    return d


def detector_outputs(name: str, conf: dict, splits: dict) -> dict:
    """p [E, T] per split for one detector."""
    out = {}
    if name in LEARNED:
        model = LEARNED[name]()
        tmpl = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 13)), jnp.zeros((1, 64)))
        params = serialization.from_bytes(tmpl, (DET_DIR / f"{name}.msgpack").read_bytes())
        predict = jax.jit(lambda o, w: jax.nn.sigmoid(model.apply(params, o, w)))
        for s, d in splits.items():
            E, T = d["avo"].shape
            out[s] = np.asarray(predict(jnp.asarray(d["obs"].reshape(E * T, -1)), jnp.asarray(d["wave"].reshape(E * T, -1)))).reshape(E, T)
    else:
        det = SyntheticDetector(**conf["config"])
        for i, (s, d) in enumerate(splits.items()):
            dd = synthetic_detections(det, d["avo"], d["v_aortic"], seed=10 + i)
            out[s] = np.where(dd, det.p_pos, det.p_neg).astype(np.float32)
    return out


@jax.jit
def run_naive_ibu(p):
    """p [E, T] -> (naive one-hot beliefs, IBU beliefs), both [E, T, N]."""
    E = p.shape[0]

    def body(carry, pt):
        u_hat, b = carry
        u_hat = naive_update(u_hat, pt > 0.5, N_RM)
        b = ibu_update(b, pt, N_RM)
        return (u_hat, b), (jax.nn.one_hot(u_hat, N_RM), b)

    init = (jnp.zeros(E, jnp.int32), jnp.broadcast_to(initial_belief(N_RM), (E, N_RM)))
    _, (bn, bi) = jax.lax.scan(body, init, p.T)
    return jnp.swapaxes(bn, 0, 1), jnp.swapaxes(bi, 0, 1)


def belief_metrics(b: np.ndarray, u: np.ndarray) -> dict:
    """b [E, T, N], u [E, T]."""
    pb = np.take_along_axis(b, u[..., None].astype(int), axis=-1)[..., 0]
    closed = u >= 1
    risk = u == N_RM - 1
    return {
        "log_likelihood": float(np.log(np.maximum(pb, LL_FLOOR)).mean()),
        "accuracy": float((b.argmax(-1) == u).mean()),
        "mass_on_u0_when_closed": float(b[..., 0][closed].mean()),
        "p_risk_when_at_risk": float(b[..., N_RM - 1][risk].mean()),
        "p_risk_when_not_at_risk": float(b[..., N_RM - 1][~risk].mean()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = {s: load(s) for s in ("train", "val", "test")}
    dets = json.loads((DET_DIR / "detectors.json").read_text())
    names = list(dets["learned"]) + list(dets["synthetic"])
    results = {}
    for name in names:
        t0 = time.perf_counter()
        conf = dets["synthetic"].get(name, {})
        p = detector_outputs(name, conf, splits)
        u = {s: splits[s]["u_next"].astype(np.int32) for s in splits}
        te, ute = p["test"], u["test"]
        bn, bi = (np.asarray(a) for a in run_naive_ibu(jnp.asarray(te)))
        beliefs = {"naive": bn, "ibu": bi}
        res = {}
        for variant, suffix in (("tdm_det", "__det"), ("tdm", "")):
            feats = "full" if variant == "tdm" else "detector"
            x = {s: np.asarray(tdm_input(jnp.asarray(p[s]), jnp.asarray(splits[s]["obs"]), feats)) for s in splits}
            model, params, hist = train_tdm(x["train"], u["train"], x["val"], u["val"], N_RM, epochs=30, log=print)
            (OUT / f"{name}{suffix}.msgpack").write_bytes(serialization.to_bytes(params))
            bt = np.asarray(jax.nn.softmax(tdm_sequence_logits(model, params, jnp.swapaxes(jnp.asarray(x["test"]), 0, 1))))
            beliefs[variant] = np.swapaxes(bt, 0, 1)
            res[f"{variant}_history"] = hist
        res |= {m: belief_metrics(b, ute) for m, b in beliefs.items()}
        pol = splits["test"]["policy"]
        res["by_policy"] = {
            name_p: {m: belief_metrics(b[pol == i], ute[pol == i]) for m, b in beliefs.items()}
            for i, name_p in enumerate(("random", "oracle"))
        }
        results[name] = res
        ll = "  ".join(f"{m} {res[m]['log_likelihood']:.3f}" for m in METHODS)
        mass = " ".join(f"{m} {res[m]['mass_on_u0_when_closed']:.3f}" for m in METHODS)
        print(f"{name:22s} LL {ll} | mass on u0 when closed: {mass} ({time.perf_counter() - t0:.0f} s)", flush=True)
    common.save_json("tdm/inference.json", results, phase=4)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x_ = np.arange(len(names))
    w = 0.8 / len(METHODS)
    for j, m in enumerate(METHODS):
        ax[0].bar(x_ + (j - 1.5) * w, [results[n][m]["log_likelihood"] for n in names], w, label=m)
        ax[1].bar(x_ + (j - 1.5) * w, [results[n][m]["mass_on_u0_when_closed"] for n in names], w, label=m)
    for a, t in zip(ax, ("log-likelihood of the true RM state (floor ln 0.01)", "belief on u = 0 when the valve stayed closed")):
        a.set_xticks(x_)
        a.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        a.set_title(t)
    ax[0].axhline(np.log(LL_FLOOR), c="gray", lw=0.6, ls="--")
    ax[0].legend()
    common.savefig(fig, "inference.png", phase=4)
    plt.close(fig)
    print("wrote", OUT / "inference.json")


if __name__ == "__main__":
    main()
