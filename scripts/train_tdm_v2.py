"""Phase 6 TDMs: trained on Phase 4 data + wide-scenario and on-policy data, any detector and N (GPU).

    python scripts/train_tdm_v2.py --detector mlp --n 5
    python scripts/train_tdm_v2.py --detector iid_e0.3 --n 5   # synthetic: calibrated here if new

Synthetic detector names: iid_e<eps>, markov_e<eps>_r<rho>. New ones are calibrated on the Phase 4
training split and added to results/phase6/detectors/detectors.json.

Belief quality (log-likelihood of the true RM state, etc.) of Naive, IBU, the Phase 4 TDM (N = 5,
if it exists) and the new TDM on every test set: p4 (default scenarios), wide_random, wide_oracle,
onpolicy_default, onpolicy_wide.
Writes results/phase6/tdm/{detector}_N{n}_v2.{msgpack,json}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from lvadrm.agents.belief import DET_DIR_P6, load_tdm, synthetic_config
from lvadrm.detectors import LEARNED, SyntheticDetector, calibrate
from lvadrm.inference import LL_FLOOR, ibu_update, initial_belief, naive_update, rm_state_sequence, tdm_input, tdm_sequence_logits, train_tdm
from train_detectors import synthetic_detections
from train_tdm import detector_outputs

P4 = Path("results/phase4/data")
P6 = Path("results/phase6/data")
OUT = Path("results/phase6/tdm")
V2_SOURCES = ("wide_random", "wide_oracle", "onpolicy_default", "onpolicy_wide")


def load_all():
    """{split: {source: data}} with AVO extracted."""
    data = {}
    for split in ("train", "val", "test"):
        data[split] = {"p4": dict(np.load(P4 / f"{split}.npz"))}
        for src in V2_SOURCES:
            data[split][src] = dict(np.load(P6 / f"{split}_{src}.npz"))
        for d in data[split].values():
            d["avo"] = d["labels"][..., 0]
    return data


def synthetic_conf(name: str, p4_train: dict) -> dict:
    """Calibrated config of a synthetic detector, creating and storing it if needed."""
    try:
        return {"config": synthetic_config(name)}
    except KeyError:
        pass
    m = re.fullmatch(r"(iid|markov)_e([\d.]+)(?:_r([\d.]+))?", name)
    if not m:
        raise ValueError(f"cannot parse synthetic detector name {name!r}")
    det = SyntheticDetector(kind=m.group(1), eps=float(m.group(2)), rho=float(m.group(3) or 0.9))
    d_tr = synthetic_detections(det, p4_train["avo"], p4_train["v_aortic"], seed=1)
    det = calibrate(det, p4_train["avo"].ravel(), d_tr.ravel())
    assert det.name == name, (det.name, name)
    DET_DIR_P6.mkdir(parents=True, exist_ok=True)
    f = DET_DIR_P6 / "detectors.json"
    store = json.loads(f.read_text()) if f.exists() else {"synthetic": {}}
    store["synthetic"][name] = {"config": dataclasses.asdict(det)}
    f.write_text(json.dumps(store, indent=2))
    return store["synthetic"][name]


def naive_ibu(p: np.ndarray, n: int):
    E = p.shape[0]

    @jax.jit
    def run(p):
        def body(carry, pt):
            u_hat, b = carry
            u_hat = naive_update(u_hat, pt > 0.5, n)
            b = ibu_update(b, pt, n)
            return (u_hat, b), (jax.nn.one_hot(u_hat, n), b)

        init = (jnp.zeros(E, jnp.int32), jnp.broadcast_to(initial_belief(n), (E, n)))
        _, (bn, bi) = jax.lax.scan(body, init, p.T)
        return jnp.swapaxes(bn, 0, 1), jnp.swapaxes(bi, 0, 1)

    return (np.asarray(a) for a in run(jnp.asarray(p)))


def metrics(b: np.ndarray, u: np.ndarray, n: int) -> dict:
    pb = np.take_along_axis(b, u[..., None], axis=-1)[..., 0]
    closed, risk = u >= 1, u == n - 1
    return {
        "log_likelihood": float(np.log(np.maximum(pb, LL_FLOOR)).mean()),
        "accuracy": float((b.argmax(-1) == u).mean()),
        "mass_on_u0_when_closed": float(b[..., 0][closed].mean()),
        "p_risk_when_at_risk": float(b[..., n - 1][risk].mean()),
    }


def tdm_beliefs(model, params, x):
    return np.swapaxes(np.asarray(jax.nn.softmax(tdm_sequence_logits(model, params, jnp.swapaxes(jnp.asarray(x), 0, 1)))), 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all()
    conf = {} if args.detector in LEARNED else synthetic_conf(args.detector, data["train"]["p4"])

    x, u = {}, {}
    for split, sources in data.items():
        p = detector_outputs(args.detector, conf, sources)
        x[split] = {s: np.asarray(tdm_input(jnp.asarray(p[s]), jnp.asarray(sources[s]["obs"]))) for s in sources}
        u[split] = {s: rm_state_sequence(sources[s]["avo"], args.n) for s in sources}
        x[split]["_p"] = p
    cat = lambda d: np.concatenate([d[s] for s in ("p4", *V2_SOURCES)])  # noqa: E731
    model, params, hist = train_tdm(cat(x["train"]), cat(u["train"]), cat(x["val"]), cat(u["val"]), args.n, epochs=args.epochs, log=print)
    stem = f"{args.detector}_N{args.n}_v2"
    (OUT / f"{stem}.msgpack").write_bytes(serialization.to_bytes(params))

    old = None
    if args.n == 5 and (Path("results/phase4/tdm") / f"{args.detector}.msgpack").exists():
        old = load_tdm(args.detector, 5)
    res = {"detector": args.detector, "n": args.n, "history": hist, "test": {}}
    for s in ("p4", *V2_SOURCES):
        ut = u["test"][s]
        bn, bi = naive_ibu(x["test"]["_p"][s], args.n)
        row = {"naive": metrics(bn, ut, args.n), "ibu": metrics(bi, ut, args.n), "tdm_v2": metrics(tdm_beliefs(model, params, x["test"][s]), ut, args.n)}
        if old is not None:
            row["tdm_p4"] = metrics(tdm_beliefs(*old, x["test"][s]), ut, args.n)
        res["test"][s] = row
        print(f"{stem} test {s:17s} LL " + "  ".join(f"{k} {v['log_likelihood']:.3f}" for k, v in row.items())
              + " | u0 when closed " + " ".join(f"{k} {v['mass_on_u0_when_closed']:.3f}" for k, v in row.items()), flush=True)
    common.save_json(f"tdm/{stem}.json", res, phase=6)


if __name__ == "__main__":
    main()
