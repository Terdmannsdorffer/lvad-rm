"""AVO detectors: train the learned ones, calibrate the synthetic ones, and report their errors (GPU).

Learned: logistic (observation features), MLP and 1-D CNN (features + 64-sample pump-flow waveform),
binary cross-entropy on results/phase4/data/train.npz. Synthetic: the configurations in SYNTHETIC,
with p = P(AVO | d) calibrated on the training split.

Test metrics: accuracy, sensitivity, specificity, AUC, Brier score, expected calibration error, and
the temporal structure of the errors (P(error | error at the previous beat) vs P(error)).
Writes results/phase4/detectors/{<model>.msgpack, detectors.json}.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import common
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from lvadrm.detectors import LEARNED, SynState, SyntheticDetector, calibrate, synthetic_errors

DATA = Path("results/phase4/data")
OUT = Path("results/phase4/detectors")
SYNTHETIC = [
    SyntheticDetector(kind="iid", eps=0.1),
    SyntheticDetector(kind="iid", eps=0.2),
    SyntheticDetector(kind="markov", eps=0.1, rho=0.9),
    SyntheticDetector(kind="markov", eps=0.2, rho=0.9),
    SyntheticDetector(kind="patient", eps=0.2, spread=1.0),
    SyntheticDetector(kind="margin", eps_max=0.5, scale_ml=1.0),
]


def load(split):
    return dict(np.load(DATA / f"{split}.npz"))


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def detector_metrics(p: np.ndarray, avo: np.ndarray) -> dict:
    """p, avo: [E, T]."""
    d = p > 0.5
    err = d != avo
    bins = np.clip((p * 10).astype(int), 0, 9)
    ece = sum(abs(p[bins == b].mean() - avo[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    prev, cur = err[:, :-1], err[:, 1:]
    return {
        "accuracy": float((~err).mean()),
        "sensitivity": float(d[avo].mean()),
        "specificity": float((~d[~avo]).mean()),
        "auc": auc(p.ravel(), avo.ravel()),
        "brier": float(((p - avo) ** 2).mean()),
        "ece": float(ece),
        "p_error": float(err.mean()),
        "p_error_given_prev_error": float(cur[prev].mean()) if prev.any() else float("nan"),
        "fp_rate": float(d[~avo].mean()),
        "fn_rate": float((~d[avo]).mean()),
    }


def train_learned(name, tr, va, epochs=12, batch=4096, lr=1e-3, seed=0):
    model = LEARNED[name]()
    obs, wave, y = (tr[k].reshape(-1, tr[k].shape[-1]) if tr[k].ndim == 3 else tr[k].reshape(-1) for k in ("obs", "wave", "avo"))
    params = model.init(jax.random.PRNGKey(seed), jnp.asarray(obs[:2]), jnp.asarray(wave[:2]))
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(params, opt_state, o, w, t):
        def loss_fn(p):
            return optax.sigmoid_binary_cross_entropy(model.apply(p, o, w), t).mean()

        loss, g = jax.value_and_grad(loss_fn)(params)
        upd, opt_state = tx.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    predict = jax.jit(lambda p, o, w: jax.nn.sigmoid(model.apply(p, o, w)))
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(len(y))
        for i in range(0, len(y) - batch + 1, batch):
            idx = perm[i : i + batch]
            params, opt_state, loss = step(params, opt_state, jnp.asarray(obs[idx]), jnp.asarray(wave[idx]), jnp.asarray(y[idx], jnp.float32))
    pv = np.asarray(predict(params, jnp.asarray(va["obs"].reshape(-1, va["obs"].shape[-1])), jnp.asarray(va["wave"].reshape(-1, va["wave"].shape[-1]))))
    return model, params, predict, float(((pv > 0.5) == va["avo"].ravel()).mean())


def synthetic_detections(det: SyntheticDetector, avo: np.ndarray, v_ao: np.ndarray, seed: int) -> np.ndarray:
    """Binary detections [E, T] for whole episodes (detector state carried along each episode)."""
    E, T = avo.shape

    def episode(k, a, v):
        k0, ks = jax.random.split(k)
        eps_ep = det.eps * jax.random.uniform(k0, (), minval=1.0 - det.spread, maxval=1.0 + det.spread)

        def body(ds, xs):
            kk, at, vt = xs
            err, ds = synthetic_errors(kk, ds, at, vt, det)
            return ds, at ^ err

        return jax.lax.scan(body, SynState(jnp.asarray(False), eps_ep), (jax.random.split(ks, T), a, v))[1]

    keys = jax.random.split(jax.random.PRNGKey(seed), E)
    return np.asarray(jax.jit(jax.vmap(episode))(keys, jnp.asarray(avo), jnp.asarray(v_ao)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va, te = load("train"), load("val"), load("test")
    for d in (tr, va, te):
        d["avo"] = d["labels"][..., 0]
    out = {"learned": {}, "synthetic": {}, "prior_avo_train": float(tr["avo"].mean())}

    for name in LEARNED:
        t0 = time.perf_counter()
        model, params, predict, val_acc = train_learned(name, tr, va)
        p = np.asarray(predict(params, jnp.asarray(te["obs"]), jnp.asarray(te["wave"])))
        m = detector_metrics(p, te["avo"])
        m_by = {pol: detector_metrics(p[te["policy"] == i], te["avo"][te["policy"] == i]) for i, pol in enumerate(("random", "oracle"))}
        out["learned"][name] = m | {"val_accuracy": val_acc, "by_policy": m_by, "seconds": time.perf_counter() - t0}
        (OUT / f"{name}.msgpack").write_bytes(serialization.to_bytes(params))
        print(f"learned {name:9s}: acc {m['accuracy']:.4f} sens {m['sensitivity']:.4f} spec {m['specificity']:.4f} AUC {m['auc']:.4f} "
              f"ECE {m['ece']:.4f} | P(err) {m['p_error']:.4f} P(err|prev err) {m['p_error_given_prev_error']:.3f}", flush=True)

    for det in SYNTHETIC:
        d_tr = synthetic_detections(det, tr["avo"], tr["v_aortic"], seed=1)
        det_c = calibrate(det, tr["avo"].ravel(), d_tr.ravel())
        d_te = synthetic_detections(det_c, te["avo"], te["v_aortic"], seed=2)
        p = np.where(d_te, det_c.p_pos, det_c.p_neg)
        m = detector_metrics(p, te["avo"])
        out["synthetic"][det_c.name] = m | {"config": dataclasses.asdict(det_c)}
        print(f"synthetic {det_c.name:22s}: acc {m['accuracy']:.4f} (p_pos {det_c.p_pos:.3f}, p_neg {det_c.p_neg:.3f}) | "
              f"P(err) {m['p_error']:.4f} P(err|prev err) {m['p_error_given_prev_error']:.3f}", flush=True)

    common.save_json("detectors/detectors.json", out, phase=4)
    print("wrote", OUT / "detectors.json")


if __name__ == "__main__":
    main()
