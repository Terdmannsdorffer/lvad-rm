"""Belief over the RM state given to the policy (Li et al. 2024 framework; HANDOFF section 10).

A ``Belief`` keeps a per-environment state and is updated after every beat from what can be observed
(the agent's observation and the measured pump-flow waveform) through an AVO detector:

    true    the true RM state (oracle)
    naive   u_hat <- delta(u_hat, d) with binary detections
    ibu     exact belief update assuming independent detector errors
    tdm     a GRU trained to output the RM-state distribution from the detector and observation history
    tdm_det the same with only the detector output and the non-flow observations (speed, HR, actions)
    ibu_hist IBU fed by a recurrent proposition model: the TDM network trained to predict AVO (2 classes)
            from the same inputs and history as tdm (Phase 6.5: isolates "predict the RM state" from
            "use the history")

Detector names starting with "v2_" are Phase 6.5 learned detectors (results/phase6_5/detectors).

Every episode starts in u_0, which is known, so all beliefs start as one-hot(0).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from flax import serialization

from lvadrm.detectors import LEARNED, Detector, SyntheticDetector, learned_detector, synthetic_detector
from lvadrm.inference import TDM, ibu_update, initial_belief, naive_update, tdm_input

DET_DIR = Path("results/phase4/detectors")
DET_DIR_P6 = Path("results/phase6/detectors")
DET_DIR_P65 = Path("results/phase6_5/detectors")
TDM_DIR = Path("results/phase4/tdm")
OBS_DIM = 13
WAVE_DIM = 64


class Belief(NamedTuple):
    init: Callable  # key -> bstate
    update: Callable  # (key, bstate, obs_next, info) -> bstate
    vector: Callable  # bstate -> [N]


def true_belief(n: int) -> Belief:
    return Belief(
        init=lambda key: jnp.int32(0),
        update=lambda key, u, obs_next, info: info["u_next"],
        vector=lambda u: jax.nn.one_hot(u, n),
    )


def naive_belief(det: Detector, n: int) -> Belief:
    def update(key, bs, obs_next, info):
        u_hat, ds = bs
        _, d, ds = det.step(key, ds, obs_next, info)
        return naive_update(u_hat, d, n), ds

    return Belief(lambda key: (jnp.int32(0), det.init(key)), update, lambda bs: jax.nn.one_hot(bs[0], n))


def ibu_belief(det: Detector, n: int) -> Belief:
    def update(key, bs, obs_next, info):
        b, ds = bs
        p, _, ds = det.step(key, ds, obs_next, info)
        return ibu_update(b, p, n).astype(b.dtype), ds

    return Belief(lambda key: (initial_belief(n), det.init(key)), update, lambda bs: bs[0])


def tdm_belief(det: Detector, n: int, model: TDM, params, features: str = "full") -> Belief:
    def init(key):
        return jnp.zeros(model.hidden), det.init(key), initial_belief(n)

    def update(key, bs, obs_next, info):
        h, ds, vec = bs
        p, _, ds = det.step(key, ds, obs_next, info)
        h_new, logits = model.apply(params, h, tdm_input(p, obs_next, features))
        return h_new.astype(h.dtype), ds, jax.nn.softmax(logits).astype(vec.dtype)

    return Belief(init, update, lambda bs: bs[2])


def ibu_hist_belief(det: Detector, n: int, model: TDM, params) -> Belief:
    """IBU update with p(AVO) from a recurrent 2-class model over the detector + observation history."""

    def init(key):
        return jnp.zeros(model.hidden), det.init(key), initial_belief(n)

    def update(key, bs, obs_next, info):
        h, ds, b = bs
        p, _, ds = det.step(key, ds, obs_next, info)
        h_new, logits = model.apply(params, h, tdm_input(p, obs_next, "full"))
        p_avo = jax.nn.softmax(logits)[..., 1]
        return h_new.astype(h.dtype), ds, ibu_update(b, p_avo, n).astype(b.dtype)

    return Belief(init, update, lambda bs: bs[2])


def synthetic_config(name: str) -> dict:
    """Calibrated synthetic detector configuration (Phase 4 file first, then Phase 6 additions)."""
    for f in (DET_DIR / "detectors.json", DET_DIR_P6 / "detectors.json"):
        if f.exists():
            syn = json.loads(f.read_text())["synthetic"]
            if name in syn:
                return syn[name]["config"]
    raise KeyError(f"no calibrated synthetic detector {name!r}")


def load_detector(name: str) -> Detector:
    base, folder = (name[3:], DET_DIR_P65) if name.startswith("v2_") else (name, DET_DIR)
    if base in LEARNED:
        model = LEARNED[base]()
        tmpl = model.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)), jnp.zeros((1, WAVE_DIM)))
        return learned_detector(model, serialization.from_bytes(tmpl, (folder / f"{name}.msgpack").read_bytes()))
    return synthetic_detector(SyntheticDetector(**synthetic_config(name)))


def load_tdm(name: str, n: int, hidden: int = 128, features: str = "full", path: str | None = None):
    """TDM parameters from ``path`` or, by default, the Phase 4 file of detector ``name``."""
    from lvadrm.env import PUMP_FLOW_FEATURES

    model = TDM(n_states=n, hidden=hidden)
    d_in = 2 + (OBS_DIM if features == "full" else OBS_DIM - PUMP_FLOW_FEATURES)
    tmpl = model.init(jax.random.PRNGKey(0), jnp.zeros((1, hidden)), jnp.zeros((1, d_in)))
    suffix = "" if features == "full" else "__det"
    f = Path(path) if path else TDM_DIR / f"{name}{suffix}.msgpack"
    return model, serialization.from_bytes(tmpl, f.read_bytes())


def make_belief(kind: str, n: int, detector: str | None = None, tdm_path: str | None = None) -> Belief:
    if kind == "true":
        return true_belief(n)
    det = load_detector(detector)
    if kind == "naive":
        return naive_belief(det, n)
    if kind == "ibu":
        return ibu_belief(det, n)
    if kind == "tdm":
        return tdm_belief(det, n, *load_tdm(detector, n, path=tdm_path))
    if kind == "tdm_det":
        return tdm_belief(det, n, *load_tdm(detector, n, features="detector", path=tdm_path), features="detector")
    if kind == "ibu_hist":
        return ibu_hist_belief(det, n, *load_tdm(detector, 2, path=tdm_path))
    raise ValueError(kind)
