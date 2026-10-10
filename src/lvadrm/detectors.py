"""AVO detectors: the abstraction model M of Li et al. (2024) for the proposition AVO (HANDOFF section 10).

A detector maps what can be observed after a beat to (p, d): p = probability that the aortic valve
opened in that beat, d = binary detection. The counting RM only transitions on AVO, so this is all the
belief updates need (SUC / BACK only enter the reward, which is known during training).

Two families share one interface (``Detector``: init(key) -> dstate, step(key, dstate, obs_next,
info) -> (p, d, dstate)):

* synthetic detectors corrupt the true label with controlled errors:
    iid      independent flips with rate eps;
    markov   a two-state error process with stationary rate eps and persistence rho
             (errors repeat beat after beat);
    (with eps_fp set, eps is the false-negative rate and eps_fp the false-positive rate, e.g. the
    clinical AVO detector of Clifford et al. 2019: sensitivity 86 %, specificity 93 %)
    patient  per-episode error rate drawn uniformly in eps * [1 - spread, 1 + spread];
    margin   error probability eps_max * exp(-|V_ao - eps_avo| / scale): openings near the threshold
             are confused, and since the physiology changes slowly the errors are correlated in time.
  Their p is the per-beat calibrated posterior P(AVO | d), estimated on data (``calibrate``): exactly
  what a calibrated per-beat classifier reports when it ignores the temporal structure of its errors.
* learned detectors are classifiers on the measured pump signals of the beat (features, waveform).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.labels import EPS_AVO_ML


class Detector(NamedTuple):
    init: Callable
    step: Callable


# ----------------------------------------------------------------------------------------- synthetic
@dataclass(frozen=True)
class SyntheticDetector:
    kind: str = "iid"  # iid, markov, patient or margin
    eps: float = 0.1  # error rate (false-negative rate when eps_fp is set)
    eps_fp: float | None = None  # false-positive rate (iid / markov); None: same as eps
    rho: float = 0.9  # markov persistence: P(error | error at the previous beat)
    spread: float = 1.0  # patient
    eps_max: float = 0.5  # margin
    scale_ml: float = 1.0  # margin
    p_pos: float = 0.9  # calibrated P(AVO | d = 1), see calibrate()
    p_neg: float = 0.1  # calibrated P(AVO | d = 0)

    @property
    def name(self) -> str:
        if self.eps_fp is not None and self.kind in ("iid", "markov"):
            return f"{self.kind}_fn{self.eps}_fp{self.eps_fp}" + (f"_r{self.rho}" if self.kind == "markov" else "")
        if self.kind == "markov":
            return f"markov_e{self.eps}_r{self.rho}"
        if self.kind == "margin":
            return f"margin_m{self.eps_max}_s{self.scale_ml}"
        if self.kind == "patient":
            return f"patient_e{self.eps}_s{self.spread}"
        return f"iid_e{self.eps}"


class SynState(NamedTuple):
    err: jax.Array  # error indicator of the previous beat (markov)
    eps_ep: jax.Array  # episode error rate (patient)


def synthetic_errors(key, ds: SynState, avo, v_aortic, det: SyntheticDetector, eps_avo: float = EPS_AVO_ML):
    """Error indicator for one beat and the next detector state."""
    u = jax.random.uniform(key, ())
    eps = det.eps if det.eps_fp is None else jnp.where(avo, det.eps, det.eps_fp)  # class-conditional rates
    if det.kind == "iid":
        p_err = jnp.asarray(eps)
    elif det.kind == "markov":
        p_enter = eps * (1.0 - det.rho) / (1.0 - eps)  # keeps the stationary rate at eps
        p_err = jnp.where(ds.err, det.rho, p_enter)
    elif det.kind == "patient":
        p_err = ds.eps_ep
    elif det.kind == "margin":
        p_err = det.eps_max * jnp.exp(-jnp.abs(v_aortic - eps_avo) / det.scale_ml)
    else:
        raise ValueError(det.kind)
    err = u < p_err
    return err, ds._replace(err=err)


def synthetic_detector(det: SyntheticDetector, eps_avo: float = EPS_AVO_ML) -> Detector:
    def init(key):
        eps_ep = det.eps * jax.random.uniform(key, (), minval=1.0 - det.spread, maxval=1.0 + det.spread)
        return SynState(jnp.asarray(False), eps_ep)

    def step(key, ds, obs_next, info):
        avo = info["labels"][0]
        err, ds = synthetic_errors(key, ds, avo, info["v_aortic"], det, eps_avo)
        d = avo ^ err
        p = jnp.where(d, det.p_pos, det.p_neg)
        return p, d, ds

    return Detector(init, step)


def calibrate(det: SyntheticDetector, avo: np.ndarray, d: np.ndarray) -> SyntheticDetector:
    """Per-beat calibrated outputs P(AVO | d) from labelled detections."""
    p_pos = float(avo[d].mean()) if d.any() else 0.5
    p_neg = float(avo[~d].mean()) if (~d).any() else 0.5
    return replace(det, p_pos=float(np.clip(p_pos, 1e-3, 1 - 1e-3)), p_neg=float(np.clip(p_neg, 1e-3, 1 - 1e-3)))


# ----------------------------------------------------------------------------------------- learned
class LogisticDetector(nn.Module):
    @nn.compact
    def __call__(self, obs, wave):
        return nn.Dense(1)(obs)[..., 0]


class MLPDetector(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, obs, wave):
        x = jnp.concatenate([obs, wave], axis=-1)
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)[..., 0]


class CNNDetector(nn.Module):
    """1-D convolutions over the 64-sample pump-flow waveform, concatenated with the features."""

    channels: int = 16

    @nn.compact
    def __call__(self, obs, wave):
        lead = wave.shape[:-1]  # any number of leading batch axes (including none)
        x = wave.reshape(-1, wave.shape[-1], 1)
        for _ in range(3):
            x = nn.relu(nn.Conv(self.channels, kernel_size=(5,), padding="SAME")(x))
            x = nn.max_pool(x, window_shape=(2,), strides=(2,))
        x = jnp.concatenate([x.reshape(x.shape[0], -1), obs.reshape(-1, obs.shape[-1])], axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[:, 0].reshape(lead)


LEARNED = {"logistic": LogisticDetector, "mlp": MLPDetector, "cnn": CNNDetector}


def learned_detector(model: nn.Module, params) -> Detector:
    def init(key):
        return jnp.zeros(())

    def step(key, ds, obs_next, info):
        p = jax.nn.sigmoid(model.apply(params, obs_next, info["qp_wave"]))
        return p, p > 0.5, ds

    return Detector(init, step)
