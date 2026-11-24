"""Phase 7: one-knob detector and estimators of the counting reward-machine state.

Detector (``LambdaDetector``): d_t = 1[T_t + b + eps_t > tau], with T_t the time the aortic valve was open in
the beat [ms], b ~ N(0, lam * sigma^2) drawn once per episode (the patient) and eps_t ~ N(0, (1 - lam) * sigma^2)
drawn every beat. tau and sigma reproduce the per-beat sensitivity and specificity of Clifford et al. (2019);
lam moves the errors from independent (0) to systematic (1) without changing the per-beat rates.

Estimators (``lvadrm.agents.belief.Belief`` interface); every one sees the same per-beat input (detection, mean
flow, speed, heart rate, last actions):
    true      the true counter (oracle)
    naive     the counter of the detections
    ibu_d     independent belief updating (IBU) with the calibrated P(AVO | d)
    ibu       IBU with a learned per-beat classifier P(AVO | input)
    ibu_hist  IBU with a learned recurrent classifier P(AVO_t | inputs up to t)
    tdm       recurrent network that predicts the counter directly
    filter    Bayes filter on (patient threshold speed, detector bias, counter), no recurrent network
    window    the last N detections (no reward machine)
    last      the last detection (for recurrent policies; no reward machine)
    voteKW    the counter of debounced detections: an opening counts if at least K of the last W beats are detected open
    tdm_win   a feedforward network that predicts the counter from the last WINDOW beats (TDM without a recurrent memory)
The vector given to the policy is the estimate followed by the current detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from lvadrm.agents.belief import Belief
from lvadrm.inference import TDM, ibu_update, initial_belief, naive_update

SENS, SPEC = 0.86, 0.93  # Clifford et al. (2019)
WINDOW = 16  # beats seen by tdm_win
KINDS = ("true", "naive", "ibu_d", "ibu", "ibu_hist", "tdm", "filter", "window", "last")


# ----------------------------------------------------------------------------------------- detector
@dataclass(frozen=True)
class LambdaDetector:
    tau: float = 0.0  # [ms]
    sigma: float = 1.0  # [ms], total noise
    lam: float = 0.0
    p_pos: float = 0.9  # calibrated P(AVO | d = 1)
    p_neg: float = 0.1
    perfect: bool = False  # the true label

    def init(self, key):
        """Per-episode bias b [ms]."""
        return jax.random.normal(key, ()) * np.sqrt(self.lam) * self.sigma

    def detect(self, key, b, avo, t_open_s):
        if self.perfect:
            return avo
        eps = jax.random.normal(key, jnp.shape(t_open_s)) * np.sqrt(1.0 - self.lam) * self.sigma
        return 1000.0 * t_open_s + b + eps > self.tau

    def advance(self, key, b):
        """State of the bias before the next beat (constant in this class)."""
        return b


@dataclass(frozen=True)
class DriftingDetector(LambdaDetector):
    """The bias changes during the episode with autocorrelation exp(-t / theta) and the same variance lam * sigma^2, so that
    the per-beat rates do not change: either slowly (a first-order autoregression) or abruptly (it is drawn again with
    probability 1 - exp(-1 / theta) in every beat). If ``cal_err`` is not None the bias is measured at the start of the
    episode, and again every ``cal_every`` beats (0: never), with a Gaussian error of SD ``cal_err`` [ms], and subtracted.
    State: (bias, last measurement, beats since the start)."""

    theta: float = 1e9  # [beats]
    jumps: bool = False
    cal_err: float | None = None
    cal_every: int = 0

    def _measure(self, key, b):
        return b + self.cal_err * jax.random.normal(key, ())

    def init(self, key):
        b = super().init(key)
        est = jnp.zeros(()) if self.cal_err is None else self._measure(jax.random.fold_in(key, 1), b)
        return b, est.astype(b.dtype), jnp.zeros((), jnp.int32)

    def detect(self, key, state, avo, t_open_s):
        return super().detect(key, state[0] - state[1], avo, t_open_s)

    def advance(self, key, state):
        b, est, t = state
        k1, k2, k3 = jax.random.split(key, 3)
        fresh = jax.random.normal(k1, ()) * np.sqrt(self.lam) * self.sigma
        rho = float(np.exp(-1.0 / self.theta))
        b = jnp.where(jax.random.uniform(k2) < 1.0 - rho, fresh, b) if self.jumps else rho * b + np.sqrt(1.0 - rho**2) * fresh
        t = t + 1
        if self.cal_err is not None and self.cal_every:
            est = jnp.where(t % self.cal_every == 0, self._measure(k3, b), est)
        return b.astype(state[0].dtype), est.astype(state[1].dtype), t


def calibrated_detector(det: LambdaDetector, s_b: float) -> LambdaDetector:
    """The detector of a patient whose bias was measured with a residual error of SD ``s_b`` [ms] (at most the SD of the bias)."""
    s_b = min(float(s_b), float(np.sqrt(det.lam) * det.sigma))
    s2 = (1.0 - det.lam) * det.sigma**2 + s_b**2
    return replace(det, sigma=float(np.sqrt(max(s2, 1e-6))), lam=float(s_b**2 / s2) if s2 > 0 else 0.0)


def fit_detector(t_open_s, avo, sens: float = SENS, spec: float = SPEC):
    """(tau, sigma) such that the per-beat rates over the given beats are ``sens`` and ``spec``."""
    from scipy.optimize import brentq
    from scipy.stats import norm

    t = 1000.0 * np.asarray(t_open_s)
    rng = np.random.default_rng(0)
    topen, tclosed = (rng.choice(x, min(20000, len(x)), replace=False) for x in (t[avo], t[~avo]))

    def tau_for(s):
        return brentq(lambda tau: norm.cdf((tclosed - tau) / s).mean() - (1.0 - spec), -50.0, 600.0)

    s = brentq(lambda s: norm.cdf((topen - tau_for(s)) / s).mean() - sens, 2.0, 300.0)
    return float(tau_for(s)), float(s)


# --------------------------------------------------------------------------------- learned models
class BeatClassifier(nn.Module):
    """Per-beat P(AVO | input) as a logit."""

    hidden: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(1)(x)[..., 0]


class WindowTDM(nn.Module):
    """Counter (logits) from the inputs of the last beats, newest first: [..., W, d]."""

    n_states: int
    hidden: int = 128

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[:-2] + (-1,))
        for _ in range(3):
            x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.n_states)(x)


def windows(x, w: int = WINDOW):
    """[E, T, d] -> [E, T, w, d + 1]: the last w inputs, newest first, and an indicator of the beats of the episode."""
    e, t, d = x.shape
    pad = np.concatenate([np.zeros((e, w - 1, d + 1), x.dtype), np.concatenate([x, np.ones((e, t, 1), x.dtype)], -1)], 1)
    return np.stack([pad[:, w - 1 - i : w - 1 - i + t] for i in range(w)], 2)


def beat_input(d, obs_next):
    """Estimator input of one beat: the detection and the (reduced) observation after the beat."""
    return jnp.concatenate([jnp.asarray(d, obs_next.dtype)[..., None], obs_next], axis=-1)


# ------------------------------------------------------------------------------------------ filter
class FilterModel(NamedTuple):
    """Response of the valve to x = log(m / omega_eff), with m the threshold speed of the patient."""

    x: Any  # [K] grid
    p_open: Any  # [K] P(open | x)
    t_ms: Any  # [K] mean open time [ms] given open
    alpha: float  # omega_eff <- alpha omega + (1 - alpha) omega_eff
    sigma_t: float  # spread of the open time around the table [ms]
    drift: float  # probability per beat of moving one cell of the m grid (each side)
    log_m: Any  # [Km] grid
    prior_m: Any  # [Km]


def filter_belief(det: LambdaDetector, n: int, fm: FilterModel, n_b: int = 21, lam: float | None = None) -> Belief:
    """Grid filter on (b, m, counter). ``lam`` is the value the filter assumes (default: the true one)."""
    lam = det.lam if lam is None else lam
    z = jnp.linspace(-2.8, 2.8, n_b)
    w_b = jnp.exp(-0.5 * z**2) / jnp.exp(-0.5 * z**2).sum()
    b_grid = z * np.sqrt(lam) * det.sigma
    s_eps = float(np.sqrt((1.0 - lam) * det.sigma**2 + fm.sigma_t**2))
    s_closed = float(np.sqrt(max(1.0 - lam, 1e-3)) * det.sigma)
    km = len(fm.log_m)
    x_tab, po_tab, t_tab = (jnp.asarray(a, jnp.float32) for a in (fm.x, fm.p_open, fm.t_ms))
    log_m = jnp.asarray(fm.log_m, jnp.float32)
    prior = w_b[:, None] * jnp.asarray(fm.prior_m, jnp.float32)[None, :]

    def init(key):
        B = jnp.zeros((n_b, km, n), jnp.float32).at[:, :, 0].set(prior)
        return B, det.init(key), jnp.zeros(()), jnp.zeros(())

    def update(key, bs, obs_next, info):
        B, b_true, w_eff, _ = bs
        d = det.detect(key, b_true, info["labels"][0], info["t_open"])
        omega = obs_next[1] * 2000.0 + 10000.0
        w_eff = jnp.where(w_eff > 0, fm.alpha * omega + (1.0 - fm.alpha) * w_eff, omega)
        x = log_m - jnp.log(w_eff)
        po = jnp.interp(x, x_tab, po_tab)[None, :]
        tm = jnp.interp(x, x_tab, t_tab)[None, :]
        p1_open = jnp.clip(ndtr((tm + b_grid[:, None] - det.tau) / s_eps), 1e-3, 1 - 1e-3)
        p1_closed = jnp.clip(ndtr((b_grid[:, None] - det.tau) / s_closed), 1e-3, 1 - 1e-3)
        lo = po * jnp.where(d, p1_open, 1.0 - p1_open)
        lc = (1.0 - po) * jnp.where(d, p1_closed, 1.0 - p1_closed)
        closed = B * lc[..., None]
        new = jnp.concatenate([(B.sum(-1) * lo)[..., None], closed[..., :-1]], axis=-1)
        new = new.at[..., -1].add(closed[..., -1])
        e = fm.drift
        up = jnp.concatenate([new[:, :1], new[:, :-1]], axis=1)
        dn = jnp.concatenate([new[:, 1:], new[:, -1:]], axis=1)
        new = (1.0 - 2.0 * e) * new + e * (up + dn)
        b_true = det.advance(jax.random.fold_in(key, 1), b_true)
        return (new / (new.sum() + 1e-30)).astype(B.dtype), b_true, w_eff.astype(bs[2].dtype), d.astype(bs[3].dtype)

    return Belief(init, update, lambda bs: jnp.concatenate([bs[0].sum((0, 1)), bs[3][None]]))


# -------------------------------------------------------------------------------------- factory
def make_estimator(kind: str, det: LambdaDetector, n: int, models: dict | None = None, hidden: int = 64, **kw) -> Belief:
    """``models``: {"beat": params, "hist": params, "tdm": params, "filter": FilterModel} as needed."""
    models = models or {}

    def wrap(init_est, step_est, est_vec):
        """Estimators with state (estimate, detector bias, last detection)."""

        def init(key):
            return init_est(), det.init(key), jnp.zeros(())

        def update(key, bs, obs_next, info):
            est, b, d_old = bs
            d = det.detect(key, b, info["labels"][0], info["t_open"])
            b = det.advance(jax.random.fold_in(key, 1), b)
            return step_est(est, d, obs_next, info), b, d.astype(d_old.dtype)

        return Belief(init, update, lambda bs: jnp.concatenate([est_vec(bs[0]), bs[2][None]]))

    def onehot(u):
        return jax.nn.one_hot(u, n)

    def ident(x):
        return x

    def ibu_step(b, p):
        return ibu_update(b, p, n, True).astype(b.dtype)

    if kind == "true":
        return wrap(lambda: jnp.int32(0), lambda u, d, o, info: info["u_next"].astype(jnp.int32), onehot)
    if kind == "naive":
        return wrap(lambda: jnp.int32(0), lambda u, d, o, info: naive_update(u, d, n, True).astype(jnp.int32), onehot)
    if vote := re.fullmatch(r"vote(\d)(\d)", kind):
        k, w = (int(c) for c in vote.groups())

        def step_vote(s, d, o, info):
            past = jnp.concatenate([d.astype(jnp.int32)[None], s[1][:-1]])
            return naive_update(s[0], past.sum() >= k, n, True).astype(jnp.int32), past

        return wrap(lambda: (jnp.int32(0), jnp.zeros(w, jnp.int32)), step_vote, lambda s: onehot(s[0]))
    if kind == "window":
        return wrap(lambda: jnp.ones(n), lambda w, d, o, info: jnp.concatenate([d.astype(w.dtype)[None], w[:-1]]), ident)
    if kind == "last":
        return wrap(lambda: jnp.zeros(0), lambda s, d, o, info: s, ident)
    if kind == "ibu_d":
        return wrap(lambda: initial_belief(n), lambda b, d, o, info: ibu_step(b, jnp.where(d, det.p_pos, det.p_neg)), ident)
    if kind == "ibu":
        clf = BeatClassifier()
        return wrap(lambda: initial_belief(n),
                    lambda b, d, o, info: ibu_step(b, jax.nn.sigmoid(clf.apply(models["beat"], beat_input(d, o)))), ident)
    if kind == "ibu_hist":
        hist = TDM(n_states=2, hidden=hidden)

        def step_hist(s, d, o, info):
            h2, lg = hist.apply(models["hist"], s[0], beat_input(d, o))
            return h2.astype(s[0].dtype), ibu_step(s[1], jax.nn.softmax(lg)[1])

        return wrap(lambda: (jnp.zeros(hidden), initial_belief(n)), step_hist, lambda s: s[1])
    if kind == "tdm":
        net = TDM(n_states=n, hidden=hidden)

        def step_tdm(s, d, o, info):
            h2, lg = net.apply(models["tdm"], s[0], beat_input(d, o))
            return h2.astype(s[0].dtype), jax.nn.softmax(lg).astype(s[1].dtype)

        return wrap(lambda: (jnp.zeros(hidden), initial_belief(n)), step_tdm, lambda s: s[1])
    if kind == "tdm_win":
        net = WindowTDM(n_states=n)
        dim = models["win"]["params"]["Dense_0"]["kernel"].shape[0] // WINDOW

        def step_win(s, d, o, info):
            x = jnp.concatenate([beat_input(d, o), jnp.ones(1, o.dtype)]).astype(s[0].dtype)
            buf = jnp.concatenate([x[None], s[0][:-1]])
            return buf, jax.nn.softmax(net.apply(models["win"], buf)).astype(s[1].dtype)

        return wrap(lambda: (jnp.zeros((WINDOW, dim)), initial_belief(n)), step_win, lambda s: s[1])
    if kind == "filter":
        return filter_belief(det, n, models["filter"], **kw)
    raise ValueError(kind)


# -------------------------------------------------------------------------- monitor controllers
class MonitorParams(NamedTuple):
    q_set: Any  # [l/min] pump-flow set point of the nominal speed
    m: Any  # risk = P(counter >= m)
    p: Any  # drop when risk > p
    w_fix: Any = 0.0  # fixed nominal speed (e.g. set after a ramp test); 0: track the flow set point


class BlindParams(NamedTuple):
    q_set: Any
    period: Any  # beats between the starts of two drops
    depth: Any  # number of 5 % steps down
    w_fix: Any = 0.0


class MonState(NamedTuple):
    w_nom: Any
    q_ema: Any
    k: Any
    bs: Any
    key: Any


Q_EMA, W_GAIN = 0.1, 0.01


def _nominal(ps: MonState, obs, q_set, cfg):
    """Slow integrator of the nominal speed on the noisy mean pump flow [l/min]."""
    q = obs[0] * 100.0 * 0.06
    q_ema = jnp.where(ps.q_ema > 0, (1 - Q_EMA) * ps.q_ema + Q_EMA * q, q)
    w = ps.w_nom * (1.0 + W_GAIN * jnp.clip(q_set - q_ema, -1.0, 1.0))
    return jnp.clip(w, cfg.omega_min, cfg.omega_max).astype(ps.w_nom.dtype), q_ema.astype(ps.q_ema.dtype)


def _move(drop, omega, w_nom):
    """One speed step within +-5 %: down when dropping, else towards the nominal speed."""
    return jnp.where(drop, omega * 0.95, jnp.clip(w_nom, omega * 0.95, omega * 1.05))


def monitor_fns(belief: Belief, cfg, n: int):
    """Nominal flow tracking; 5 % steps down while P(counter >= m) > p, back up to the nominal otherwise."""

    def init(prm: MonitorParams, key, st, obs):
        k1, k2 = jax.random.split(key)
        return MonState(st.omega, jnp.zeros(()), jnp.int32(0), belief.init(k1), k2)

    def act(prm: MonitorParams, key, st, obs, ps):
        w_nom, q_ema = _nominal(ps, obs, prm.q_set, cfg)
        w_nom = jnp.where(prm.w_fix > 0, prm.w_fix, w_nom).astype(w_nom.dtype)
        risk = jnp.where(jnp.arange(n) >= prm.m, belief.vector(ps.bs)[:n], 0.0).sum()
        return _move(risk > prm.p, st.omega, w_nom), 0.0, 0.0, ps._replace(w_nom=w_nom, q_ema=q_ema)

    def observe(prm, ps, st, info, obs_next):
        key, k = jax.random.split(ps.key)
        return ps._replace(bs=belief.update(k, ps.bs, obs_next, info), key=key)

    return init, act, observe


def blind_fns(cfg):
    """The same nominal tracking with periodic drops and no detector."""

    def init(prm: BlindParams, key, st, obs):
        return MonState(st.omega, jnp.zeros(()), jnp.int32(0), jnp.zeros(()), key)

    def act(prm: BlindParams, key, st, obs, ps):
        w_nom, q_ema = _nominal(ps, obs, prm.q_set, cfg)
        w_nom = jnp.where(prm.w_fix > 0, prm.w_fix, w_nom).astype(w_nom.dtype)
        drop = ps.k % prm.period >= prm.period - prm.depth
        return _move(drop, st.omega, w_nom), 0.0, 0.0, ps._replace(w_nom=w_nom, q_ema=q_ema, k=ps.k + 1)

    def observe(prm, ps, st, info, obs_next):
        return ps

    return init, act, observe
