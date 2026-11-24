"""Phase 7, stage 2: exploratory data, detector calibration, one-beat leak check and filter tables (GPU).

    python scripts/phase7_calibrate.py
Writes results/phase7/data/round1.npz and results/phase7/calibration.json.

Leak check (professor, 3.3): how well can a classifier recover AVO from one beat of (a) what the policy sees
("reduced": mean flow, speed, HR, last actions), (b) the waveform statistics used in Phases 3-6.5, with the
same noise. The detector only matters if (a) is clearly worse than the detector.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
from phase7_common import OUT, auc, rollout, scenarios, start_speeds

from lvadrm.estimators import BeatClassifier, BlindParams, LambdaDetector, MonitorParams, blind_fns, fit_detector, make_estimator, monitor_fns
from lvadrm.model import initial_state, total_volume
from lvadrm.integrate import simulate_beat
from lvadrm.params import make_params
from lvadrm.patients import HF_PATIENT as P
from lvadrm.presets import env_v3

N = 10
KEEP = ("avo", "t_open", "obs", "params", "omega", "back", "co", "u_next", "qp", "tc")


def random_fns(cfg):
    acts = jnp.asarray(cfg.actions)

    def init(prm, key, st, obs):
        return jnp.zeros(())

    def act(prm, key, st, obs, ps):
        a = jax.random.randint(key, (), 0, len(cfg.actions))
        return st.omega * (1.0 + acts[a]), 0.0, 0.0, ps

    return init, act, lambda prm, ps, st, info, obs2: ps


def collect(key, cfg, e_mon=1024, e_blind=1536, e_rand=512):
    """Randomized controllers around the operating region, all without using a detector."""
    det = LambdaDetector(perfect=True)
    parts = []
    rng = np.random.default_rng(1)
    u = lambda lo, hi, n: jnp.asarray(rng.uniform(lo, hi, n), jnp.float32)  # noqa: E731
    i = lambda lo, hi, n: jnp.asarray(rng.integers(lo, hi + 1, n))  # noqa: E731
    for name, e in (("monitor", e_mon), ("blind", e_blind), ("random", e_rand)):
        key, k1, k2, k3 = jax.random.split(key, 4)
        scn, w0 = scenarios(k1, cfg, e), start_speeds(k2, cfg, e)
        if name == "monitor":
            fns, prm = monitor_fns(make_estimator("true", det, N), cfg, N), MonitorParams(u(3.5, 6.5, e), i(1, N - 1, e), jnp.full(e, 0.5), jnp.zeros(e))
        elif name == "blind":
            fns, prm = blind_fns(cfg), BlindParams(u(3.5, 6.5, e), i(5, 20, e), i(1, 4, e), jnp.zeros(e))
        else:
            fns, prm = random_fns(cfg), jnp.zeros(e)
        o = rollout(fns, prm, scn, cfg, k3, w0, keep=KEEP)
        print(f"{name}: AVO {o['avo'].mean():.3f} BACK {o['back'].mean():.3f} CO {o['co'].mean():.2f} omega {o['omega'].mean():.0f}", flush=True)
        parts.append(o | {"src": np.full(o["avo"].shape, name)})
    return {k: np.concatenate([p[k] for p in parts], axis=1) for k in parts[0]}


def fit_classifier(x, y, xt, yt, hidden=64, steps=3000, linear=False, seed=0):
    mu, sd = x.mean(0), x.std(0) + 1e-6
    x, xt = (x - mu) / sd, (xt - mu) / sd
    if linear:
        from scipy.optimize import minimize

        x1 = np.c_[x, np.ones(len(x))]

        def f(w):
            z = x1 @ w
            return np.logaddexp(0, -z * (2 * y - 1)).mean(), x1.T @ (1 / (1 + np.exp(-z)) - y) / len(y)

        w = minimize(f, np.zeros(x1.shape[1]), jac=True, method="L-BFGS-B").x
        s = np.c_[xt, np.ones(len(xt))] @ w
    else:
        net = BeatClassifier(hidden)
        params = net.init(jax.random.PRNGKey(seed), jnp.asarray(x[:1]))
        tx = optax.adam(1e-3)
        opt = tx.init(params)

        @jax.jit
        def step(params, opt, xb, yb):
            g = jax.grad(lambda p: optax.sigmoid_binary_cross_entropy(net.apply(p, xb), yb).mean())(params)
            up, opt = tx.update(g, opt)
            return optax.apply_updates(params, up), opt

        rng = np.random.default_rng(seed)
        for _ in range(steps):
            idx = rng.integers(0, len(x), 4096)
            params, opt = step(params, opt, jnp.asarray(x[idx]), jnp.asarray(y[idx], jnp.float32))
        s = np.asarray(net.apply(params, jnp.asarray(xt)))
    thr = np.quantile(s[~yt], 0.93)
    best = max(((s[yt] > t).mean() + (s[~yt] <= t).mean()) / 2 for t in np.quantile(s, np.linspace(0.02, 0.98, 60)))
    return {"auc": auc(s, yt), "acc": float(((s > 0) == yt).mean()), "sens_at_spec93": float((s[yt] > thr).mean()), "balanced_acc": float(best)}


def w_avo_regression(n=3000, seed=2):
    """omega_AVO of the periodic regime as a quadratic function of log parameters."""
    rng = np.random.default_rng(seed)
    e, hr, rs, vt = rng.uniform(0.5, 1.3, n) * 0.7 * P.emax, rng.uniform(60, 105, n), rng.uniform(0.75, 1.35, n) * P.rs, rng.uniform(0.86, 1.07, n) * P.vtot

    def opens(w, e, hr, rs, vt):
        p = make_params(emax=e, emin=P.emin, v0=P.v0, hr=hr, rs=rs)

        def body(y, _):
            y, s, _ = simulate_beat(y, p, w, 0.0, (vt - total_volume(y, p)) * hr / 60.0, 0.0, n_seg=64, n_inner=105, unroll=8)
            return y, s.v_aortic

        return jax.lax.scan(body, initial_state(p, vt).astype(jnp.float32), None, length=20)[1][-1] > 0.5

    f = jax.jit(jax.vmap(opens))
    a = [jnp.asarray(x, jnp.float32) for x in (e, hr, rs, vt)]
    lo, hi = np.full(n, 4500.0), np.full(n, 11000.0)
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        ok = np.asarray(f(jnp.asarray(mid, jnp.float32), *a))
        lo, hi = np.where(ok, mid, lo), np.where(ok, hi, mid)
    X = design(rs, hr, e, vt)
    c, *_ = np.linalg.lstsq(X[: n // 2], np.log(lo[: n // 2]), rcond=None)
    err = X[n // 2 :] @ c - np.log(lo[n // 2 :])
    print(f"omega_AVO regression: rms error {100 * err.std():.2f} %, range {lo.min():.0f}-{lo.max():.0f}")
    return c, float(err.std())


def design(rs, hr, e, vt):
    z = [np.log(rs / P.rs), np.log(hr / 80.0), np.log(e / P.emax), np.log(vt / P.vtot)]
    cols = [np.ones_like(z[0])] + z + [z[i] * z[j] for i in range(4) for j in range(i, 4)]
    return np.stack(cols, -1)


def main() -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    cfg = env_v3(N, "unknown")
    d = collect(jax.random.PRNGKey(70), cfg)
    T, E = d["avo"].shape
    near = d["src"] != "random"
    skip = np.arange(T)[:, None] >= 20
    sel = near & skip
    topen = d["t_open"][sel & d["avo"]] * 1000
    print("open time [ms] in open beats, percentiles 1/5/25/50/75/95:", np.percentile(topen, [1, 5, 25, 50, 75, 95]).round(0))
    print("closed beats with 0 < T:", float((d["t_open"][sel & ~d["avo"]] > 0).mean()))
    tau, sigma = fit_detector(d["t_open"][sel], d["avo"][sel])
    print(f"detector: tau {tau:.1f} ms, sigma {sigma:.1f} ms")
    rng = np.random.default_rng(3)
    rates = {}
    for lam in (0.0, 0.5, 0.9, 1.0):
        b = rng.normal(size=E) * np.sqrt(lam) * sigma
        det = 1000 * d["t_open"] + b + rng.normal(size=(T, E)) * np.sqrt(1 - lam) * sigma > tau
        rates[str(lam)] = [float(det[sel & d["avo"]].mean()), float((~det)[sel & ~d["avo"]].mean())]
        if lam == 0.0:
            dd, av = det[sel], d["avo"][sel]
            p_pos, p_neg = float(av[dd].mean()), float(av[~dd].mean())
    print("sens / spec by lambda:", rates, "P(AVO|d=1), P(AVO|d=0):", round(p_pos, 3), round(p_neg, 3))

    # filter tables
    y = d["avo"]
    c, reg_err = w_avo_regression()
    par = d["params"]  # rs, hr, emax, vtot
    m = np.exp(design(par[..., 0], par[..., 1], par[..., 2], par[..., 3]) @ c)
    best = None
    for alpha in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        w = d["omega"].copy()
        for t in range(1, T):
            w[t] = alpha * d["omega"][t] + (1 - alpha) * w[t - 1]
        x = np.log(m / w)
        a = auc(x[sel], y[sel])
        if best is None or a > best[0]:
            best = (a, alpha, x)
    a, alpha, x = best
    # leak check
    tr = np.zeros(E, bool)
    tr[rng.permutation(E)[: E // 2]] = True
    # only controllers that never use the label: otherwise their actions give it away
    free = skip & (d["src"] == "blind")
    # and only beats whose speed is within 10 % of the patient's threshold, where controllers operate
    free = free & (np.abs(x) < 0.10)
    print(f"leak check beats: {free.sum()}, AVO {y[free].mean():.2f}")
    m_tr, m_te = free & tr[None], free & ~tr[None]
    qp = d["qp"]
    stats = np.stack([qp.min(-1), qp.max(-1), qp.std(-1)], -1)
    det0 = (1000 * d["t_open"] + rng.normal(size=(T, E)) * sigma > tau).astype(float)[..., None]
    feats = {
        "reduced (policy)": d["obs"],
        "reduced without mean flow": d["obs"][..., 1:],
        "reduced + flow min/max/std": np.concatenate([d["obs"], stats], -1),
        "speed + flow mean/min/max (feedback)": np.concatenate([d["obs"][..., :2], stats[..., :2]], -1),
        "reduced + detection": np.concatenate([d["obs"], det0], -1),
        "detection only": det0,
    }
    leak = {}
    for name, f in feats.items():
        f = f.astype(np.float32)
        leak[name] = {"mlp": fit_classifier(f[m_tr], y[m_tr], f[m_te], y[m_te]), "logistic": fit_classifier(f[m_tr], y[m_tr], f[m_te], y[m_te], linear=True)}
        print(f"leak {name:38s}", {k: round(v, 3) for k, v in leak[name]["mlp"].items()}, "| logistic", {k: round(v, 3) for k, v in leak[name]["logistic"].items()}, flush=True)
    chosen_auc = leak["reduced (policy)"]["mlp"]["auc"]

    edges = np.linspace(-0.25, 0.25, 41)
    mid = 0.5 * (edges[1:] + edges[:-1])
    xs, ys, ts = x[skip[:, 0]], y[skip[:, 0]], 1000 * d["t_open"][skip[:, 0]]
    po, tm, tsd = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (xs >= lo) & (xs < hi)
        po.append((ys[b].sum() + 0.5) / (b.sum() + 1.0))
        o = b & ys
        tm.append(ts[o].mean() if o.sum() > 20 else np.nan)
        tsd.append(ts[o].std() if o.sum() > 20 else np.nan)
    po, tm = np.array(po), np.array(tm)
    ok = np.isfinite(tm)
    tm = np.interp(mid, mid[ok], tm[ok])
    po = np.maximum.accumulate(np.clip(po, 0.005, 0.995))
    sigma_t = float(np.nanmedian(tsd))
    log_m = np.linspace(np.log(6000), np.log(9800), 41)
    h, _ = np.histogram(np.log(m[0]), bins=np.r_[log_m - 0.5 * (log_m[1] - log_m[0]), log_m[-1] + 0.5 * (log_m[1] - log_m[0])])
    prior = (h + 1.0) / (h + 1.0).sum()
    dm = np.diff(np.log(m), axis=0) / (log_m[1] - log_m[0])
    drift = float(0.5 * (dm**2).mean())
    print(f"filter: AUC of x for AVO {a:.4f} (alpha {alpha}), sigma_T {sigma_t:.1f} ms, drift per side {drift:.5f}")
    print("P(open|x):", dict(zip(mid.round(3)[10:31:2], po.round(3)[10:31:2])))
    cal = {"tau": tau, "sigma": sigma, "p_pos": p_pos, "p_neg": p_neg, "rates": rates, "leak": leak, "chosen_auc": chosen_auc,
           "open_ms_percentiles": np.percentile(topen, [5, 25, 50, 75, 95]).tolist(), "w_avo_coef": c.tolist(), "w_avo_reg_rms": reg_err,
           "filter_auc": a,
           "filter": dict(x=mid.tolist(), p_open=po.tolist(), t_ms=tm.tolist(), alpha=alpha, sigma_t=sigma_t, drift=max(drift, 0.002),
                          log_m=log_m.tolist(), prior_m=prior.tolist())}
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))
    np.savez_compressed(OUT / "data" / "round1.npz", avo=d["avo"], t_open=d["t_open"].astype(np.float32), obs=d["obs"].astype(np.float32),
                        u=d["u_next"], src=d["src"][0], omega=d["omega"].astype(np.float32), params=d["params"].astype(np.float32))
    print("wrote", OUT / "calibration.json")


if __name__ == "__main__":
    main()
