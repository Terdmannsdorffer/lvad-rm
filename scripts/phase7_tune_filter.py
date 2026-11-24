"""Phase 7: tune the drift, threshold softness and bias resolution of the grid filter on calibration data (GPU or CPU).

Uses results/phase7/data/round1.npz (blind and monitor controllers) with freshly sampled detections at lambda = 0.5,
so neither the test scenarios nor the episodes used to report belief quality are involved. Updates the "filter"
entry of results/phase7/calibration.json with the best drift / n_b / softness by log-likelihood of the counter.
"""

from __future__ import annotations

import itertools
import json

import jax
import jax.numpy as jnp
import numpy as np
from phase7_common import OUT, detector, load_calibration, load_filter
from phase7_models import quality

from lvadrm import estimators as es


class Recorded(es.LambdaDetector):
    def detect(self, key, b, avo, t_open_s):
        return avo


def beliefs(x, n, det, fm, **kw):
    rec = Recorded(tau=det.tau, sigma=det.sigma, lam=det.lam, p_pos=det.p_pos, p_neg=det.p_neg)
    est = es.filter_belief(rec, n, fm, **kw)

    def one(xe):
        def body(bs, xt):
            bs = est.update(jax.random.PRNGKey(0), bs, xt[1:], {"labels": jnp.stack([xt[0] > 0.5]), "t_open": jnp.zeros(())})
            return bs, est.vector(bs)[:n]

        return jax.lax.scan(body, est.init(jax.random.PRNGKey(0)), xe)[1]

    f = jax.jit(jax.vmap(one))
    return np.concatenate([np.asarray(f(jnp.asarray(x[i : i + 128]))) for i in range(0, len(x), 128)])


def soften(fm, width):
    """Convolve P(open | x) with a Gaussian of the given width (log speed): model error around the threshold."""
    if width <= 0:
        return fm
    x = np.asarray(fm.x)
    k = np.exp(-0.5 * ((x[:, None] - x[None]) / width) ** 2)
    return fm._replace(p_open=np.clip((k * np.asarray(fm.p_open)[None]).sum(1) / k.sum(1), 1e-3, 1 - 1e-3))


def tables(z, alpha, km):
    """Response tables with fine bins around the threshold and an m grid of ``km`` cells (all from round 1)."""
    from phase7_calibrate import design

    cal = load_calibration()
    par, om = z["params"], z["omega"]
    m = np.exp(design(par[..., 0], par[..., 1], par[..., 2], par[..., 3]) @ np.asarray(cal["w_avo_coef"]))
    w = om.copy()
    for t in range(1, len(w)):
        w[t] = alpha * om[t] + (1 - alpha) * w[t - 1]
    x = np.log(m / w)[20:].ravel()
    y, ts = z["avo"][20:].ravel(), 1000 * z["t_open"][20:].ravel()
    edges = np.r_[np.arange(-0.25, -0.04, 0.02), np.arange(-0.04, 0.0401, 0.004), np.arange(0.06, 0.26, 0.02)]
    mid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.digitize(x, edges) - 1
    po, tm, sd = [], [], []
    for i in range(len(mid)):
        b = idx == i
        o = b & y
        po.append((y[b].sum() + 0.5) / (b.sum() + 1.0))
        tm.append(ts[o].mean() if o.sum() > 20 else np.nan)
        sd.append(ts[o].std() if o.sum() > 20 else np.nan)
    po, tm = np.maximum.accumulate(np.clip(po, 0.002, 0.998)), np.array(tm)
    ok = np.isfinite(tm)
    log_m = np.linspace(np.log(5800), np.log(10000), km)
    h = np.histogram(np.log(m[0]), bins=np.r_[log_m - 0.5 * (log_m[1] - log_m[0]), log_m[-1] + 0.5 * (log_m[1] - log_m[0])])[0]
    h = np.convolve(h, np.ones(3) / 3, "same") + 1.0
    return es.FilterModel(x=mid, p_open=po, t_ms=np.interp(mid, mid[ok], tm[ok]), alpha=alpha, sigma_t=float(np.nanmedian(sd)), drift=0.0,
                          log_m=log_m, prior_m=h / h.sum())


def main() -> None:
    z = np.load(OUT / "data" / "round1.npz")
    sel = np.flatnonzero(np.isin(z["src"], ["blind", "monitor"]))
    sel = np.random.default_rng(0).permutation(sel)[:384]
    n = 10
    rng = np.random.default_rng(1)
    t_ms = 1000 * z["t_open"][:, sel]
    e1, e2 = rng.normal(size=len(sel)), rng.normal(size=t_ms.shape)
    u = z["u"][:, sel].T.astype(int)
    best = None
    for km, alpha, drift, nb in ((41, 0.7, 0.008, 21), (61, 0.7, 0.02, 21), (41, 0.7, 0.02, 21), (61, 0.7, 0.05, 21)):
        fm = tables(z, alpha, km)._replace(drift=drift)
        tot = []
        for lam in (0.5, 0.9):
            det = detector(lam)
            d = t_ms + e1 * np.sqrt(lam) * det.sigma + e2 * np.sqrt(1 - lam) * det.sigma > det.tau
            x = np.concatenate([d[..., None].astype(np.float32), z["obs"][:, sel]], -1).swapaxes(0, 1)
            q = quality(beliefs(x, n, det, fm, n_b=nb)[:, 20:], u[:, 20:], n)
            tot.append(q["ll"])
            print(f"lam {lam} km {km} alpha {alpha} drift {drift:.4f} n_b {nb}: LL {q['ll']:.4f} risk {q['risk_pred']:.3f}/{q['risk_true']:.3f} ECE {q['ece']:.3f} worst5 {q['missed_worst5']:.2f}", flush=True)
        if best is None or np.mean(tot) > best[0]:
            best = (float(np.mean(tot)), drift, nb, 0.0, fm)
    cal = load_calibration()
    cal["filter"] = {k: (np.asarray(v).tolist() if np.ndim(v) else float(v)) for k, v in best[4]._asdict().items()}
    cal["filter_n_b"] = best[2]
    cal["filter_tuning"] = dict(ll=best[0], drift=best[1], n_b=best[2], soft=best[3])
    (OUT / "calibration.json").write_text(json.dumps(cal, indent=1))
    print("chosen", cal["filter_tuning"])


if __name__ == "__main__":
    main()
