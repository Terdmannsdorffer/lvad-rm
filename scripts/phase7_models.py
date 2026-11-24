"""Phase 7, stage 3: data and learned estimators (per-beat classifier, recurrent classifier, TDM) for each lambda (GPU).

    python scripts/phase7_models.py --lams 0 0.5 0.9 --n 10
    python scripts/phase7_models.py --lams 0.9 --sb 10 --skip-quality     models for a detector calibrated in each patient
    python scripts/phase7_models.py --lams 0 0.5 0.9 --window             only the window network, on the same episodes

Two rounds (as in DAgger): round 1 uses controllers driven by nothing (blind, random) or by estimators without
learned parts (naive, ibu_d, filter); round 2 adds controllers driven by the round-1 models. No data comes
from a controller that sees the true label, because its actions would give the label away. All models of a
(lambda, N) pair are trained on exactly the same episodes.
Writes results/phase7/models/lam<lam>[_sb<sb>]_N<n>[_s<seed>]_r{1,2}.pkl, *_win.pkl and beliefs_*.json (held-out quality).
"""

from __future__ import annotations

import argparse
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
from phase7_calibrate import random_fns
from phase7_common import OUT, detector, load_filter, load_models, model_path, pause_point, rollout, scenarios, start_speeds, window_path, write_atomic

from lvadrm.estimators import (
    WINDOW,
    BeatClassifier,
    BlindParams,
    MonitorParams,
    WindowTDM,
    blind_fns,
    calibrated_detector,
    make_estimator,
    monitor_fns,
    windows,
)
from lvadrm.inference import ibu_update, rm_state_sequence, tdm_sequence_logits, train_tdm
from lvadrm.presets import env_v3

HIDDEN = 64
KEEP = ("avo", "t_open", "obs", "u_next", "co", "back")


def collect(key, cfg, n, det, kinds, models, e_each, rng):
    """Episodes [E, T]: x (beat inputs), avo, u. Randomized controller parameters."""
    lam = det.lam
    parts = []
    for kind, e in zip(kinds, e_each):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        scn, w0 = scenarios(k1, cfg, e), start_speeds(k2, cfg, e)
        q = jnp.asarray(rng.uniform(3.8, 6.2, e), jnp.float32)
        vec = None
        if kind == "blind":
            fns, prm = blind_fns(cfg), BlindParams(q, jnp.asarray(rng.integers(4, 2 * n + 1, e)), jnp.asarray(rng.integers(1, 6, e)), jnp.zeros(e))
        elif kind == "random":
            fns, prm = random_fns(cfg), jnp.zeros(e)
        else:
            est = make_estimator(kind, det, n, models)
            fns = monitor_fns(est, cfg, n)
            prm = MonitorParams(q, jnp.asarray(rng.integers(1, n, e)), jnp.asarray(10 ** rng.uniform(-1.7, -0.05, e), jnp.float32), jnp.zeros(e))
            vec = lambda ps, est=est: est.vector(ps.bs)[-1:]  # noqa: E731
        o = rollout(fns, prm, scn, cfg, k3, w0, keep=KEEP, belief_vec=vec)
        if vec is None:
            kb, ke = jax.random.split(k4)
            b = np.asarray(jax.random.normal(kb, (e,))) * np.sqrt(lam) * det.sigma
            eps = np.asarray(jax.random.normal(ke, o["avo"].shape)) * np.sqrt(1 - lam) * det.sigma
            d = (1000 * o["t_open"] + b + eps > det.tau).astype(np.float32)
        else:
            d = o["belief"][..., 0].astype(np.float32)
        x = np.concatenate([d[..., None], o["obs"].astype(np.float32)], -1)
        parts.append((np.swapaxes(x, 0, 1), o["avo"].T, o["u_next"].T.astype(np.int32)))
        print(f"  {kind:9s} e={e} AVO {o['avo'].mean():.3f} CO {o['co'].mean():.2f} sens {d[o['avo']].mean():.3f} spec {1 - d[~o['avo']].mean():.3f}", flush=True)
    return [np.concatenate(z) for z in zip(*parts)]


def train_beat(x, y, xv, yv, steps=6000, seed=0):
    net = BeatClassifier()
    params = net.init(jax.random.PRNGKey(seed), jnp.asarray(x[:1, 0]))
    tx = optax.adam(optax.cosine_decay_schedule(2e-3, steps))
    opt = tx.init(params)
    X, Y = jnp.asarray(x.reshape(-1, x.shape[-1])), jnp.asarray(y.reshape(-1), jnp.float32)

    @jax.jit
    def step(params, opt, key):
        idx = jax.random.randint(key, (8192,), 0, X.shape[0])
        loss, g = jax.value_and_grad(lambda p: optax.sigmoid_binary_cross_entropy(net.apply(p, X[idx]), Y[idx]).mean())(params)
        up, opt = tx.update(g, opt)
        return optax.apply_updates(params, up), opt, loss

    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, k = jax.random.split(key)
        params, opt, loss = step(params, opt, k)
    val = optax.sigmoid_binary_cross_entropy(net.apply(params, jnp.asarray(xv)), jnp.asarray(yv, jnp.float32)).mean()
    print(f"    beat classifier: val BCE {float(val):.4f}", flush=True)
    return params


def train_window(x, u, xv, uv, n, steps=20000, batch=4096, seed=0):
    """Feedforward network on the last WINDOW beats, the same targets and episodes as TDM; the best of 10 checkpoints."""
    net = WindowTDM(n_states=n)
    X, Y = jnp.asarray(np.concatenate([x, np.ones(x.shape[:2] + (1,), x.dtype)], -1)), jnp.asarray(u)
    params = net.init(jax.random.PRNGKey(seed), jnp.zeros((1, WINDOW, X.shape[-1])))
    tx = optax.adam(optax.cosine_decay_schedule(1e-3, steps))
    opt = tx.init(params)
    back = jnp.arange(WINDOW)

    @jax.jit
    def step(params, opt, key):
        k1, k2 = jax.random.split(key)
        e, t = jax.random.randint(k1, (batch,), 0, X.shape[0]), jax.random.randint(k2, (batch,), 0, X.shape[1])
        pos = t[:, None] - back
        xb = jnp.where((pos >= 0)[..., None], X[e[:, None], jnp.maximum(pos, 0)], 0.0)
        loss, g = jax.value_and_grad(lambda p: optax.softmax_cross_entropy_with_integer_labels(net.apply(p, xb), Y[e, t]).mean())(params)
        up, opt = tx.update(g, opt)
        return optax.apply_updates(params, up), opt, loss

    val = jax.jit(lambda p, xb, yb: optax.softmax_cross_entropy_with_integer_labels(net.apply(p, xb), yb).mean())
    key, best = jax.random.PRNGKey(seed + 1), (np.inf, params)
    for i in range(steps):
        key, k = jax.random.split(key)
        params, opt, loss = step(params, opt, k)
        if (i + 1) % (steps // 10) == 0:
            m = len(xv) // 32 * 32
            nll = float(np.mean([val(params, jnp.asarray(windows(xv[j : j + 32])), jnp.asarray(uv[j : j + 32])) for j in range(0, m, 32)]))
            if nll < best[0]:
                best = (nll, params)
            print(f"    window {i + 1}: train NLL {float(loss):.4f}, val NLL {nll:.4f}", flush=True)
    return best[1]


def replay(kind, x, n, det, models):
    """Beliefs [E, T, n] of an estimator on recorded inputs (detections included)."""
    E, T, _ = x.shape
    d = x[..., 0] > 0.5
    if kind == "naive":
        return np.eye(n, dtype=np.float32)[rm_state_sequence(d, n, True)]
    if kind in ("ibu_d", "ibu", "ibu_hist"):
        if kind == "ibu_d":
            p = np.where(d, det.p_pos, det.p_neg).astype(np.float32)
        elif kind == "ibu":
            p = np.asarray(jax.nn.sigmoid(BeatClassifier().apply(models["beat"], jnp.asarray(x))))
        else:
            from lvadrm.inference import TDM

            p = np.asarray(jax.nn.softmax(tdm_sequence_logits(TDM(2, HIDDEN), models["hist"], jnp.swapaxes(jnp.asarray(x), 0, 1)))[..., 1]).T

        def body(b, pt):
            b = ibu_update(b, pt, n, True)
            return b, b

        b0 = jnp.zeros((E, n)).at[:, 0].set(1.0)
        return np.asarray(jnp.swapaxes(jax.lax.scan(body, b0, jnp.asarray(p.T))[1], 0, 1))
    if kind == "tdm":
        from lvadrm.inference import TDM

        return np.asarray(jnp.swapaxes(jax.nn.softmax(tdm_sequence_logits(TDM(n, HIDDEN), models["tdm"], jnp.swapaxes(jnp.asarray(x), 0, 1))), 0, 1))
    if kind.startswith("filter"):
        est = make_estimator("filter", det, n, models, **({"lam": 0.5} if kind == "filter_lam0.5" else {}))
        # feed the recorded detection: labels[0] = detection with a "perfect" detector inside the filter
        from lvadrm import estimators as es

        class Recorded(es.LambdaDetector):
            def detect(self, key, b, avo, t_open_s):
                return avo

        rec = Recorded(tau=det.tau, sigma=det.sigma, lam=det.lam, p_pos=det.p_pos, p_neg=det.p_neg)
        est = es.filter_belief(rec, n, models["filter"], lam=0.5 if kind == "filter_lam0.5" else None)

        def one(xe):
            bs = est.init(jax.random.PRNGKey(0))

            def body(bs, xt):
                bs = est.update(jax.random.PRNGKey(0), bs, xt[1:], {"labels": jnp.stack([xt[0] > 0.5]), "t_open": jnp.zeros(())})
                return bs, est.vector(bs)[:n]

            return jax.lax.scan(body, bs, xe)[1]

        return np.concatenate([np.asarray(jax.vmap(one)(jnp.asarray(x[i : i + 256]))) for i in range(0, E, 256)])
    raise ValueError(kind)


def quality(b, u, n):
    """Log-likelihood, mean predicted and true P(counter = n - 1), calibration error and missed danger per patient."""
    pb = np.take_along_axis(b, u[..., None], -1)[..., 0]
    r, y = b[..., n - 1], u == n - 1
    bins = np.minimum((r * 10).astype(int), 9)
    ece = sum(abs(r[bins == i].mean() - y[bins == i].mean()) * (bins == i).mean() for i in range(10) if (bins == i).any())
    miss, dang = ((r < 0.1) & y).sum(1), y.sum(1)
    has = dang > 0
    per = np.sort(miss[has] / dang[has])
    curve = [[float(r[bins == i].mean()), float(y[bins == i].mean()), int((bins == i).sum())] for i in range(10) if (bins == i).any()]
    return {"ll": float(np.log(np.maximum(pb, 0.01)).mean()), "risk_pred": float(r.mean()), "risk_true": float(y.mean()), "ece": float(ece),
            "missed": float(miss.sum() / max(dang.sum(), 1)), "missed_worst5": float(per[-max(1, len(per) // 20):].mean()), "reliability": curve}


EVAL_KINDS = ("naive", "ibu_d", "ibu", "ibu_hist", "tdm", "filter", "filter_lam0.5")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--eval-only", action="store_true", help="only the held-out belief quality, with saved models")
    ap.add_argument("--sb", type=float, default=None, help="train for a detector calibrated in each patient: residual SD of the bias [ms]")
    ap.add_argument("--seed", type=int, default=0, help="seed of the episodes and the networks")
    ap.add_argument("--window", action="store_true", help="only the window network; the episodes are collected again with the saved round-1 models")
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args()
    n = args.n
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    cfg = env_v3(n, "unknown")
    sz = lambda v: [int(v_ * args.scale) for v_ in v]  # noqa: E731
    for lam in args.lams:
        print(f"== lambda {lam} N {n}", flush=True)
        det = detector(lam) if args.sb is None else calibrated_detector(detector(lam), args.sb)
        rng = np.random.default_rng(int(1000 * lam) + n + 100000 * args.seed)
        key = jax.random.PRNGKey(int(1000 * lam) + 7 * n + 100000 * args.seed)
        k1, k2, k3 = jax.random.split(key, 3)
        models = {"filter": load_filter()}
        if args.eval_only:
            models = load_models(lam, n, 2, args.sb, args.seed)
        else:
            data = collect(k1, cfg, n, det, ("blind", "random", "naive", "ibu_d", "filter"), models, sz((1024, 384, 768, 768, 768)), rng)
            for rnd in (1, 2):
                if rnd == 2:
                    pause_point()
                    extra = collect(k2, cfg, n, det, ("ibu", "ibu_hist", "tdm", "filter"), models, sz((1024, 1024, 1024, 512)), rng)
                    data = [np.concatenate(z) for z in zip(data, extra)]
                x, avo, u = data
                perm = np.random.default_rng(rnd).permutation(len(x))
                nv = max(64, len(x) // 12)
                tr, va = perm[nv:], perm[:nv]
                if args.window:
                    if rnd == 1:
                        models = load_models(lam, n, 1, args.sb, args.seed)
                    else:
                        win = train_window(x[tr], u[tr], x[va], u[va], n, seed=args.seed)
                        write_atomic(window_path(lam, n, args.sb, args.seed), pickle.dumps({"win": jax.tree.map(np.asarray, win)}))
                    continue
                models["beat"] = train_beat(x[tr], avo[tr], x[va], avo[va], seed=args.seed)
                _, models["hist"], _ = train_tdm(x[tr], avo[tr].astype(np.int32), x[va], avo[va].astype(np.int32), 2, hidden=HIDDEN, epochs=20, seed=args.seed)
                _, models["tdm"], _ = train_tdm(x[tr], u[tr], x[va], u[va], n, hidden=HIDDEN, epochs=30, seed=args.seed)
                write_atomic(model_path(lam, n, rnd, args.sb, args.seed), pickle.dumps({k: jax.tree.map(np.asarray, models[k]) for k in ("beat", "hist", "tdm")}))
        if args.window or args.skip_quality:
            continue
        # held-out quality, on episodes of controllers driven by every estimator
        test = collect(k3, cfg, n, det, ("blind", "naive", "ibu", "tdm", "filter"), models, sz((256, 256, 256, 256, 256)), rng)
        x, avo, u = test
        res = {k: quality(replay(k, x, n, det, models)[:, 20:], u[:, 20:], n) for k in EVAL_KINDS}
        for k, v in res.items():
            print(f"  {k:14s} LL {v['ll']:.3f}  risk {v['risk_pred']:.3f} / {v['risk_true']:.3f}  ECE {v['ece']:.3f}  missed {v['missed']:.2f}  worst 5% {v['missed_worst5']:.2f}", flush=True)
        name = model_path(lam, n, 2, args.sb, args.seed).stem.removesuffix("_r2")
        write_atomic(OUT / "models" / f"beliefs_{name}.json", json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
