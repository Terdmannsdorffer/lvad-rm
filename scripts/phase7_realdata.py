"""Phase 7: the bias of independent belief updates on real beat labels (CPU).

Annotations of five PhysioNet databases (no waveforms; about 20 MB). Two tasks:
    af   C = beat inside atrial fibrillation (ltafdb; afdb as external test, with its automatic beat detections)
    pvc  C = ventricular beat (mitdb and svdb, split by record; incartdb only for training: several records per patient)
The "detector" is a classifier of RR intervals. Beliefs about u = number of consecutive beats with C = 1 (saturated at K):
naive (counts the thresholded beats), IBU with a classifier of the last RR interval and its ratio to the previous one,
IBU with a classifier of the last 16 beats, IBU with a recurrent classifier, a filter that assumes a two-state Markov
chain of labels, and TDM. Records are split at random several times; all the numbers are on test records.

    python scripts/phase7_realdata.py --download
    python scripts/phase7_realdata.py --task af --seeds 0 1 2 3 4
    python scripts/phase7_realdata.py --task pvc --seeds 0 1 2 3 4
Writes results/phase7/realdata_<task>.json. The predictions were written before the first run (docs/phase7_decisions.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

OUT = Path("results/phase7")
DATA = OUT / "data" / "physionet"
FS = {"mitdb": 360, "svdb": 128, "incartdb": 257, "afdb": 250, "ltafdb": 128}
BEATS = {1: "N", 2: "L", 3: "R", 4: "a", 5: "V", 6: "F", 7: "J", 8: "A", 9: "S", 10: "E", 11: "j", 12: "/", 13: "Q", 25: "B", 30: "?", 34: "e", 35: "n", 38: "f", 41: "r"}
RHYTHM = 28
SEG, FAR = 400, 100  # segment length; distance between beats for the persistence of the errors
K = {"af": 30, "pvc": 5}
THR = {"af": 0.5, "pvc": 0.3}
P_NOM = 0.05


# --------------------------------------------------------------------------------------------------- data
def download() -> None:
    def get(args):
        url, path = args
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            if path.exists():
                return
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
            except OSError:
                time.sleep(2.0 * (attempt + 1))
                continue
            path.write_bytes(data)
        if not path.exists():
            raise RuntimeError(f"could not download {url}")

    jobs = []
    for db in FS:
        base = f"https://physionet.org/files/{db}/1.0.0"
        get((f"{base}/RECORDS", DATA / db / "RECORDS"))
        for r in (DATA / db / "RECORDS").read_text().split():
            jobs += [(f"{base}/{r}.{ext}", DATA / db / f"{r}.{ext}") for ext in (("atr", "qrs") if db == "afdb" else ("atr",))]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(get, jobs))
    print(len(jobs), "files in", DATA)


def read_annotations(path):
    """Minimal reader of MIT-format annotation files: (sample, code, aux string) of every annotation."""
    raw = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    w = raw[: len(raw) // 2 * 2].reshape(-1, 2).astype(int)
    w = w[:, 0] + (w[:, 1] << 8)
    t, i, out = 0, 0, []
    while i < len(w):
        code, dt = w[i] >> 10, w[i] & 1023
        i += 1
        if code == 0 and dt == 0:
            break
        if code == 59:  # SKIP: 4-byte interval, high word first
            v = (w[i] << 16) + w[i + 1]
            t += v - (1 << 32) if v >= 1 << 31 else v
            i += 2
        elif code == 63:  # AUX
            if out:
                out[-1][2] = raw[2 * i : 2 * i + dt].tobytes().split(b"\0")[0].decode("latin1")
            i += (dt + 1) // 2
        elif code not in (60, 61, 62):  # NUM, SUB, CHN carry no time
            t += dt
            out.append([t, code, ""])
    return np.array([o[0] for o in out]), np.array([o[1] for o in out]), [o[2] for o in out]


def load(db, task):
    recs = []
    for f in sorted((DATA / db).glob("*.atr")):
        s, c, aux = read_annotations(f)
        if db == "afdb":
            beats = read_annotations(f.with_suffix(".qrs"))[0]
        else:
            isb = np.isin(c, list(BEATS))
            beats, sym = s[isb], np.array([BEATS[x] for x in c[isb]])
        if task == "af":
            rs = s[c == RHYTHM]
            ra = [x for x, y in zip(aux, c) if y == RHYTHM]
            k = np.searchsorted(rs, beats, side="right") - 1
            lab = np.array([int(i >= 0 and ra[i].startswith("(AFIB")) for i in k])
        else:
            lab = (sym == "V").astype(int)
        rr = np.diff(beats) / FS[db]
        x = np.stack([np.log(np.clip(rr[1:], 0.2, 3.0) / 0.8), np.log(np.clip(rr[1:] / np.clip(rr[:-1], 0.05, None), 0.2, 5.0))], -1) * 2
        recs.append((f.stem, x.astype(np.float32), lab[2:].astype(np.int32)))
    return recs


def segments(recs, n_per, rng):
    """Random segments (n_per per record) or, with n_per = 0, all the consecutive ones; with the index of the record."""
    X, Y, R = [], [], []
    for i, (_, x, y) in enumerate(recs):
        if len(y) <= SEG:
            continue
        for a in rng.integers(0, len(y) - SEG, n_per) if n_per else np.arange(0, len(y) - SEG + 1, SEG):
            X.append(x[a : a + SEG])
            Y.append(y[a : a + SEG])
            R.append(i)
    return np.stack(X), np.stack(Y), np.array(R)


def stack(X, w=16):
    return np.concatenate([np.pad(X, ((0, 0), (j, 0), (0, 0)), mode="edge")[:, : X.shape[1]] for j in range(w)], -1)


def counter(y, k):
    u, c = np.zeros_like(y), np.zeros(len(y), dtype=y.dtype)
    for t in range(y.shape[1]):
        c = np.where(y[:, t] == 1, np.minimum(c + 1, k), 0)
        u[:, t] = c
    return u


# ------------------------------------------------------------------------------------------------- models
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = nn.tanh(nn.Dense(64)(x))
        return nn.Dense(1)(x)[..., 0]


class RNN(nn.Module):
    out: int

    @nn.compact
    def __call__(self, x):
        h = nn.RNN(nn.GRUCell(64))(nn.tanh(nn.Dense(64)(x)))
        return nn.Dense(self.out)(nn.tanh(nn.Dense(64)(h)))


def bce(lg, y):
    return optax.sigmoid_binary_cross_entropy(lg, y).mean()


def fit(net, loss, X, Y, Xv, Yv, seed, steps=3000, bs=64, lr=2e-3):
    p = net.init(jax.random.PRNGKey(seed), X[:2])
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(optax.cosine_decay_schedule(lr, steps)))
    st = tx.init(p)

    @jax.jit
    def step(p, st, x, y):
        g = jax.grad(lambda q: loss(net.apply(q, x), y))(p)
        u, st = tx.update(g, st)
        return optax.apply_updates(p, u), st

    ev = jax.jit(lambda p, x, y: loss(net.apply(p, x), y))
    rng, best, bp = np.random.default_rng(seed), np.inf, p
    for i in range(steps):
        idx = rng.integers(0, len(X), bs)
        p, st = step(p, st, X[idx], Y[idx])
        if i % 100 == 99:
            v = float(np.mean([ev(p, Xv[j : j + 128], Yv[j : j + 128]) for j in range(0, len(Xv) - 127, 128)]))
            if v < best:
                best, bp = v, p
    return jax.jit(partial(net.apply, bp)), best


def batched(f, X, n=256):
    return np.concatenate([np.asarray(f(X[i : i + n])) for i in range(0, len(X), n)])


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


# ------------------------------------------------------------------------------------------------ beliefs
def tail_ibu(m, k):
    """P(u >= j), j = 1..k, by IBU: the product of the last j probabilities."""
    out = np.zeros((*m.shape, k), np.float32)
    prev = np.zeros((m.shape[0], k + 1))
    prev[:, 0] = 1.0
    for t in range(m.shape[1]):
        cur = np.ones_like(prev)
        cur[:, 1:] = m[:, t][:, None] * prev[:, :-1]
        out[:, t], prev = cur[:, 1:], cur
    return out


def tail_hmm(m, y_tr, k):
    """Filter on the counter for a two-state Markov chain of labels, with the classifier as likelihood ratio."""
    pi = y_tr.mean()
    a11 = (y_tr[:, 1:] * y_tr[:, :-1]).sum() / y_tr[:, :-1].sum()
    a01 = (y_tr[:, 1:] * (1 - y_tr[:, :-1])).sum() / (1 - y_tr[:, :-1]).sum()
    l1, l0 = m / pi, (1 - m) / (1 - pi)
    b = np.zeros((m.shape[0], k + 1))
    b[:, 0], b[:, k] = 1 - pi, pi
    out = np.zeros((*m.shape, k), np.float32)
    for t in range(m.shape[1]):
        n = np.zeros_like(b)
        n[:, 0] = l0[:, t] * (b[:, 0] * (1 - a01) + b[:, 1:].sum(1) * (1 - a11))
        n[:, 1] = l1[:, t] * b[:, 0] * a01
        n[:, 2:] += l1[:, t][:, None] * b[:, 1:k] * a11
        n[:, k] += l1[:, t] * b[:, k] * a11
        b = n / n.sum(1, keepdims=True)
        out[:, t] = b[:, ::-1].cumsum(1)[:, ::-1][:, 1:]
    return out


def persistence(err, p):
    """Phi and latent (tetrachoric) correlation of the errors of two beats of one segment at least FAR beats apart."""
    from scipy.optimize import brentq
    from scipy.stats import multivariate_normal as mvn
    from scipy.stats import norm

    h = (SEG - FAR) // 2
    a, b = err[:, :h], err[:, h + FAR :]
    n1, n2, s1, s2 = np.isfinite(a).sum(1), np.isfinite(b).sum(1), np.nansum(a, 1), np.nansum(b, 1)
    cov = ((s1 - n1 * p) * (s2 - n2 * p)).sum() / max((n1 * n2).sum(), 1)
    target, z = cov + p * p, norm.ppf(p)

    def f(r):
        return mvn.cdf([z, z], mean=[0, 0], cov=[[1, r], [r, 1]]) - target

    lat = float(brentq(f, 0.0, 0.999)) if f(0.0) < 0 < f(0.999) else 0.0
    return float(cov / (p * (1 - p))), lat


# --------------------------------------------------------------------------------------------------- main
def evaluate(recs, task, fs, y_tr, seed):
    k = K[task]
    fm, fw, fh, ft = fs
    X, Y, R = segments(recs, 0 if task == "pvc" else 100, np.random.default_rng(1000 + seed))
    u = counter(Y, k)[:, k:]
    m = {"beat": sig(batched(fm, X)), "w16": sig(batched(lambda x: fw(stack(x)), X)), "hist": sig(batched(fh, X))[..., 0]}
    pt = np.asarray(jax.nn.softmax(batched(ft, X), -1))
    bel = {"naive": tail_ibu((m["beat"] > THR[task]).astype(float), k), "ibu": tail_ibu(m["beat"], k), "ibu_w16": tail_ibu(m["w16"], k), "ibu_hist": tail_ibu(m["hist"], k),
           "hmm": tail_hmm(m["beat"], y_tr, k), "tdm": pt[..., ::-1].cumsum(-1)[..., ::-1][..., 1:]}
    truth = np.array([(u >= j).mean() for j in range(1, k + 1)])
    last = u >= k
    out = {"truth": truth.tolist(), "records": len(recs), "beats": int(u.size), "est": {}, "persistence": {}}
    for e, b in bel.items():
        b = b[:, k:]
        risk, quiet = b[..., -1], b[..., -1] <= P_NOM
        d = b - (u[..., None] >= np.arange(1, k + 1))
        per = np.stack([d[R == r].mean((0, 1)) for r in np.unique(R)])
        rng = np.random.default_rng(0)
        bs = np.stack([per[rng.integers(0, len(per), len(per))].mean(0) for _ in range(500)])
        bins = np.clip((risk * 10).astype(int), 0, 9)
        out["est"][e] = {"ratio": (b.mean((0, 1)) / truth).tolist(), "beta": per.mean(0).tolist(), "beta_lo": np.percentile(bs, 2.5, 0).tolist(),
                         "beta_hi": np.percentile(bs, 97.5, 0).tolist(), "missed": float((last & quiet).sum() / last.sum()), "realized": float(last[quiet].mean()),
                         "ece": float(sum(abs(risk[bins == i].mean() - last[bins == i].mean()) * (bins == i).mean() for i in range(10) if (bins == i).any()))}
    for name in ("beat", "w16"):
        d = m[name] > THR[task]
        for c in (0, 1):
            err = np.where(Y == c, (d != c).astype(float), np.nan)
            p = float(np.nanmean(err))
            phi, lat = persistence(err, p)
            out["persistence"][f"{name}_{'false' if c == 0 else 'missed'}"] = {"rate": p, "phi": phi, "latent": lat}
    return out


def run(task, seed):
    k, rng = K[task], np.random.default_rng(seed)
    if task == "af":
        r = load("ltafdb", "af")
        o = rng.permutation(len(r))
        tr, va, te, ext, n_tr = [r[i] for i in o[:50]], [r[i] for i in o[50:60]], [r[i] for i in o[60:]], load("afdb", "af"), 100
    else:
        r = load("mitdb", "pvc") + load("svdb", "pvc")
        o = [i for i in rng.permutation(len(r)) if r[i][0] != "202"]  # mitdb 201 and 202 are the same patient
        te, va, tr, ext, n_tr = [r[i] for i in o[:50]], [r[i] for i in o[50:65]], [r[i] for i in o[65:]] + load("incartdb", "pvc"), None, 30
    g = np.random.default_rng(100 + seed)
    (X, Y, _), (Xv, Yv, _) = segments(tr, n_tr, g), segments(va, 60, g)
    U, Uv = counter(Y, k), counter(Yv, k)

    def ce(lg, u):
        return optax.softmax_cross_entropy_with_integer_labels(lg[:, k:], u[:, k:]).mean()

    fm, l1 = fit(MLP(), bce, X, Y, Xv, Yv, seed)
    fw, l2 = fit(MLP(), bce, stack(X), Y, stack(Xv), Yv, seed)
    fh, l3 = fit(RNN(1), lambda lg, y: bce(lg[..., 0], y), X, Y, Xv, Yv, seed)
    ft, l4 = fit(RNN(k + 1), ce, X, U, Xv, Uv, seed)
    m = sig(batched(fm, X))

    def cov(a, b):
        return float((a * b).mean() - a.mean() * b.mean())

    res = {"val_loss": [l1, l2, l3, l4], "lag1": {"cov_m": cov(m[:, :-1], m[:, 1:]), "cov_C": cov(Y[:, :-1], Y[:, 1:])}, "test": evaluate(te, task, (fm, fw, fh, ft), Y, seed)}
    if ext:
        res["external"] = evaluate(ext, task, (fm, fw, fh, ft), Y, seed)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--task", choices=["af", "pvc"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = ap.parse_args()
    if a.download:
        download()
    if not a.task:
        return
    path = OUT / f"realdata_{a.task}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for seed in a.seeds:
        if str(seed) in res:
            continue
        res[str(seed)] = r = run(a.task, seed)
        path.write_text(json.dumps(res))
        for part in ("test", "external"):
            if part in r:
                print(a.task, seed, part, {e: [round(x["ratio"][j - 1], 3) for j in (1, 3, 5, 30) if j <= K[a.task]] + [round(x["missed"], 3)] for e, x in r[part]["est"].items()},
                      {p: round(x["latent"], 2) for p, x in r[part]["persistence"].items()}, flush=True)


if __name__ == "__main__":
    main()
