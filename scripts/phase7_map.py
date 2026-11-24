"""Phase 7, stage 1 (no RL): threshold map and brute-force reference pattern (GPU, float32).

Map: for random patients around a candidate base, the highest constant speed that opens the aortic valve
(omega_AVO), the cardiac output there, and the lowest speed with <= 1 ml of backflow per beat. Reference
pattern: among periodic speed patterns that respect the action set (hold, 5 % steps down, 5 % steps up), the
one with the highest mean cardiac output that never leaves the valve closed N beats in a row.

    python scripts/phase7_map.py [--base-emax 0.7]
Writes results/phase7/map.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.integrate import simulate_beat
from lvadrm.model import initial_state, total_volume
from lvadrm.params import make_params
from lvadrm.patients import HF_PATIENT as P

OUT = Path("results/phase7")
KEYS = ("emax", "hr", "rs", "vtot")
T, EVAL = 170, 90
N_LIST = (3, 5, 10, 20)
CO_TARGET = 5.0


def run(seq, emax, hr, rs, vtot):
    p = make_params(emax=emax, emin=P.emin, v0=P.v0, hr=hr, rs=rs)
    tc = 60.0 / hr

    def body(y, w):
        y, s, _ = simulate_beat(y, p, w, 0.0, (vtot - total_volume(y, p)) / tc, 0.0, n_seg=64, n_inner=105, unroll=8)
        return y, (s.v_aortic, s.co, s.v_back, s.t_open)

    return jax.lax.scan(body, initial_state(p, vtot).astype(jnp.float32), seq)[1]


vrun = jax.jit(jax.vmap(run))


def simulate(seqs, cond, chunk=8192):
    """seqs [M, T]; cond dict of [M] arrays. Returns arrays [M, T]: V_ao, CO, V_back, T_open."""
    out = []
    for i in range(0, len(seqs), chunk):
        s = np.asarray(seqs[i : i + chunk], np.float32)
        c = [np.asarray(cond[k][i : i + chunk], np.float32) for k in KEYS]
        m = len(s)
        if m < chunk and len(seqs) > chunk:
            s = np.concatenate([s, np.repeat(s[:1], chunk - m, 0)])
            c = [np.concatenate([x, np.repeat(x[:1], chunk - m)]) for x in c]
        out.append([np.asarray(x)[:m] for x in vrun(jnp.asarray(s), *map(jnp.asarray, c))])
    return [np.concatenate(z) for z in zip(*out)]


def pattern(h, d, low):
    """Relative speeds of one period: h beats high, d steps of -5 %, low beats at the bottom, +5 % steps back."""
    m = [1.0] * h + [0.95**i for i in range(1, d + 1)] + [0.95**d] * (low - 1)
    while m[-1] * 1.05 < 0.9999 and 1.0 / m[-1] > 1.0101:
        m.append(m[-1] * 1.05)
    return np.array(m)


SHAPES = [(h, d, low) for h in (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18) for d in (1, 2, 3, 4, 5, 6, 8, 10) for low in (1, 2)]


def longest_closed(avo):
    r = np.zeros(avo.shape[0], int)
    best = r.copy()
    for t in range(avo.shape[1]):
        r = np.where(avo[:, t], 0, r + 1)
        best = np.maximum(best, r)
    return best


def threshold_map(cond, speeds=np.arange(5500, 12001, 50.0), beats=40):
    """Steady state at constant speed for every patient and speed."""
    n, g = len(cond["emax"]), len(speeds)
    c = {k: np.repeat(v, g) for k, v in cond.items()}
    seqs = np.repeat(np.tile(speeds, n)[:, None], beats, 1)
    va, co, vb, _ = (x[:, -5:].mean(1).reshape(n, g) for x in simulate(seqs, c))
    op = va > 0.5
    w_avo = np.array([speeds[o].max() if o.any() else np.nan for o in op])
    co_avo = np.array([co[i][speeds == w_avo[i]][0] if np.isfinite(w_avo[i]) else np.nan for i in range(n)])
    w_back = np.array([speeds[v <= 1.0].min() if (v <= 1.0).any() else np.nan for v in vb])
    return dict(w_avo=w_avo, co_avo=co_avo, w_back=w_back, speeds=speeds, co=co, open=op, v_back=vb)


def reference(cond1, w_avo, co_avo):
    """Best periodic pattern per N for one patient."""
    w_hi = np.linspace(0.92, 1.6, 40) * w_avo
    seqs, meta = [], []
    for s in SHAPES:
        base = np.resize(pattern(*s), T)
        for w in w_hi:
            seqs.append(w * base)
            meta.append((s, w, len(pattern(*s))))
    seqs = np.array(seqs)
    va, co, vb, _ = simulate(seqs, {k: np.full(len(seqs), v) for k, v in cond1.items()})
    per = np.array([m[2] for m in meta])
    mask = np.zeros(seqs.shape, bool)
    for length in np.unique(per):
        mask[per == length, T - (EVAL // length) * length :] = True

    def mean(x):
        return (x * mask).sum(1) / mask.sum(1)

    comean, back, opn = mean(co), mean(vb > 1.0), mean(va > 0.5)
    dev = mean(np.abs(co - CO_TARGET) / CO_TARGET)
    runlen = longest_closed(va[:, T - EVAL :] > 0.5)
    out = {}
    for n in N_LIST:
        ok = (runlen <= n - 1) & (back <= 0.05)
        if not ok.any():
            out[n] = None
            continue
        j, jd = int(np.argmax(np.where(ok, comean, -1))), int(np.argmin(np.where(ok, dev, 9)))
        out[n] = dict(shape=meta[j][0], w_hi=float(meta[j][1]), w_lo=float(seqs[j].min()), co=float(comean[j]),
                      gain_pct=float(100 * (comean[j] / co_avo - 1)), open=float(opn[j]), seq=seqs[j][-40:].tolist(),
                      dev_best=float(dev[jd]), dev_shape=meta[jd][0], dev_w_hi=float(meta[jd][1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-emax", type=float, default=0.7)
    ap.add_argument("--n-random", type=int, default=400)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    e = args.base_emax
    named = {"base": (e, 80, 1, 1), "weaker": (0.8 * e, 80, 1, 1), "stronger": (1.2 * e, 80, 1, 1), "Rs +25 %": (e, 80, 1.25, 1),
             "Rs -20 %": (e, 80, 0.8, 1), "HR 100": (e, 100, 1, 1), "volume -10 %": (e, 80, 1, 0.9), "HF (Emax 1.0)": (1.0, 80, 1, 1)}
    cond = {k: np.array([v[i] * s for v in named.values()], float)
            for i, (k, s) in enumerate(zip(KEYS, (P.emax, 1.0, P.rs, P.vtot)))}
    m = threshold_map(cond)
    res = {"base_emax": e, "named": {}}
    for i, name in enumerate(named):
        c1 = {k: cond[k][i] for k in KEYS}
        ref = reference(c1, m["w_avo"][i], m["co_avo"][i])
        dconst = np.abs(m["co"][i] - CO_TARGET) / CO_TARGET + 9 * (~m["open"][i])
        res["named"][name] = dict(cond=c1, w_avo=m["w_avo"][i], co_avo=m["co_avo"][i], w_back=m["w_back"][i],
                                  best_open_const_dev=float(dconst.min()), speeds=m["speeds"].tolist(), co=m["co"][i].tolist(),
                                  open=m["open"][i].tolist(), reference=ref)
        print(f"{name:14s} wAVO {m['w_avo'][i]:.0f} CO {m['co_avo'][i]:.2f} wBack {m['w_back'][i]:.0f} dev const {dconst.min():.3f} | " + "  ".join(
            f"N{n}: {r['gain_pct']:+.1f}% CO {r['co']:.2f} dev {r['dev_best']:.3f} {r['dev_shape']}" if r else f"N{n}: none" for n, r in ref.items()), flush=True)
    rng = np.random.default_rng(0)
    n = args.n_random
    rc = dict(emax=rng.uniform(0.75, 1.25, n) * e * P.emax, hr=rng.uniform(65, 100, n),
              rs=rng.uniform(0.8, 1.3, n) * P.rs, vtot=rng.uniform(0.88, 1.05, n) * P.vtot)
    mr = threshold_map(rc, beats=30)
    res["random"] = {k: mr[k].tolist() for k in ("w_avo", "co_avo", "w_back")} | {"cond": {k: v.tolist() for k, v in rc.items()}}

    def q(x):
        return np.nanpercentile(x, [0, 5, 25, 50, 75, 95, 100]).round(2).tolist()

    print("random: wAVO", q(mr["w_avo"]), "CO@wAVO", q(mr["co_avo"]), "wBack-wAVO", q(mr["w_back"] - mr["w_avo"]),
          "frac CO@wAVO<5:", float(np.mean(mr["co_avo"] < 5)), "frac no window:", float(np.mean(mr["w_back"] > mr["w_avo"])))
    (OUT / "map.json").write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
