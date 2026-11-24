"""Sign of the bias of independent belief updating (IBU) in counting tasks: exact numerical checks (CPU, NumPy).

Model: labels C_t (1 = closed) form a stationary symmetric two-state Markov chain with persistence s = P(C_{t+1} = C_t);
detections d_t = 1[T0 (1 - C_t) + b + eps_t > tau], b ~ N(0, lam) per episode, eps_t ~ N(0, 1 - lam), with tau and T0
set by the sensitivity and specificity. Everything below is computed exactly (Gauss-Hermite quadrature over b and
enumeration / forward recursions over labels and detections), not by simulation.

    truth(k)     P(C_1 = ... = C_k = 1)
    ibu(k)       E[prod_j P(C_j = 1 | d_j)]                 per-beat calibrated classifier
    ibuh(k)      E[prod_j P(C_j = 1 | d_1..d_j)]            exact recurrent classifier (IBU-hist)
The identity  truth - ibuh = sum_j E[ prod_{i<j} m_i  Cov(C_j, prod_{i>j} C_i | d_1..d_j) ]  is verified to machine precision.

Writes results/phase7/theory.json.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

OUT = Path("results/phase7")
SENS, SPEC = 0.86, 0.93
TAU, T0 = norm.ppf(SPEC), norm.ppf(SPEC) + norm.ppf(SENS)


def b_grid(lam, n=64):
    """Nodes and weights of b ~ N(0, lam)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x * np.sqrt(lam), w / w.sum()


def p_d1(b, lam):
    """P(d = 1 | C, b), columns C = 0 (open), 1 (closed)."""
    s = np.sqrt(max(1.0 - lam, 1e-12))
    return np.stack([norm.sf((TAU - T0 - b) / s), norm.sf((TAU - b) / s)], -1)


def truth(k, s):
    return 0.5 * s ** (k - 1)


def ibu(k, s, lam):
    """E[prod g(d_j)], g(d) = P(C = 1 | d): forward recursion over C for every b."""
    b, w = b_grid(lam)
    pd = p_d1(b, lam)  # [B, 2]
    marg = (w[:, None] * pd).sum(0)  # P(d = 1 | C)
    g1 = marg[1] / (marg[0] + marg[1])
    g0 = (1 - marg[1]) / (2 - marg[0] - marg[1])
    h = pd * g1 + (1 - pd) * g0  # E[g | C, b]
    P = np.array([[s, 1 - s], [1 - s, s]])
    f = 0.5 * h
    for _ in range(k - 1):
        f = (f @ P) * h
    return float((w * f.sum(1)).sum())


def history(k, s, lam, n_b=24):
    """Exact E[prod_j m_j] with m_j = P(C_j = 1 | d_1..d_j), and the right-hand side of the identity, by enumeration of d."""
    b, w = b_grid(lam, n_b)
    pd = p_d1(b, lam)
    P = np.array([[s, 1 - s], [1 - s, s]])
    e_ibuh, rhs = 0.0, 0.0
    # state: joint over (b, C) given the detections so far, times the probability of the detections
    stack = [(w[:, None] * 0.5 * np.ones((len(b), 2)), 1.0, 0)]
    while stack:
        pred, prod, j = stack.pop()  # pred: P(b, C_{j+1}, d_1..d_j)
        for d in (0, 1):
            post = pred * (pd if d else 1 - pd)
            pr = post.sum()
            if pr < 1e-300:
                continue
            m = post[:, 1].sum() / pr
            # future product: P(C_{j+2..k} all 1 | C_{j+1}); a Markov chain, so it does not depend on b
            r = k - j - 1
            if r > 0:
                v1, v0 = s**r, (1 - s) * s ** (r - 1)
                cov = m * v1 - m * (m * v1 + (1 - m) * v0)
                rhs += pr * prod * cov
            if j + 1 == k:
                e_ibuh += pr * prod * m
            else:
                stack.append((post @ P, prod * m, j + 1))
    return float(e_ibuh), float(rhs)


def k2_law(s, lam):
    """Cov(g1, g2) - Cov(C1, C2): exact sign of E[IBU] - truth for k = 2."""
    return ibu(2, s, lam) - truth(2, s)


def mixture(q, w, k):
    """Professor's case: labels i.i.d. given the trait theta with P(closed) = q_theta and no detector."""
    return float((w * q**k).sum()), float((w * q).sum() ** k)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    S, L = np.linspace(0.5, 0.98, 49), np.linspace(0.0, 0.98, 50)
    for k in (2, 5, 10):
        res[f"log_ratio_k{k}"] = [[float(np.log(ibu(k, s, l) / truth(k, s))) for s in S] for l in L]
    res["s"], res["lam"] = S.tolist(), L.tolist()
    # sign boundary
    res["boundary"] = {}
    from scipy.optimize import brentq

    for k in (2, 5, 10):
        bd = []
        for l in L[1:]:
            f = lambda s: np.log(ibu(k, s, l) / truth(k, s))  # noqa: B023, E731
            bd.append(float(brentq(f, 0.5, 0.995)) if f(0.5) * f(0.995) < 0 else None)
        res["boundary"][str(k)] = bd
    # identity for IBU-hist
    worst = 0.0
    rows = []
    for k, s, lam in itertools.product((2, 3, 5, 8), (0.5, 0.7, 0.9, 0.95), (0.0, 0.5, 0.9)):
        e, rhs = history(k, s, lam)
        worst = max(worst, abs(truth(k, s) - e - rhs))
        rows.append(dict(k=k, s=s, lam=lam, truth=truth(k, s), ibu=ibu(k, s, lam), ibuh=e, rhs=rhs))
    res["identity_max_error"], res["history"] = worst, rows
    print("max |truth - ibuh - rhs| =", worst)
    for r in rows:
        if r["k"] in (5, 8):
            print(f"k={r['k']} s={r['s']:.2f} lam={r['lam']:.1f}: truth {r['truth']:.4f}  IBU {r['ibu']:.4f}  IBU-hist {r['ibuh']:.4f}")
    # theorem checks: lam = 0 => ibu <= truth for s >= 1/2; s = 1/2 => ibu >= truth
    res["thm_label_persistence_min_gap"] = float(min(truth(k, s) - ibu(k, s, 0.0) for k in range(1, 21) for s in S))
    res["thm_error_persistence_min_gap"] = float(min(ibu(k, 0.5, l) - truth(k, 0.5) for k in range(1, 21) for l in L))
    print("theorem gaps (>= 0 up to rounding):", res["thm_label_persistence_min_gap"], res["thm_error_persistence_min_gap"])
    # worst patients: P(belief underestimates by a factor > 10 | closed for k beats) as lam grows
    (OUT / "theory.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
