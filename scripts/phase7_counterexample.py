"""Phase 7: adversarial search for counterexamples to the sign of the IBU bias (CPU).

Labels i.i.d. Bernoulli(q_theta) given a trait theta, binary detections whose sensitivity and
specificity depend on a latent bias B independent of theta, exact recurrent classifier with a warm-up history of n
detections (exchangeable: the count is sufficient). Hill climbing over the parameters to make beta_k = E prod m_j - E q^k POSITIVE.
Also the most negative conditional covariance (is the a.s. condition of the corollary violated?) and the same search
with a dependent trait (sanity check: it must find violations).

    python scripts/phase7_counterexample.py SEED
The output of seeds 0-3 is results/phase7/counterexample.txt.
"""
import itertools
import sys

import numpy as np
from scipy.special import gammaln

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
rng = np.random.default_rng(seed)


def beta_k(q, W, u1, u0, n, k):
    """q [T], W [B, T] joint prior, u1 [B] = P(d=1|open), u0 [B] = P(d=1|closed). Returns (beta_k, min conditional covariance on histories with weight)."""
    s = np.arange(n + 1)
    lc = gammaln(n + 1) - gammaln(s + 1) - gammaln(n - s + 1)
    p1 = np.clip(q[None, :] * u0[:, None] + (1 - q[None, :]) * u1[:, None], 1e-12, 1 - 1e-12)  # P(d=1 | theta, B)
    lw = lc[:, None, None] + s[:, None, None] * np.log(p1) + (n - s)[:, None, None] * np.log1p(-p1)
    H = np.exp(lw) * W  # [s, B, T] joint with the warm-up count
    val, mincov = 0.0, 0.0
    for seq in itertools.product([0, 1], repeat=k):
        P, w = H, np.ones(n + 1)
        for j, d in enumerate(seq):
            l1 = (u0 if d else 1 - u0)[None, :, None]  # closed
            l0 = (u1 if d else 1 - u1)[None, :, None]
            cl, op = P * q * l1, P * (1 - q) * l0
            P2 = cl + op
            Z = P2.sum((1, 2))
            m = cl.sum((1, 2)) / np.maximum(Z, 1e-300)
            if j < k - 1:
                r = k - 1 - j
                cov = (cl * q**r).sum((1, 2)) / np.maximum(Z, 1e-300) - m * (P2 * q**r).sum((1, 2)) / np.maximum(Z, 1e-300)
                ok = Z > 1e-9
                if ok.any():
                    mincov = min(mincov, cov[ok].min())
            w = w * m
            P = P2
        val += (P.sum((1, 2)) * w).sum()
    truth = (W.sum(0) * q**k).sum()
    return val - truth, mincov, truth


def sample(dep):
    nb, nt = rng.integers(2, 4), rng.integers(2, 4)
    q = np.clip(rng.beta(0.7, 0.7, nt), 0.01, 0.99)
    if dep:
        W = rng.dirichlet(np.ones(nb * nt) * 0.5).reshape(nb, nt)
    else:
        W = rng.dirichlet(np.ones(nb) * 0.5)[:, None] * rng.dirichlet(np.ones(nt) * 0.5)[None]
    return q, W, rng.beta(0.5, 0.5, nb), rng.beta(0.5, 0.5, nb)


def perturb(x, dep, s):
    q, W, u1, u0 = x
    f = lambda p: 1 / (1 + np.exp(-(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) + s * rng.normal(size=np.shape(p)))))  # noqa: E731
    if dep:
        W2 = np.clip(W, 1e-9, None) * np.exp(s * rng.normal(size=W.shape))
        W2 /= W2.sum()
    else:
        a, b = np.clip(W.sum(1), 1e-9, None) * np.exp(s * rng.normal(size=W.shape[0])), np.clip(W.sum(0), 1e-9, None) * np.exp(s * rng.normal(size=W.shape[1]))
        W2 = (a / a.sum())[:, None] * (b / b.sum())[None]
    return f(q), W2, f(u1), f(u0)


for dep in (False, True):
    for k in (2, 3, 4):
        best, bx, mc = -1.0, None, 0.0
        for n in (0, 5, 30, 120):
            for _ in range(8):
                x = max((sample(dep) for _ in range(12)), key=lambda z: beta_k(*z, n, k)[0])
                v, c, _ = beta_k(*x, n, k)
                s = 1.0
                for _ in range(220):
                    y = perturb(x, dep, s)
                    vy, cy, ty = beta_k(*y, n, k)
                    mc = min(mc, cy)
                    # relative objective keeps the search away from the trivial optimum (truth = 0)
                    if vy / max(ty, 1e-6) > v / max(beta_k(*x, n, k)[2], 1e-6):
                        x, v = y, vy
                    else:
                        s = max(s * 0.985, 0.02)
                if v > best:
                    best, bx = v, (n, x)
        print(f"dependent={dep} k={k}: largest beta_k found {best:+.3e}; most negative conditional covariance {mc:+.3e}", flush=True)
        if best > 1e-10:
            n, (q, W, u1, u0) = bx
            print("   n", n, "q", q.round(3), "u1", u1.round(3), "u0", u0.round(3), "W", W.round(4).tolist(), "truth", round(beta_k(q, W, u1, u0, n, k)[2], 4))
