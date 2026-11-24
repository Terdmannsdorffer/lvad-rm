"""Phase 7: bias of the beliefs as a function of the length of the run, in open and in closed loop (GPU).

For k = 1..N-1, the true probability of {counter >= k} and the mean belief of every estimator, on 512 episodes of
(a) the blind pattern and (b) the monitor driven by TDM (both with the configuration chosen on validation, flow-tracking
nominal speed). Theorem 1 says that the sign of the IBU bias is that of the residual association of the labels, which
feedback could reverse; this measures it.

    python scripts/phase7_bias.py
Writes results/phase7/bias.json.
"""

from __future__ import annotations

import json

import jax
import numpy as np
from phase7_common import OUT, detector, load_models, rollout, scenarios, start_speeds
from phase7_models import replay
from phase7_monitor import BEATS, bli, mon
from phase7_tables import episodes

from lvadrm.estimators import blind_fns, make_estimator, monitor_fns
from lvadrm.presets import env_v3

N, E, SEED, SKIP = 10, 512, 4242, 30
KINDS = ("naive", "ibu_d", "ibu", "ibu_hist", "tdm", "filter")


def main() -> None:
    cfg = env_v3(N, "unknown", episode_beats=BEATS)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(SEED), 3)
    scn, w0 = scenarios(k1, cfg, E), start_speeds(k2, cfg, E)
    ref = json.loads((OUT / "monitor/reference_N10.json").read_text())["unknown"]
    res = {}
    for lam in (0.0, 0.5, 0.9):
        det, models = detector(lam), load_models(lam, N)
        est_rows = json.loads((OUT / f"monitor/estimators_N10_lam{lam:g}.json").read_text())
        for policy in ("blind", "tdm", "ibu_d"):
            if policy == "blind":
                c = np.array(episodes(ref["blind"])[3], float)
                # the blind configuration chosen for the 1 % budget hardly ever closes the valve; use a longer period
                c = np.array([5.0, 12.0, 4.0])
                o = rollout(blind_fns(cfg), bli(np.repeat(c[None], E, 0)), scn, cfg, k3, w0, keep=("avo", "t_open", "obs", "u_next"))
                kb, ke = jax.random.split(jax.random.PRNGKey(SEED + 1))
                b = np.asarray(jax.random.normal(kb, (E,))) * np.sqrt(lam) * det.sigma
                eps = np.asarray(jax.random.normal(ke, o["avo"].shape)) * np.sqrt(1 - lam) * det.sigma
                d = (1000 * o["t_open"] + b + eps > det.tau).astype(np.float32)
            else:
                est = make_estimator(policy, det, N, models)
                c = np.array(episodes(est_rows[policy])[3], float)
                o = rollout(monitor_fns(est, cfg, N), mon(np.repeat(c[None], E, 0)), scn, cfg, k3, w0, keep=("avo", "t_open", "obs", "u_next"),
                            belief_vec=lambda ps, est=est: est.vector(ps.bs)[-1:])
                d = o["belief"][..., 0].astype(np.float32)
            x = np.concatenate([d[..., None], o["obs"].astype(np.float32)], -1).swapaxes(0, 1)
            u = o["u_next"].T[:, SKIP:]
            row = {"config": c.tolist(), "truth": [float((u >= k).mean()) for k in range(1, N)], "persistence": float(
                ((~o["avo"][1:]) & (~o["avo"][:-1])).sum() / max((~o["avo"][:-1]).sum(), 1))}
            for kind in KINDS:
                bel = replay(kind, x, N, det, models)[:, SKIP:]
                row[kind] = [float(bel[..., k:].sum(-1).mean()) for k in range(1, N)]
            res[f"{policy}_lam{lam:g}"] = row
            print(policy, lam, "truth", np.round(row["truth"], 3), "P(closed|closed)", round(row["persistence"], 3), flush=True)
            for kind in KINDS:
                print("   ", kind, np.round(np.array(row[kind]) / np.maximum(row["truth"], 1e-9), 2), flush=True)
    (OUT / "bias.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
