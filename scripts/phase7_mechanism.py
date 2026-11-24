"""Phase 7: which patients are in violation? Violations against the detector bias of the patient, same controller.

Rolls out the configuration chosen on validation for every estimator and stores, per episode, the detector bias b [in
units of its standard deviation], the violations, and the rates of the detections. Episodes of their own seed (neither
validation nor test).

    python scripts/phase7_mechanism.py --nominal flow
    python scripts/phase7_mechanism.py --nominal ramp --lams 0.9
Writes results/phase7/mechanism[_ramp].json.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np
from phase7_common import OUT, SKIP, detector, load_models, outcome, rollout, scenarios, start_speeds
from phase7_monitor import BEATS, mon, threshold_speeds
from phase7_tables import episodes

from lvadrm.estimators import make_estimator, monitor_fns
from lvadrm.presets import env_v3

N, SEED = 10, 9001


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=512)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.9, 0.5])
    ap.add_argument("--kinds", nargs="+", default=["naive", "ibu_d", "ibu", "ibu_hist", "tdm", "filter"])
    ap.add_argument("--nominal", choices=["flow", "ramp"], default="flow")
    a = ap.parse_args()
    sfx = "_ramp" if a.nominal == "ramp" else ""
    cfg = env_v3(N, "unknown", episode_beats=BEATS)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(SEED), 3)
    scn, w0 = scenarios(k1, cfg, a.episodes), start_speeds(k2, cfg, a.episodes)
    ref = threshold_speeds(cfg, SEED, a.episodes) if sfx else None
    path = OUT / f"mechanism{sfx}.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    for lam in a.lams:
        det, models = detector(lam), load_models(lam, N)
        rows = json.loads((OUT / f"monitor/estimators_N10_lam{lam:g}{sfx}.json").read_text())
        for kind in a.kinds:
            name = f"{kind}_lam{lam:g}"
            if name in res:
                continue
            t0 = time.perf_counter()
            c = np.array(episodes(rows[kind])[3], float)
            est = make_estimator(kind, det, N, models)
            o = rollout(monitor_fns(est, cfg, N), mon(np.repeat(c[None], a.episodes, 0), ref), scn, cfg, k3, w0,
                        keep=("avo", "back", "co", "omega", "t_open"), belief_vec=lambda ps, est=est: jax.numpy.stack([est.vector(ps.bs)[-1], ps.bs[1]]))
            q = outcome(o, N)
            d, b, avo = o["belief"][SKIP:, :, 0] > 0.5, o["belief"][0, :, 1], o["avo"][SKIP:]
            z = b / det.sigma / np.sqrt(lam)
            res[name] = dict(config=c.tolist(), b=z.round(4).tolist(), viol=q["viol"].round(5).tolist(), dev=q["dev"].round(5).tolist(), open=q["open"].round(5).tolist(),
                             fp=((d & ~avo).sum(0) / np.maximum((~avo).sum(0), 1)).round(4).tolist(), fn=((~d & avo).sum(0) / np.maximum(avo.sum(0), 1)).round(4).tolist())
            v = q["viol"]
            print(f"{name:16s} viol {100 * v.mean():.2f}% " + "  ".join(f"[{lo:+.1f},{hi:+.1f}): {100 * v[(z >= lo) & (z < hi)].mean():5.1f}%" for lo, hi in ((-9, -1), (-1, 0), (0, 1), (1, 1.5), (1.5, 9))) +
                  f"   share of all violations from b>1: {100 * v[z > 1].sum() / v.sum():.0f}%  [{time.perf_counter() - t0:.0f} s]", flush=True)
            path.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
