"""Phase 7 diagnostic (no RL): how much of the attainable advantage of counting is hidden by the flow sensor and by the
controller class? True counter, N = 10, for several flow-error scales and nominal-tracking gains (GPU).

    python scripts/phase7_headroom.py
Writes results/phase7/headroom.json. Run before launching the agents.
"""

from __future__ import annotations

import json

import numpy as np
from phase7_common import OUT, detector, outcome, rollout, scenarios, start_speeds  # noqa: F401
from phase7_monitor import bli, grid, mon, pareto, run_configs

import lvadrm.estimators as es
from lvadrm.presets import env_v3

N, BEATS, N_SCN = 10, 400, 96
Q = (4.5, 5.0, 5.5, 6.0, 6.5)


def main() -> None:
    res = {}
    det = detector(0.0, perfect=True)
    for regime in ("fixed", "unknown"):
        for scale in (1.0, 0.3, 0.0):
            for gain in (0.01, 0.04):
                es.W_GAIN = gain
                cfg = env_v3(N, regime, episode_beats=BEATS, flow_noise_scale=scale)
                fns = es.monitor_fns(es.make_estimator("true", det, N), cfg, N)
                c = grid(Q, range(2, N), (0.5,))
                o = run_configs(fns, mon, c, cfg, N, 777, N_SCN)
                v, d, co = o["viol"].mean(1), o["dev"].mean(1), o["co"].mean(1)
                ok = v <= 0.01
                j = int(np.argmin(np.where(ok, d, 9)))
                jc = int(np.argmax(np.where(ok, co, 0)))
                row = dict(dev=float(d[j]), cfg=c[j].tolist(), viol=float(v[j]), open=float(o["open"].mean(1)[j]), co=float(co[j]),
                           co_max=float(co[jc]), cfg_co=c[jc].tolist(), front=[(float(v[i]), float(d[i]), c[i].tolist()) for i in pareto(v, d)])
                res[f"{regime}_scale{scale}_gain{gain}"] = row
                print(f"{regime} scale {scale} gain {gain}: min dev {row['dev']:.3f} {row['cfg']} viol {100 * row['viol']:.2f}% open {row['open']:.2f} "
                      f"CO {row['co']:.2f} | max CO {row['co_max']:.2f} {row['cfg_co']}", flush=True)
    (OUT / "headroom.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
