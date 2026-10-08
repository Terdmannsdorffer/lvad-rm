"""GPU throughput (beats per second) of the per-beat simulator in float32, and float32 accuracy.

* Throughput: ``jit(vmap(simulate_beat))`` over 1024-8192 environments with random
  patients (Rs, Emax, HR) and speeds, for several ``lax.scan`` unroll factors.
* Accuracy: periodic regime of the sweep_dt.py cases in float32 at the chosen step,
  compared with the float64 fine-step reference saved by sweep_dt.py.

Run sweep_dt.py first. Writes results/phase1/bench_gpu.json.
"""

from __future__ import annotations

import math
import time

import common
import jax
import jax.numpy as jnp
import numpy as np

from lvadrm.calibration import batch_axes, run_beats
from lvadrm.integrate import simulate_beat
from lvadrm.model import initial_state
from lvadrm.params import make_params

N_ENVS = [1024, 2048, 4096, 8192, 16384, 32768]
UNROLLS = [1, 4, 8, 16]
N_TIMED_BEATS = 5


def random_patients(key, n):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    p = make_params(
        rs=jax.random.uniform(k1, (n,), minval=0.5, maxval=2.0),
        emax=jax.random.uniform(k2, (n,), minval=0.6, maxval=2.0),
        hr=jax.random.uniform(k3, (n,), minval=60.0, maxval=135.0),
        emin=common.PUMP_EMIN,
    )
    omega = jax.random.uniform(k4, (n,), minval=8000.0, maxval=15000.0)
    return p, omega


def main() -> None:
    sweep = common.load_json("sweep_dt.json")
    dt = sweep["recommended_dt"]
    n_inner = math.ceil(1.0 / dt / common.N_SEG)  # dt at HR 60 (the lowest HR) <= recommended dt
    n_sub = common.N_SEG * n_inner
    vtot = float(sweep["vtot"])
    print(jax.devices(), f"dt_rec={dt}, N_sub={n_sub} (n_seg={common.N_SEG}, n_inner={n_inner})")

    out = {"device": str(jax.devices()[0]), "dtype": "float32", "N_sub": n_sub, "n_seg": common.N_SEG, "n_inner": n_inner, "throughput": []}
    for n_envs in N_ENVS:
        p, omega = random_patients(jax.random.PRNGKey(0), n_envs)
        y0 = jnp.broadcast_to(initial_state(p, vtot), (n_envs, 6))
        axes = batch_axes(p)
        for unroll in UNROLLS:
            step = jax.jit(
                jax.vmap(
                    lambda y, pp, w: simulate_beat(y, pp, w, n_seg=common.N_SEG, n_inner=n_inner, unroll=unroll)[:2],
                    in_axes=(0, axes, 0),
                )
            )
            t0 = time.perf_counter()
            y, _ = step(y0, p, omega)
            jax.block_until_ready(y)
            t_compile = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(N_TIMED_BEATS):
                y, stats = step(y, p, omega)
            jax.block_until_ready(y)
            el = time.perf_counter() - t0
            bps = n_envs * N_TIMED_BEATS / el
            finite = bool(jnp.all(jnp.isfinite(y)))
            out["throughput"].append(
                {"n_envs": n_envs, "unroll": unroll, "beats_per_s": bps, "s_per_batch_beat": el / N_TIMED_BEATS,
                 "compile_s": t_compile, "finite": finite}
            )
            print(f"n_envs={n_envs:5d} unroll={unroll:2d}: {bps:10.0f} beats/s, {1e3 * el / N_TIMED_BEATS:7.1f} ms per batch-beat, "
                  f"compile {t_compile:5.1f} s, finite={finite}")

    # float32 accuracy against the float64 fine-step periodic regime
    out["float32_vs_float64"] = {}
    for name, case in sweep["cases"].items():
        p = make_params(emin=common.PUMP_EMIN, **({"hr": common.PUMP_HR} | case["params"]))
        hr = float(p.heart.hr)
        n_in = math.ceil(60.0 / hr / dt / common.N_SEG)
        _, st, _ = run_beats(p, vtot, case["omega"], n_beats=30, pump_on=case["pump_on"], n_seg=common.N_SEG, n_inner=n_in)
        last = {k: float(np.asarray(v)[-1]) for k, v in st._asdict().items()}
        ref = case["steady_reference"]
        errs = {}
        for k in ("edv", "esv", "edp", "lvp_max", "aop_max", "aop_min", "aop_mean", "lap_mean", "co", "qlv", "v_aortic", "qp_min", "qp_max", "qp_mean"):
            if ref[k] is None or (isinstance(ref[k], float) and np.isnan(ref[k])):
                continue
            floor = 0.1 if k in ("co", "qlv") else 1.0
            errs[k] = abs(last[k] - ref[k]) / max(abs(ref[k]), floor)
        out["float32_vs_float64"][name] = {"max_err": max(errs.values()), "worst": max(errs, key=errs.get), "vtot_drift_ml": last["vtot"] - vtot}
        print(f"float32 {name:32s}: max rel err {100 * max(errs.values()):.4f}% ({max(errs, key=errs.get)}), "
              f"Vtot drift after 30 beats {last['vtot'] - vtot:+.2e} ml")

    common.save_json("bench_gpu.json", out)
    print("wrote results/phase1/bench_gpu.json")


if __name__ == "__main__":
    main()
