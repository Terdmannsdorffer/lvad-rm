"""Check that JAX sees the GPU and can run a jitted matmul."""

import time

import jax
import jax.numpy as jnp


def main() -> None:
    print("jax", jax.__version__)
    for d in jax.devices():
        print(d, getattr(d, "compute_capability", ""), getattr(d, "device_kind", ""))
    f = jax.jit(lambda a: a @ a)
    x = jnp.ones((4096, 4096), dtype=jnp.float32)
    f(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        y = f(x)
    y.block_until_ready()
    dt = (time.perf_counter() - t0) / 10
    print(f"matmul 4096^2 float32: {dt * 1e3:.2f} ms, {2 * 4096**3 / dt / 1e12:.1f} TFLOP/s, y[0,0]={float(y[0, 0])}")


if __name__ == "__main__":
    main()
