"""Batched (JAX) port of BdNMC.

Every module mirrors one part of stock BdNMC (the reference is RobMina/BdNMC bf45d75; the physics is that of
pgdeniverville/BdNMC master 6cba825) and cites the stock function it ports.

Precision: float32 by default (1.4-2x faster than float64 on an A100 or V100, 3-9x on GPUs with slow float64; it
agrees with float64 within about 0.1 % in rate and in every validated distribution, see docs/gpu-validation.md).
`BDNMC_JAX_PRECISION=float64` in the environment, set before the package is first imported, runs the device code in
float64, the precision stock uses and V0 compares at (JAX's 64-bit mode on). Host-side setup and pass 2 sums stay in
numpy float64 either way. `PRECISION` says which one this process runs.
"""
import os

import jax

PRECISION = os.environ.get("BDNMC_JAX_PRECISION", "float32")
if PRECISION not in ("float64", "float32"):
    raise ValueError(f"BDNMC_JAX_PRECISION={PRECISION!r}: use float64 or float32")
jax.config.update("jax_enable_x64", PRECISION == "float64")
