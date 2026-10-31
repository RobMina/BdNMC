"""Batched (JAX) port of BdNMC.

Every module mirrors one part of stock BdNMC (the reference is RobMina/BdNMC bf45d75; the physics is that of
pgdeniverville/BdNMC master 6cba825) and cites the stock function it ports.

Physics runs in float64, the validated precision. `BDNMC_JAX_PRECISION=float32` in the environment, set before the
package is first imported, runs the device code in float32 instead (JAX's 64-bit mode off), for endpoints without
fast float64; its physics against float64 is measured in docs/gpu-validation.md. Host-side setup and pass 2 stay in
numpy float64 either way.
"""
import os

import jax

PRECISION = os.environ.get("BDNMC_JAX_PRECISION", "float64")
if PRECISION not in ("float64", "float32"):
    raise ValueError(f"BDNMC_JAX_PRECISION={PRECISION!r}: use float64 or float32")
jax.config.update("jax_enable_x64", PRECISION == "float64")
