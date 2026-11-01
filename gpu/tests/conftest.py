"""The test suite runs in float64 (V0 compares with stock at 1e-12; the reproducibility tests compare to the bit), so
it pins BDNMC_JAX_PRECISION before bdnmc_jax is imported; float32 is tested in subprocesses (test_precision.py).
An explicit BDNMC_JAX_PRECISION in the environment is kept."""
import os

os.environ.setdefault("BDNMC_JAX_PRECISION", "float64")
