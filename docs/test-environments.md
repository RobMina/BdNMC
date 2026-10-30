# Test environments

The hardware and software on which the stock reference runs and the GPU-port tests in
[gpu-validation.md](gpu-validation.md) and [gpu-performance.md](gpu-performance.md) were run. Each result in those
pages names its environment by the label used here.

## L1: laptop (stock builds, CPU tests of the port)

| Item | Value |
|---|---|
| CPU | Apple M5 (arm64) |
| OS | macOS 26 (Darwin 25.6) |
| C++ compiler | Apple clang 21.0.0 (default FMA contraction), C++14 |
| Stock builds | out-of-tree builds of `RobMina/BdNMC` `bf45d75`, whose acceptance, burn-in and prediction code is the same as upstream master `6cba825`: as shipped (`-O0`, no optimisation flag in `build/makefile`) and `-O2 -g` |
| Python | 3.14.6 |
| Python packages | `jax` and `jaxlib` 0.12.0.1 (CPU), `numpy` 2.5.4, `scipy` 1.18.1, `pytest` 9.2.0 (full list: `gpu/requirements-lock.txt`) |
| JAX settings | float64 (enabled by `import bdnmc_jax`); `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` for every timed or long run, one port process at a time |
| Timing rules | stock timing runs one process per core with nothing else heavy running; reference-set runs four at a time (not used for timing) |

## G1: GPU node of a university HPC cluster (Slurm)

| Item | Value |
|---|---|
| GPU | one NVIDIA A100-SXM4-80GB per job (full device, MIG disabled), compute capability 8.0, driver 595.71.05 |
| Host | x86-64 node; each job allocated 8 CPU cores and 64 GB RAM |
| Python | 3.14.9 (a virtual environment on the cluster's scratch file system) |
| Python packages | `jax` 0.12.0.1 with `jax-cuda12-plugin` and `jax-cuda12-pjrt` 0.12.0.1 (CUDA runtime 12.9.79 from the pip wheels), `numpy` 2.5.4, `scipy` 1.18.1, `pytest` 9.2.0 |
| JAX settings | float64; `XLA_PYTHON_CLIENT_PREALLOCATE=false`; every job checks `jax.default_backend() == "gpu"` before starting |
| Code | a `git archive` of one commit of this branch per job; the stock reference data (event summaries and logs, about 1 GB) copied next to it |

The same JAX, NumPy and SciPy versions are used on L1 and G1, so differences between them come from the backend
(XLA CPU against XLA CUDA), not from the library versions.

## C1: CPU node of the same cluster (stock builds)

| Item | Value |
|---|---|
| CPU | Intel Xeon Gold 6248 (x86-64, AVX-512), 3.2 GHz during the runs; one core per job, nodes shared with other jobs |
| OS | an EL8 Linux distribution, glibc 2.28 (its `libm` provides `pow`, `exp`, `sin` and the rest) |
| C++ compiler | gcc 14.2.0 (cluster module, also needed at run time for its `libstdc++`), C++14 |
| Stock build | in-tree build of `RobMina/BdNMC` `bf45d75` (no local changes), `-O2 -fPIC` |
| Used for | the second stock reference at two eta points ([gpu-validation.md](gpu-validation.md)); stock timing on C1 is in [gpu-performance.md](gpu-performance.md#stock-bdnmc-on-a-linux-cluster-node-c1) |

A different compiler and standard library than L1 may give different events at the same seed; the sampled law is the
same.

## Not yet run

- A MIG slice of an A100 (P1 asks for one figure there).
- A whole-node comparison: a GPU against every core of the same node running stock jobs.
- GPUs other than the A100-80GB.
