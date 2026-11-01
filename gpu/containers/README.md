# Containers

Apptainer recipes for running BdNMC on any compute endpoint. Experiment-specific images (an experiment's detector
simulation together with BdNMC) belong in the experiment integration repositories, which start from these.

| Recipe | Contents | Use |
|---|---|---|
| `bdnmc-jax.def` | the JAX port (`bdnmc_jax`) with `jax[cuda12]`, numpy, scipy, h5py on Debian bookworm (glibc 2.36) | event generation on a GPU (`--nv`) or CPU; output HDF5, stock text or HEPEVT (`output.write_hepevt`) for a detector simulation |
| `bdnmc-stock.def` | stock BdNMC from the `shared_library` branch: `libBdNMC.so`, its C API headers and the `BDNMC` program, gcc 14.2.0 | stock runs, and the base for images on a current OS that call BdNMC in process |

The JAX port needs glibc >= 2.27 (its wheels are `manylinux_2_27`), so it cannot run inside an SL7 image; an
experiment whose simulation needs SL7 runs generation and simulation as two stages of one job, or calls stock
libBdNMC in process, built with that experiment's own compiler.

## Build

```bash
apptainer build bdnmc-jax.sif gpu/containers/bdnmc-jax.def
```

```bash
apptainer build bdnmc-stock.sif gpu/containers/bdnmc-stock.def
```

`--build-arg BDNMC_REF=<commit or branch>` pins the source (defaults: `gpu-port` and `shared_library`); each image
records the commit it built in `COMMIT`.

## Run

```bash
apptainer run --nv bdnmc-jax.sif /opt/BdNMC/gpu/phase0/cards/ref/mb_pi0_m005.dat --trials 16777216 --hdf5 events.h5
```

```bash
apptainer exec bdnmc-stock.sif bash -c 'cd /opt/bdnmc && BDNMC /path/to/card.dat'
```

The JAX image runs `examples/quickstart.py` by default; `apptainer exec` gives a Python with `bdnmc_jax` for other
scripts. Precision: float32 by default; `--env BDNMC_JAX_PRECISION=float64` for float64.
