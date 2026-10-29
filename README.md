# BdNMC

BdNMC is a Monte Carlo event generator for light dark matter produced in proton fixed-target experiments. It
simulates meson decays, proton bremsstrahlung and parton-level production of a mediator, the propagation of dark
matter to a detector, and its scattering there. Full documentation of the physics is in the appendix of
[arXiv:1609.01770](http://arxiv.org/abs/1609.01770).

## Quick start

```bash
git clone https://github.com/pgdeniverville/BdNMC.git
cd BdNMC
git submodule update --init --recursive   # BdNMC_LANL: required by the build
./bin/BDNMC parameter.dat
```

- **What `bin/BDNMC` does:** builds `build/main` with `make` if needed, then runs it on each parameter card given;
  with no argument it uses `parameter.dat`.
- **Card paths:** relative to the directory you run it from.
- **Run directory:** BdNMC runs from the repository root, where it finds `data/`.
- **Sample card:** `parameter.dat` is a commented example that ends with the result to expect.

The makefile sets no optimisation flag. For a faster build, see
[docs/install-and-quickstart.md](docs/install-and-quickstart.md#stock-bdnmc-c).

## Documentation

| Page | Content |
|---|---|
| [docs/install-and-quickstart.md](docs/install-and-quickstart.md) | building and running stock BdNMC; installing and running the GPU port |
| [docs/parameter-card.md](docs/parameter-card.md) | every parameter-card option: keys, blocks, defaults, allowed values, parser behaviour |
| [docs/gpu-validation.md](docs/gpu-validation.md) | how the GPU port is validated against stock BdNMC, and the results |
| [docs/gpu-performance.md](docs/gpu-performance.md) | throughput of stock BdNMC and of the GPU port |
| [docs/test-environments.md](docs/test-environments.md) | the hardware and software the tests ran on |

## GPU port

[`gpu/`](gpu) holds `bdnmc-jax`, a batched JAX port of part of BdNMC that runs on GPUs and CPUs. It reads the same
parameter cards and is validated statistically against this code.

**Implemented and validated:**
- pi0 decay with Sanford-Wang or BMPT production;
- sphere and cuboid detectors;
- `NCE_electron` scattering.

**Throughput:** on one NVIDIA A100, 86-126x one core of an optimised stock build. See [gpu/README.md](gpu/README.md).

## Layout

| Path | Content |
|---|---|
| `src/` | BdNMC sources |
| `BdNMC_LANL/` | submodule with the `DMNscattering` sources |
| `build/makefile` | build of `build/main` |
| `bin/BDNMC` | build-and-run script |
| `data/` | tables read at run time (`rratio.dat`, `delta_production_form_factor.dat`), an example Sanford-Wang parameter file (`sanfordwang_parameters.dat`) and an example card that writes a Burman-Smith particle list (`bmpt_gen.dat`) |
| `parameter.dat` | sample parameter card |
| `Parameter_Card_Generator.py` | helper that writes and runs modified cards |
| `docs/` | documentation |
| `gpu/` | JAX port |
