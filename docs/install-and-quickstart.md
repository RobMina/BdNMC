# Install and quickstart

Two codes live in this repository: stock BdNMC (C++, top level) and its batched JAX port (`gpu/`). They read the
same parameter cards ([parameter-card.md](parameter-card.md)); the port accepts only a subset of them (below).

## Stock BdNMC (C++)

**Requirements:** git, make and a C++14 compiler. `build/makefile` calls `g++`; on macOS that is Apple clang, which
works.

```bash
git clone https://github.com/pgdeniverville/BdNMC.git
cd BdNMC
git submodule update --init --recursive
```

The submodule `BdNMC_LANL` (BSD-3-Clause) provides `DMNscattering.cpp` and `DMNscattering_Baryonic.cpp`, which the
makefile needs. The build fails without it.

**Build and run:**

```bash
./bin/BDNMC parameter.dat
```

`bin/BDNMC`:
- creates `Events/`;
- runs `make main` in `build/` (an incremental build: the first run compiles everything);
- runs `build/main` on each card given, from the repository root. Card paths are relative to the directory you
  started in; an absolute path does not work, because the script prefixes it with that directory.
- With no argument it runs `parameter.dat`.

BdNMC opens `data/*.dat` relative to the working directory, and `output_file` / `summary_file` in the card are
relative to the repository root.

The sample card ends with the result to expect:

```
Predicted number of signal events = 284.31
Predicted number of signal events from channel 1 pi0_decay = 284.31
```

The card says this varies by about ±10 at its sample size of 1,000 events.

**Optimised build.** The makefile sets no optimisation flag, so the default build is `-O0`. To build with `-O2`:

```bash
cd build
rm -f *.o main && rm -rf .d
make main CXXFLAGS="-std=c++14 -O2"
cd ..
```

On the generic cards this is 3-7x faster. With the same seed it gives the same events, with printed values within
2.3e-9 relative ([gpu-performance.md](gpu-performance.md)).

**Card generator.** `Parameter_Card_Generator.py` copies a card, replaces the values of the keys you give
(`run_output_file(card, new_card, {"dark_photon_mass": 0.1, ...})`) and, by default, runs `./bin/BDNMC` on the
copy; its `test()` scans three dark-photon masses with `parameter.dat`. Run it from the repository root.

**Outputs:**
- `output_mode comprehensive` writes every particle of every event to `output_file`;
- `summary` writes only the summary line;
- `particle_list` writes parent four-momenta for reuse as `production_distribution particle_list`.
- `dm_detector_distribution` writes one line per DM that reaches the detector.

All of them, with the pitfalls of the card format, are in [parameter-card.md](parameter-card.md#output).

## JAX port (`gpu/`)

**Requirements:**
- Python >= 3.11 (tested with 3.14).
- JAX: tested with 0.12.0.1; pinned versions of the CPU environment are in `gpu/requirements-lock.txt`.
- For a GPU, an NVIDIA GPU with fast float64 (tested on the A100). Consumer and workstation GPUs run, but slowly
  in float64.

```bash
cd gpu
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'                # CPU: jax, numpy, pytest, scipy
.venv/bin/python -m pip install 'jax[cuda12]==0.12.0.1'      # NVIDIA GPU: adds the CUDA plugin and libraries
.venv/bin/python -c 'import jax; print(jax.default_backend(), jax.devices())'
.venv/bin/python -m pytest -q
```

- `import bdnmc_jax` turns on float64 in JAX; the port relies on it.
- If the backend prints `cpu` on a GPU node, the CUDA wheels did not install or do not match the driver.
- Optional: `XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX reserving most of the GPU memory at start-up.

**Quickstart:**

```bash
.venv/bin/python examples/quickstart.py phase0/cards/ref/mb_pi0_m005.dat --trials 4194304
```

prints, for example (laptop CPU):

```
backend cpu (cpu), 4194304 trials in 3.5 s (compile included), parent sampler stock
DM reaching the detector: 8777; accepted (two-pass unweighted): 3211
Predicted number of signal events (weighted) = 190.379
Predicted number of signal events (unweighted, two-pass) = 190.107
```

For comparison, stock gives 190.82 ± 0.28 (weighted) for this card. `--save out.npz` keeps the pass-1 candidates:
every DM that reached the detector, with its scatter probability, recoil electron, recoil DM, vertex, parent and
mediator.

**From Python:**

```python
import bdnmc_jax                       # enables float64
from bdnmc_jax import pipeline

res = pipeline.run("phase0/cards/ref/he_pi0_cuboid_m005.dat", n_trials=1 << 24, batch=64)
print(res["se_weighted"], res["se"], res["accepted"])
c = res["candidates"]                  # dict of arrays: prob, electron (E, px, py, pz), vertex (x, y, z, t), ...
electrons = c["electron"][res["accepted_mask"]]
```

- **Large runs:** for a run that does not fit one process, split the blocks over processes with
  `validation/run_port.py` and merge the shards with `bdnmc_jax.shards.load_all` and `pipeline.merge`. The merged
  run is identical to a single-process run.
- **`batch`:** 2^16-trial blocks per device call (8 by default). Use 64-256 on an A100; it does not change the
  results.

### What the port accepts

| Card setting | Port |
|---|---|
| `model` | `Dark_Photon` (and `Dark_Photon_DM`) |
| `production_channel` | exactly one `pi0_decay` block, without `dist_mod` |
| `production_distribution` | `pi0_sanfordwang` (built-in parameters, no `sanfordwang_file`/`distribution_parameter_file`) or `bmpt` (no `max_meson_momentum`) |
| masses | 2 m_chi <= m_pi0, and stock must take its on-shell `DecayDM` path: the total pi0 -> gamma chi chi branching ratio within 1.3x of pi0 -> gamma V x V -> chi chi, or 2 m_chi <= m_V < m_pi0 with 2 m_chi >= 0.97 m_V |
| `signal_channel` | `NCE_electron` |
| `detector` | `sphere`, `cuboid` (rotations included) |
| `timing_cut` | must be 0 (unset) |
| used as in stock | `epsilon`, `alpha_D`, `dark_matter_mass`, `dark_photon_mass`, `POT`, `pi0_per_POT`, `meson_per_pi0`, `beam_energy`, `p_num_target` + `n_num_target` (BMPT target A), materials, `min/max_scatter_energy`, `min/max_scatter_angle`, `efficiency`, `seed` |
| ignored | `samplesize`, `max_trials`, `burn_max`, `burn_timeout`, `repeat_count`, `run`, `output_file`, `summary_file`, `output_mode` (the trial count is an argument; nothing is written unless asked) |
| `weighted` | not needed: the port always returns both the weighted and the two-pass unweighted prediction |

Anything else raises `NotImplementedError` rather than running with a silent difference.
- **Seeds:** the same seed gives the same events in the port, on any batch size or device. It does not give
  stock's events: the random-number schemes differ.
- **Card parser:** it follows stock's semantics, including its quirks. Its production-channel block also accepts
  the extra keys of the `RobMina/BdNMC` fork (`secondary_spectrum_file`, `z_production_start`/`_end`,
  `spectrum_prune_fraction`, `brem_table_*`, `target_Z`, `target_A`, `target_n_density`), which upstream's parser
  does not know.

## Running the validation

Each validation step and its tools are in [gpu-validation.md](gpu-validation.md).
- **Stock side:** build stock at `-O2` into `gpu/stock/O2/` (`gpu/phase0/run_stock.sh` expects it there), generate
  the reference cards with the `phase0/make_*_cards.py` scripts, run them with `run_stock.sh`, and extract the
  electrons with `validation/extract_stock.py`.
- **Port side and comparison:** run `validation/run_v_set.sh`, then `validation/compare_v.py`.
