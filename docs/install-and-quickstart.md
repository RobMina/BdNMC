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
.venv/bin/python -m pip install -e '.[test]'                # CPU: jax, numpy, pytest, scipy, h5py
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

**Event output:**

```bash
.venv/bin/python examples/quickstart.py phase0/cards/ref/mb_pi0_m005.dat --trials 1048576 \
    --text events.dat --summary summary.dat --hdf5 events.h5 [--weighted]
```

- **`--text`:** stock's comprehensive event format, the same lines as stock's `output_file`.
  - Layout: `Run`; then per event `event N`, the `pion`, `V`, `DM`, `Recoil_DM` and `electron` lines with stock's
    17 fields at 10 significant digits, and `endevent N`; then the ` Summary` line.
  - Tools that read stock event files read it unchanged.
- **`--summary`:** appends stock's three summary lines.
- **`--hdf5`:** one row per accepted DM in group `accepted`.
  - Arrays: `event`, `weight`, `prob`, `dm`, `electron`, `recoil_dm`, `vertex`, `parent`, `mediator` (four-vectors
    E, px, py, pz; vertex x, y, z, t), `L`, `crossing`, `block`, `trial`, `slot`.
  - Attributes: the card text, the seed and the predictions.
  - `weight` sums to the predicted number of signal events. HDF5 is the fast format; text is 10-100x slower per
    event to write (13x on a laptop core, about 120x on the GPU node).
- **`--weighted`:** writes every DM that reached the detector and passed the angle cuts, with its weight, instead of
  the unweighted (accepted) sample.

Differences from stock's files:
- the Summary line's last number is the number of events written, not the card's `samplesize` (the port runs a
  fixed number of trials);
- weighted text output writes one DM per event, with that DM's scatter probability on the `event` line. Stock writes
  one weight per event, the last DM's, which misstates events with two DM.

From Python: `bdnmc_jax.output.write_text(res, path, summary_path=None)` and `write_hdf5(res, path)`, where `res`
is `pipeline.finish(candidates, weighted=...)`, or, faster for large outputs, `pipeline.events(card, n_trials,
weighted=..., batch=..., cap=...)`: the same events, with only the accepted rows copied off the device. With
`stream=True` the rows stay on the device per batch, copies started, and `write_hdf5` writes each batch as it
arrives; `output.collect(res)` joins them in memory instead.

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
- **Precision:** float64 by default, the validated configuration. `BDNMC_JAX_PRECISION=float32` in the
  environment, before `bdnmc_jax` is first imported, runs the device code in float32 for GPUs without fast float64;
  it is not validated, and its measured cost is in [gpu-validation.md](gpu-validation.md#float32).
- **Mass scans:** masses are runtime inputs, so the cards of a scan share the compiled kernel;
  `validation/run_grid.py` runs a grid of base cards and masses to a target number of accepted events each.

### What the port accepts

| Card setting | Port |
|---|---|
| `model` | `Dark_Photon` (and `Dark_Photon_DM`) |
| `production_channel` | exactly one block, without `dist_mod`: `pi0_decay`, `eta_decay`, or `V_decay` (also `Brem_V`) with `production_distribution proton_brem` |
| `production_distribution` | for `pi0_decay`: `pi0_sanfordwang` (built-in parameters, no `sanfordwang_file`/`distribution_parameter_file`) or `bmpt` (no `max_meson_momentum`); for `eta_decay`: `k0_sanfordwang` or `bpmt_k0`, with the same limits; for either meson: `particle_list` (with `particle_list_file`, optionally `particle_list_position true`); for `V_decay`: `proton_brem` |
| `particle_list` file | as stock reads it (E ignored, rows reused from the top), but lines stock would skip or misread raise an error: fewer than four fields, an empty field, zero momentum with positions, an empty file. A relative path is opened from the working directory, as in stock |
| masses, meson decays | 2 m_chi < m_meson. Stock's choice between the on-shell `DecayDM` and the off-shell `DecayDM_Off_Shell` (meson -> gamma V* -> gamma chi chi) is reproduced: off-shell when the total branching ratio exceeds 1.3x the on-shell one, except when 2 m_chi <= m_V < m_meson with 2 m_chi >= 0.97 m_V |
| masses, `proton_brem` | 2 m_chi < m_V |
| `proton_brem` window | `zmin`, `zmax`, `ptmax` (and `ptmin`, default 0) in the channel block, with 0 <= `zmin` <= `zmax` and `ptmin` <= `ptmax`; otherwise an error (stock continues with a zero-rate channel) |
| `parent_sampler` (port option) | `"envelope"` works for the analytic meson distributions (validated for pi0 and on-shell eta; not run with off-shell decays); refused for `particle_list` and `proton_brem` |
| `signal_channel` | `NCE_electron` |
| `detector` | `sphere`, `cuboid` (rotations included) |
| `timing_cut` | must be 0 (unset) |
| used as in stock | `epsilon`, `alpha_D`, `dark_matter_mass`, `dark_photon_mass`, `POT`, `pi0_per_POT`, `meson_per_pi0`, `beam_energy`, `p_num_target` + `n_num_target` (BMPT target A), materials, `min/max_scatter_energy`, `min/max_scatter_angle`, `efficiency`, `seed` |
| ignored | `samplesize`, `max_trials`, `burn_max`, `burn_timeout`, `repeat_count`, `output_file`, `summary_file`, `output_mode` (the trial count is an argument; files are written only where asked) |
| `run` | used as the run name in text output (default: the current Unix time, as in stock) |
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
- **Port side and comparison:** run `validation/run_v_set.sh`, then `validation/compare_v.py`. For the other point
  sets: `V_SET=<set> validation/run_v_set.sh`, then `compare_v.py --config validation/v_config_<set>.json`, with
  `<set>` one of `brem`, `eta`, `list`, `offshell`. The list points need the lists from `phase0/make_lists.py`.
- **Split references:** `phase0/make_split_ref_cards.py` writes a reference point as parallel parts;
  `validation/merge_stock_parts.py` joins their electrons, and a config's `stock_runs` tells `compare_v.py` where the
  parts are.
