# bdnmc-jax: a batched (JAX) port of BdNMC

A JAX port of BdNMC's generators that runs on GPUs and CPUs, validated statistically against stock BdNMC with
generic, experiment-independent parameter cards.

**Status:**
- **Validated:** pi0-decay dark-photon production with Sanford-Wang or BMPT parents, on-shell decays, sphere and
  cuboid detectors and `NCE_electron` scattering pass the V0, V1 and V2 tests on CPU and on an NVIDIA A100, with
  either parent sampler. The A100 gives the same counts as the CPU.
- **Proton bremsstrahlung** (`V_decay` with `proton_brem`, stock parent sampler) passes V0 on CPU, and V1 and V2
  on an A100.
- **eta decay** (`k0_sanfordwang`, `bpmt_k0`) passes V0, and V1 and V2 on an A100 with either parent sampler.
- **Off-shell decays** (meson -> gamma V* -> gamma chi chi, stock sampler) pass V1 and V2 on an A100 for pi0; their
  V0 entries wait for a harness rebuild.
- **`particle_list`** passes V1 on an A100; V2 passes except for the parent histograms, whose inputs the port and
  stock share ([docs](../docs/gpu-validation.md#particle_list)).
- **The same five sets pass again on an NVIDIA V100** in float64 (`particle_list` with the same parent-only V2
  failure).
- **Output:** stock's text event format or HDF5.
- **Throughput** on an A100 with HDF5 output, in accepted events per second against one core of stock `-O2`:
  - stock parent sampler: 158x (Sanford-Wang card), 120x (BMPT card);
  - envelope sampler: 250x and 873x;
  - brem (stock sampler): 293x on the 120 GeV card, 90-116x on the 8.9 GeV card (5.9 M rows per run; the rate
    depends on the shared node's host load).
  - Other GPU types: [docs](../docs/gpu-performance.md#all-four-cards-on-several-gpu-types-commit-85db916).
- **float32:** `BDNMC_JAX_PRECISION=float32` for GPUs without fast float64; not validated, cost measured in the
  [docs](../docs/gpu-validation.md#float32).
- **Not yet ported:** other channels, models and signals; the envelope sampler for brem.

| Page | Content |
|---|---|
| [docs/install-and-quickstart.md](../docs/install-and-quickstart.md) | install (CPU and GPU), quickstart, Python API, which card settings the port accepts |
| [docs/parameter-card.md](../docs/parameter-card.md) | every parameter-card option of stock BdNMC |
| [docs/gpu-validation.md](../docs/gpu-validation.md) | physics validation: method, stock reference sets, V0/V1/V2 results, reproducibility |
| [docs/gpu-performance.md](../docs/gpu-performance.md) | stock and port throughput, profile |
| [docs/test-environments.md](../docs/test-environments.md) | the hardware and software the tests ran on |

**Design** (option A of the plan: stock's physics and samplers, batched):
- **Blocks:** trials run in blocks of 2^16. Each block's random numbers depend only on (seed, channel, block), so
  results do not depend on the batch size, the device or how a run is split into shards.
- **Pass 1** keeps every DM that reaches the detector, with its recoil electron already drawn.
- **Pass 2** unweights against the maximum scatter probability of the whole run, so the unweighted prediction is
  free of stock's running-maximum bias.
- **No recompile per mass:** masses and couplings are runtime inputs.
- **Parent sampler:** `parent_sampler="envelope"` is an opt-in parent sampler for the same law with much higher
  acceptance. It is validated; the stock proposal stays the default.
- **Output:** `bdnmc_jax.output` writes the accepted events in stock's text format or as HDF5.

| Path | Role |
|---|---|
| `src/bdnmc_jax/` | the port: `card`, `detector`, `electron`, `branching`, `kinematics`, `decay`, `offshell`, `parents`, `particle_list`, `brem`, `rng`, `pipeline`, `shards`, `output` |
| `examples/quickstart.py` | run one card, print the predicted signal events, optionally write text or HDF5 events |
| `tests/` | pytest (`python -m pytest`); `test_v0.py` needs `v0/v0_dump.txt` |
| `v0/` | V0 harness: C++ dump of stock function values, and its output `v0_dump.txt` (`v0/README.md`) |
| `phase0/cards/` | the ten generic base cards; `ref/` (stock unweighted reference: 40 points, 38 cards, as the 5 MeV brem points reuse the base-card runs), `ref_w/` (weighted replicas), `ref_wl/` (long weighted runs), `ref_r2/` (repeat references), `ref_s/` (split references), each with the generator script that wrote it; `phase0/make_lists.py` writes the synthetic particle lists |
| `phase0/run_stock.sh` | `run_stock.sh <build> <card-stem>`: one card through a stock build in `stock/<build>/` (not in the repo), outputs in `phase0/out/` |
| `phase0/bias_weighted.py`, `phase0/compare_long_weighted.py` | stock unweighted against weighted `se` (running-pMax bias), and BMPT's run-length bias |
| `validation/v_config*.json` | V1/V2 thresholds and binning for the pi0, brem, eta, list and off-shell points, each frozen before its first port-against-stock comparison; the pre-declared repeat and second-reference rules beside them |
| `validation/run_v_set.sh`, `run_port.py`, `extract_stock.py`, `merge_stock_parts.py`, `compare_v.py` | V1 (normalisation) and V2 (distributions) of the port against the stock reference sets |
| `validation/compare_stock_refs.py`, `diagnose_ties.py` | one stock reference against another; repeated values at bin edges |
| `validation/check_shard_repro.py` | rerun blocks of a saved shard and compare (code changes, CPU against GPU) |
| `validation/compare_parent_samplers.py` | high-statistics check of the envelope parent sampler against the stock proposal |
| `validation/bench.py`, `validation/profile_port.py`, `profile_host.py` | throughput (compile, steady state, device only) and where the time goes, on the device and on the host |
| `validation/run_grid.py` | goal P2: a grid of base cards and masses to a target number of accepted events each, end to end |
| `validation/compare_precision.py`, `precision_v0.py`, `diagnose_nonfinite.py` | float32 against float64 (V1/V2 statistics), the V0 checks as an error report, and rows of a block with a non-finite electron |
| `validation/summarize_v.py` | one line per V1/V2 result file, optionally per point and histogram |

Outputs (`phase0/out/`, `validation/out/`), stock builds (`stock/`) and the virtual environment (`.venv/`) are not
tracked. Pinned Python versions for the CPU environment: `requirements-lock.txt`.
