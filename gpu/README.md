# bdnmc-jax: a batched (JAX) port of BdNMC

A JAX port of BdNMC's generators that runs on GPUs and CPUs, validated statistically against stock BdNMC with
generic, experiment-independent parameter cards.

**Status:**
- **Validated:** pi0-decay dark-photon production with Sanford-Wang or BMPT parents, on-shell decays, sphere and
  cuboid detectors and `NCE_electron` scattering pass the V0, V1 and V2 tests on CPU and on an NVIDIA A100, with
  either parent sampler. The A100 gives the same counts as the CPU.
- **Proton bremsstrahlung** (`V_decay` with `proton_brem`, stock parent sampler) passes V0 on CPU, and V1 and V2
  on an A100.
- **Output:** stock's text event format or HDF5.
- **Throughput** with HDF5 output, in accepted events per second against one core of stock `-O2`:
  - stock parent sampler: 126x (Sanford-Wang card), 109x (BMPT card);
  - envelope sampler: 188x and 500x.
  - brem (stock sampler): 140x on the 120 GeV card; 43x on the 8.9 GeV card, where writing the HDF5 file is 58 %
    of the run.
- **Not yet ported:** eta and other channels, models and signals; the envelope sampler for brem.

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
| `src/bdnmc_jax/` | the port: `card`, `detector`, `electron`, `branching`, `kinematics`, `decay`, `parents`, `brem`, `rng`, `pipeline`, `shards`, `output` |
| `examples/quickstart.py` | run one card, print the predicted signal events, optionally write text or HDF5 events |
| `tests/` | pytest (`python -m pytest`); `test_v0.py` needs `v0/v0_dump.txt` |
| `v0/` | V0 harness: C++ dump of stock function values, and its output `v0_dump.txt` (`v0/README.md`) |
| `phase0/cards/` | the four generic base cards; `ref/` (stock unweighted reference, 16 points), `ref_w/` (weighted replicas), `ref_wl/` (long weighted runs), each with the generator script that wrote it |
| `phase0/run_stock.sh` | `run_stock.sh <build> <card-stem>`: one card through a stock build in `stock/<build>/` (not in the repo), outputs in `phase0/out/` |
| `phase0/bias_weighted.py`, `phase0/compare_long_weighted.py` | stock unweighted against weighted `se` (running-pMax bias), and BMPT's run-length bias |
| `validation/v_config.json`, `v_config_brem.json` | V1/V2 thresholds and binning for the pi0 and brem points, each frozen before its first port-against-stock comparison |
| `validation/run_v_set.sh`, `run_port.py`, `extract_stock.py`, `compare_v.py` | V1 (normalisation) and V2 (distributions) of the port against the stock reference sets |
| `validation/check_shard_repro.py` | rerun blocks of a saved shard and compare (code changes, CPU against GPU) |
| `validation/compare_parent_samplers.py` | high-statistics check of the envelope parent sampler against the stock proposal |
| `validation/bench.py`, `validation/profile_port.py` | throughput (compile, steady state, device only) and where the time goes |

Outputs (`phase0/out/`, `validation/out/`), stock builds (`stock/`) and the virtual environment (`.venv/`) are not
tracked. Pinned Python versions for the CPU environment: `requirements-lock.txt`.
