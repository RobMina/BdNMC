# bdnmc-jax: a batched (JAX) port of BdNMC

A JAX port of BdNMC's generators that runs on GPUs (and on CPUs), validated statistically against stock BdNMC
with generic, experiment-independent parameter cards. Work in progress: the Phase 1 scope below is implemented and
validated on CPU; GPU validation and timing are next.

**Implemented:** card parser (stock quirks included), sphere and cuboid detectors, `NCE_electron` (`sigma2`,
`dsigmadEe`, recoil sampler), pi0 branching ratios and `Vnum` (stock's own integrator), on-shell `DecayDM`,
Sanford-Wang and BMPT pi0 production, a counter-based RNG (key = seed, channel, 2^16-trial block, stage, round) and a
two-pass pipeline that unweights against the global maximum. Cards using anything else (brem, eta, off-shell
decays, other detectors, timing cuts) are refused.

| Path | Role |
|---|---|
| `src/bdnmc_jax/` | the port: `card`, `detector`, `electron`, `branching`, `kinematics`, `decay`, `parents`, `rng`, `pipeline`, `shards` |
| `tests/` | pytest (`python -m pytest`); `test_v0.py` needs `v0/v0_dump.txt` |
| `v0/` | V0 harness: C++ dump of stock function values, and its output `v0_dump.txt` (`v0/README.md`) |
| `phase0/cards/` | the four generic base cards; `ref/` (stock unweighted reference, 16 points), `ref_w/` (weighted replicas), `ref_wl/` (long weighted runs), each with the generator script that wrote it |
| `phase0/run_stock.sh` | `run_stock.sh <build> <card-stem>`: one card through a stock build in `stock/<build>/` (not in the repo), outputs in `phase0/out/` |
| `phase0/bias_weighted.py`, `phase0/compare_long_weighted.py` | stock unweighted `se` against stock weighted `se` (running-pMax bias), and BMPT run-length bias |
| `validation/v_config.json` | V1/V2 thresholds and binning, frozen before the first port-against-stock comparison |
| `validation/run_v_set.sh`, `run_port.py`, `extract_stock.py`, `compare_v.py` | V1 (normalisation) and V2 (distributions) of the port against the stock reference sets |

Outputs (`phase0/out/`, `validation/out/`), stock builds (`stock/`) and the virtual environment (`.venv/`) are not
tracked. Pinned Python versions: `requirements-lock.txt`.
