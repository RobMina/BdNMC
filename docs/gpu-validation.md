# GPU port: physics validation

How the JAX port in [`gpu/`](../gpu) is checked against stock BdNMC, and the results so far. Environments (L1 laptop,
G1 A100 node) are described in [test-environments.md](test-environments.md); throughput is in
[gpu-performance.md](gpu-performance.md).

**Status (2026-10-29).** For the scope the port implements (pi0 decay with Sanford-Wang or BMPT production, on-shell
`DecayDM`, sphere and cuboid detectors, `NCE_electron`), the default configuration (`parent_sampler="stock"`) passes
V0, V1 and V2 on CPU (L1) and on an A100 (G1). The opt-in `parent_sampler="envelope"` **fails V2** and is not
validated. Nothing else (eta, brem, off-shell decays, other detectors or signals, output files) is validated yet.

## Method

The port is compared with stock BdNMC statistically, not event by event: it uses its own random-number scheme, so
the same seed gives different events. Four levels:

| Level | What is compared | Pass criterion |
|---|---|---|
| Unit tests | parser, geometry, kinematics, sampler laws, reproducibility | `python -m pytest` in `gpu/` |
| V0 deterministic | closed-form functions, port against the stock C++ called directly | relative difference <= 1e-12 + 64 eps x condition number; `Ldet` within 1e-9 m with identical hit/miss |
| V1 normalisation | predicted signal events `se` per point | pulls against stock weighted `se` (below) |
| V2 distributions | shapes of 14 histograms per point | two-sample chi2 p-values (below) |

Validation uses generic, experiment-independent cards only (no experiment's meson lists or detector geometry).

### Reference build and cards

- Stock BdNMC built from `RobMina/BdNMC` `bf45d75` (environment L1), at `-O0` (as shipped) and `-O2`. Its
  acceptance, burn-in, prediction and BMPT code is the same as upstream master `6cba825`. With the same seed `-O0`
  and `-O2` give the same events, with printed values within 2.3e-9 relative, so `-O2` is used for the references.
- Four base cards in `gpu/phase0/cards/`, all `model Dark_Photon`, epsilon 1e-3, alpha_D 0.1, m_V = 100 MeV,
  `signal_channel NCE_electron`, carbon + hydrogen detector material, recoil window 35 MeV up to the beam
  energy (or 0.9 GeV):

  | Card | Beam | Production | Detector |
  |---|---|---|---|
  | `mb_pi0` | 8.9 GeV | `pi0_decay`, `pi0_sanfordwang` (upstream `parameter.dat` setup) | sphere, r = 5 m, 491 m downstream |
  | `he_pi0_cuboid` | 120 GeV, carbon target | `pi0_decay`, `bmpt` | cuboid 3 x 3 x 10 m, 500 m downstream |
  | `mb_brem` | 8.9 GeV | `V_decay`, `proton_brem` (z 0.3-0.7, pT < 0.2 GeV) | as `mb_pi0` |
  | `he_brem_cuboid` | 120 GeV | `V_decay`, `proton_brem` (z 0.1-0.9, pT < 1 GeV) | as `he_pi0_cuboid` |

- Each base card at m_chi = 1, 5, 20 and 45 MeV gives 16 reference points (`phase0/make_ref_cards.py`).

### Stock reference sets

| Set | Content | Generator |
|---|---|---|
| unweighted (`ref`) | one run per point: about 1e6 recoil electrons for the pi0 points, 2e7 trials for brem; event files in comprehensive mode | `phase0/make_ref_cards.py` |
| weighted (`ref_w`) | `weighted true`, 8 independent-seed replicas per point; 5e8 / 4e7 / 2e7 / 2e7 trials per point in all for `mb_pi0` / `he_pi0` / `mb_brem` / `he_brem` | `phase0/make_weighted_cards.py` |
| long weighted (`ref_wl`) | `he_pi0` only: one weighted run of 8.6e7 trials per mass, as long as the unweighted run | `phase0/make_long_weighted_cards.py` |

Stock's weighted mode is the main V1 reference. Its unweighted prediction is biased (next section). The weighted
error is the spread of the replicas.

### Stock behaviours that matter for the comparison

Measured on the reference sets (`phase0/bias_weighted.py`, `phase0/compare_long_weighted.py`):

- **Unweighted `se` is biased high by the running pMax.** Each event is accepted against the pMax known when it is
  drawn, but the prediction uses the final pMax.

  | Card | unweighted / weighted - 1, at 1, 5, 20, 45 MeV | Combined |
  |---|---|---|
  | `mb_pi0` (Sanford-Wang) | +0.40, +0.48, +0.35, +0.16 % | +0.34 ± 0.08 % |
  | `he_pi0_cuboid` (BMPT; against the long weighted runs) | +0.43, +0.27, +0.62, +0.39 % | +0.44 ± 0.07 % |
  | `mb_brem` | -0.00, +0.04, -0.09, +0.08 % | +0.03 ± 0.04 % |
  | `he_brem_cuboid` | +0.09, +0.01, -0.00, +0.04 % | +0.03 ± 0.05 % |

- **BMPT's parent sampler has a run-length bias.** It keeps its own running maximum (10,000 burn-in draws), so
  early parents are spread too widely. 5e6-trial weighted runs come out 0.20 ± 0.07 % below an 8.6e7-trial run
  (-0.24, +0.11, -0.27, -0.24 % at the four masses).
- Behaviours the port reproduces on purpose:
  - Sanford-Wang uses a fixed bar (54.25) and its built-in 8.89 GeV parameters whatever `beam_energy` says; values
    above the bar are clipped.
  - `min/max_scatter_energy` are total electron energies.
  - The branching-ratio integrator counts its first node twice (a few 1e-4 high).
  - `ThetaEe` can return NaN at the recoil endpoint; the port fails on it rather than writing it.
  - The value of a card key keeps everything after the first space.

The port differs from stock where batching needs it:
- it unweights against the global maximum of the whole run (two passes), so its unweighted `se` is unbiased;
- the trial count is fixed (stock stops at `samplesize` events);
- the BMPT bar is fixed (a grid maximum x 1.02), so it samples the density without clipping, and it fails if any
  proposal exceeds the bar.

## V0: function values against the stock C++

`gpu/v0/v0_dump.cpp` links the stock `-O2` library and prints 10,172 values at `%.17g`:
- `EeTMax`, `EeTMin`, `sigma2`, `dsigmadEe`, `ThetaEe`;
- `Ldet` for a sphere, a cuboid and a rotated cuboid (6,000 rays);
- the pi0 branching ratios;
- the Sanford-Wang density and the BMPT invariant cross section.

`tests/test_v0.py` compares the port with them.
- **Tolerance:** (1e-12 + 64 eps x condition number) relative. The condition number covers cancellation (EeTMax at
  high energy, `F2(Emax) - F2(Emin)`, BMPT near xR = 1, Sanford-Wang at p = p_max), where clang's FMA contraction
  and XLA round differently by about eps x condition.
- **`ThetaEe` endpoint:** rows at the recoil endpoint may be NaN or <= 3e-8 in either code.
- **Result:** passes on L1 and on G1.

## V1 and V2

Thresholds and binning were frozen in `gpu/validation/v_config.json` before the first port-against-stock comparison
(sha256 `7058b254019718dd6d98f269a6f212398abd7c57006f8b0e2e279aa971c0467a`, recorded in every result).

- **Points:** the eight pi0 points (`mb_pi0` and `he_pi0_cuboid` at 1, 5, 20, 45 MeV).
- **Port runs:** 5e8 trials per `mb_pi0` point and 4e7 per `he_pi0` point, the reference card's seed, 4 block-range
  shards per point merged exactly (`validation/run_v_set.sh`).
- **V1:**
  - Port weighted `se` = sum of scatter probabilities / trials x `Vnum`; its sigma comes from per-trial sums.
  - Compared with stock weighted `se` (replica spread).
  - Pass: every |pull| <= 3; |mean pull| x sqrt(8) <= 3; chi2 over the 8 points p >= 0.01.
  - Secondary: within max(1 %, 2 sigma) of stock unweighted `se`.
- **V2:**
  - Port candidates with prob > 0, weighted by prob, against every recoil electron of the stock unweighted event
    file (with its incoming DM and parent pion).
  - Histograms: E_e, theta_e, E_e theta_e^2, E_chi, theta_chi, vertex x/y/z, parent p, parent theta, and theta_e in
    four E_e slices (14 per point, 112 in all).
  - Binning: 40 bins of equal stock population, >= 25 stock events per bin; two-sample chi2 with sum-of-squared
    weights, ndf 39.
  - Pass: KS of the 112 p-values against uniform p > 0.01, every p > 1e-4, and the fraction above the stock 99th
    percentile of E_e and theta_e within 3 sigma.
  - Fail closed: any NaN, under-filled stock bin or empty port bin fails.
- Tools: `validation/extract_stock.py`, `validation/compare_v.py`.

### Results

| Configuration | Env | Commit | V1: max \|pull\| / mean pull / chi2 p | V2: KS p / min p / max tail pull | Result |
|---|---|---|---|---|---|
| stock sampler, CPU | L1 | `846cead` | 2.15 / +0.80 / 0.47 | 0.50 / 0.0028 / 2.51 | **pass** |
| stock sampler, GPU | G1 | `846cead` | same as CPU to 11-12 digits | same | **pass** |
| stock sampler, GPU, traced masses + batching | G1 | `7c3fbe0` | 2.15 / +0.80 / 0.47 | 0.50 / 0.0028 / 2.51 | **pass** |
| envelope sampler, GPU | G1 | `ed09f8d` | 2.51 / +0.84 / 0.031 | 0.12 / **5.6e-5** / 2.31 | **fail** (V2 min p < 1e-4) |

Per point, stock sampler, V1:

| Point (1, 5, 20, 45 MeV) | Port / stock weighted - 1 | Pull |
|---|---|---|
| `mb_pi0` | +0.14, +0.13, +0.12, +0.04 % | +0.90, +0.69, +0.59, +0.16 |
| `he_pi0_cuboid` | +0.05, +0.13, +0.36, +0.19 % | +0.37, +0.64, +2.15, +0.88 |

- All eight pulls are positive (about 1 % likely by chance), though every criterion passes. For `he_pi0` this is
  BMPT's run-length bias in the 5e6-trial stock replicas. Against the long weighted runs the pulls are -1.51,
  +1.48, +0.67, -0.26. For `mb_pi0` the mean pull is +0.58 over 4 points: not significant, not explained.
- Secondary: the port is 0.03-0.62 % below stock unweighted `se`, inside the 1 % floor, the sign and size of the
  running-pMax bias.

### The envelope sampler (not validated)

`parent_sampler="envelope"` samples the same parent law as stock's proposal, with a piecewise bound on a (p, v) grid
instead of one bar (see `gpu/src/bdnmc_jax/parents.py`). It draws different random numbers.
- **Its V2 fails:** parent theta at `he_pi0` 45 MeV has p = 5.6e-5 (and 0.018 at 1 MeV).
- **V1** passes, but with chi2 p = 0.031 (pulls +2.06, +0.07, +0.58, -0.76 for `mb_pi0` and +2.51, -0.27, +2.30,
  +0.19 for `he_pi0`).
- **The high-statistics check found no difference in the parent law.** `validation/compare_parent_samplers.py` on
  G1 drew 1.34e8 parents from each sampler:
  - `he_pi0` (BMPT): p, theta (log bins down to 1e-5 rad), pT and v = theta / theta*(p) give p = 0.54, 0.40, 0.88
    and 0.14;
  - `mb_pi0` (Sanford-Wang): p, theta and pT give p = 0.64, 0.53 and 0.54;
  - no proposal exceeded its bound.
- The V2 failure may be a fluctuation (with 112 histograms, a minimum p this low happens about 0.6 % of the time)
  or a difference confined to the parents of detected DM. Until a fresh, pre-declared V1/V2 run settles it, the
  envelope stays opt-in and unvalidated.

## Reproducibility

- **Per seed:** results do not depend on the batch size (blocks per device call), the candidate cap or the split
  into shards; tests check this bit for bit on CPU. Each 2^16-trial block's random numbers depend only on (seed,
  channel, block).
- **GPU against CPU:** with the stock sampler, the GPU runs give the same number of detector hits and accepted
  events at all eight points as the CPU run, and `se` within 6e-14 relative.
  - Per candidate (64 blocks at two points), the candidate sets are identical and values agree within 3.6e-6
    relative (`prob`) and 2.8e-6 (`recoil_dm`, a difference of nearly equal momenta).
  - The two backends round transcendental functions differently, so this is not bit-identical.
- **Code changes:** making masses runtime inputs (`7c3fbe0`) moved values by up to 1.6e-8 relative (`prob` near the
  `sigma2` threshold) with unchanged candidate sets (`validation/check_shard_repro.py`).

## Unit tests

`python -m pytest` in `gpu/` (42 tests; 51-68 s on L1, about 2 min on G1, where compilation dominates):
- **Parser:** stock quirks.
- **Geometry:** sphere and cuboid path lengths against analytic results.
- **Physics:**
  - `sigma2` against the integral of `dsigmadEe`;
  - inverse kinematics;
  - branching ratios and `Vnum` against stock logs (5e-6);
  - the integrator's stock behaviour.
- **V0.**
- **Pipeline:**
  - finite outputs;
  - same seed gives the same result, and a shorter run is a prefix of a longer one;
  - momentum conservation at the scatter;
  - zero-probability hits are never accepted;
  - no recompile across masses;
  - batch and cap invariance.
- **Parent samplers:**
  - theta* against xR;
  - envelope against stock law (KS);
  - bound violations reported.

## Not validated yet

eta decay, `proton_brem`, off-shell decays, `particle_list` production, cylinder detectors, timing cuts,
`dist_mod`, signal channels other than `NCE_electron`, models other than `Dark_Photon`, and event-file output.
