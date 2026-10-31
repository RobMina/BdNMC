# GPU port: physics validation

How the JAX port in [`gpu/`](../gpu) is checked against stock BdNMC, and the results so far. Environments (L1 laptop,
G1 A100 node) are described in [test-environments.md](test-environments.md); throughput is in
[gpu-performance.md](gpu-performance.md).

**Status (2026-10-31).** Scope: pi0 and eta decay with Sanford-Wang or BMPT production or a particle list, on-shell
`DecayDM` and off-shell `DecayDM_Off_Shell`, and proton bremsstrahlung (`V_decay` with `proton_brem`); sphere and
cuboid detectors; `NCE_electron`.
- **pi0, `parent_sampler="stock"`** (the default) passes V0, V1 and V2 on CPU (L1) and on an A100 (G1).
- **pi0, `parent_sampler="envelope"`** is validated on G1 under a pre-declared repeat rule, after failing V2 once
  (see [below](#the-envelope-sampler)).
- **Proton brem** (stock sampler only) passes V0 on L1, and V1 and V2 on G1 at the first attempt
  ([below](#proton-bremsstrahlung)).
- **eta** passes V0, and V1 and V2 on G1 with both parent samplers ([below](#eta-decay)). The envelope sampler's
  pre-declared repeat missed one of its conditions; on review the envelope was accepted for eta anyway.
- **`particle_list`** passes V1 on G1. V2 fails, only on the parent histograms, which the port and stock draw from
  the same finite list ([below](#particle_list)).
- **Off-shell pi0 decay** (stock sampler) passes V1 and V2 on G1 at the first attempt ([below](#off-shell-decays)).
  Its V0 entries (pi0 and eta) pass.
- **On a V100** (G2, float64, stock sampler) pi0, brem, eta and off-shell pass V1 and V2 again, and `particle_list`
  gives the same V1 pass and parent-only V2 failure ([below](#v1-and-v2-on-a-v100)).
- **float32** (`BDNMC_JAX_PRECISION=float32`, for GPUs without fast float64) is not validated; its cost against
  float64 is measured [below](#float32).
- **Not validated yet:** other detectors, signals and models ([list](#not-validated-yet)).
- **Event output** (text and HDF5) is tested for format and content ([Unit tests](#unit-tests)); it does not change
  the physics.

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
- Six more base cards, added with their channels, each at the same four masses:

  | Card | Made from | Change |
  |---|---|---|
  | `mb_eta` | `mb_pi0` | `eta_decay`, `k0_sanfordwang`, `meson_per_pi0` 1/30 |
  | `he_eta_cuboid` | `he_pi0_cuboid` | `eta_decay`, `bpmt_k0`, `meson_per_pi0` 1/30 |
  | `mb_list` | `mb_pi0` | `production_distribution particle_list`, `particle_list_position true` (`phase0/make_lists.py`) |
  | `he_list_cuboid` | `he_pi0_cuboid` | as `mb_list` |
  | `mb_offshell` | `mb_pi0` | m_V = 200 MeV, above m_pi0, so pi0 -> gamma V* -> gamma chi chi at every m_chi (`phase0/make_offshell_cards.py`) |
  | `he_offshell_cuboid` | `he_pi0_cuboid` | as `mb_offshell` |

  Each has its own seed.

### Stock reference sets

| Set | Content | Generator |
|---|---|---|
| unweighted (`ref`) | one run per point: about 1e6 recoil electrons for the pi0 points, 2e7 trials for brem; event files in comprehensive mode | `phase0/make_ref_cards.py` |
| weighted (`ref_w`) | `weighted true`, 8 independent-seed replicas per point; 5e8 / 4e7 / 2e7 / 2e7 trials per point in all for `mb_pi0` / `he_pi0` / `mb_brem` / `he_brem` | `phase0/make_weighted_cards.py` |
| long weighted (`ref_wl`) | `he_pi0` only: one weighted run of 8.6e7 trials per mass, as long as the unweighted run | `phase0/make_long_weighted_cards.py` |
| repeat (`ref_r2`) | `mb_eta` and `he_eta_cuboid` at 5 MeV: the `ref` cards with seed + 1,000,000 | `phase0/make_ref_repeat_cards.py` |
| split unweighted (`ref_s`) | off-shell points: each `ref` card as N parts with their own seeds, samplesize / N events and max_trials / N (`mb_offshell` 40 x 25,000, `he_offshell_cuboid` 10 x 100,000) | `phase0/make_split_ref_cards.py` |

Stock's weighted mode is the main V1 reference. Its unweighted prediction is biased (next section). The weighted
error is the spread of the replicas.

The pi0, brem, eta and list references ran on L1 (stock `-O2`), one run per point. The eta repeat ran on C1 with
the node's gcc build, and the off-shell references on C1 with the container build
([test-environments.md](test-environments.md)): 264 single-core runs (40 x 4 + 10 x 4 parts, plus the 64 weighted
replicas) started at once and took 2-7 min each. For split references, `validation/merge_stock_parts.py` joins the
parts' electrons for V2, and `compare_v.py` combines the parts' logs for V1 (trial-weighted mean of their `se`; sigma
= `se` / sqrt(accepted events of all parts)).

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
  - Proton brem:
    - `d2N_proton_brem_to_V(z, pt2)` is called with pT in its pT^2 slot, by both the rate integral and the sampler;
      the port evaluates the same function at the same arguments.
    - The channel's DM count is `V_prod_rate x POT` (Simpson cubature, 100 x 100 steps), with no V -> chi chi
      branching ratio.
    - The V carries z times the beam momentum along the beam axis and pT transverse to it, with uniform azimuth.
    - `V_decay_gen` tests the first DM and counts the second only if the first misses the detector (a
      short-circuit `||`), so a brem event has at most one DM in the detector.
    - Brem events have no meson line: V, DM, recoil DM, electron.
  - `particle_list`: the file's E column is read but ignored (E is recomputed from the meson mass), and the rows are
    reused from the top when the list runs out. The port maps trial i to row i mod N.
  - Off-shell decays:
    - stock uses `DecayDM_Off_Shell` when the off-shell branching ratio exceeds 1.3x the on-shell one, except when
      2 m_chi <= m_V < m_meson and 2 m_chi >= 0.97 m_V;
    - it builds the V* along +z and turns it with `Rotate_z(mtheta)` then `Rotate_y(mphi)`. The first rotation
      leaves a z vector unchanged, so the V* direction is uniform in angle within the x-z plane, not isotropic.
- **`max_trials` wraps.** It is a `long long` in `Parameter` but an `int` in the event loop, so 5e9 becomes
  705,032,704 and values from 2^31 to 2^32 - 1 become negative, which means no limit. The `mb_eta` unweighted
  references stop at 705,032,704 trials (about 2.3e5 electrons per point instead of 1e6).

The port differs from stock where batching needs it:
- it unweights against the global maximum of the whole run (two passes), so its unweighted `se` is unbiased;
- the trial count is fixed (stock stops at `samplesize` events);
- the BMPT bar is fixed (a grid maximum x 1.02), so it samples the density without clipping, and it fails if any
  proposal exceeds the bar;
- the brem bar is fixed the same way. Stock's brem bar starts at d2N(zmin, ptmin) and rises when exceeded; on both
  generic brem cards that starting value is the window maximum (a unit test checks it), so stock's brem sampler
  never clips and the two sample the same law;
- an invalid brem window (z or pT bounds out of order, negative z) raises an error; stock prints one and continues
  with a zero-rate channel;
- the off-shell (s, theta) sampler has a fixed bar (a grid maximum x 1.02, with a fine grid around the V pole when it
  is inside the window); stock's starts low and rises when exceeded;
- `particle_list` refuses what stock would skip or misread: lines with fewer than four fields or an empty field, zero
  momentum when positions are read, an empty file.

## V0: function values against the stock C++

`gpu/v0/v0_dump.cpp` links the stock `-O2` library and prints about 15,400 values at `%.17g`:
- `EeTMax`, `EeTMin`, `sigma2`, `dsigmadEe`, `ThetaEe`;
- `Ldet` for a sphere, a cuboid and a rotated cuboid (6,000 rays);
- the pi0 branching ratios;
- the Sanford-Wang density and the BMPT invariant cross section;
- proton brem: `wpp` on a (z, pT^2, m_A) grid (1,323 points), `d2N` (726 points) and `V_prod_rate` for the two
  generic brem windows at m_A = 0.02, 0.1 and 0.3 GeV;
- eta: its two branching ratios, and the K0 laws used for eta (`sanfordwang_k0`, `bmpt_k0_ics`, 1,600 points each).

The off-shell entries (branching ratio and (s, theta) density for pi0 and eta, 5 regimes, 20 x 10 points each)
were added when the harness was rebuilt on 2026-10-31; every earlier value in the dump was unchanged.

`tests/test_v0.py` compares the port with them.
- **Tolerance:** (1e-12 + 64 eps x condition number) relative. The condition number covers cancellation (EeTMax at
  high energy, `F2(Emax) - F2(Emin)`, BMPT near xR = 1, Sanford-Wang at p = p_max), where clang's FMA contraction
  and XLA round differently by about eps x condition.
- **`ThetaEe` endpoint:** rows at the recoil endpoint may be NaN or <= 3e-8 in either code.
- **Brem tolerances:** `wpp` and `d2N` use the same rule, with a condition number that also covers sigma_pp(s')
  as s' cancels near z = 1. `V_prod_rate` (40,401 positive terms summed in a different order) is within 1e-10.
- **Result:** passes on L1 and on G1, with the brem and eta entries (harness rebuilt 2026-10-29); the off-shell
  entries pass on L1 (rebuilt 2026-10-31; 10 tests, none skipped). Dumps from three C1
  builds also pass: the node gcc build, Intel icpx, and the container build
  ([gpu-performance.md](gpu-performance.md#stock-bdnmc-on-a-linux-cluster-node-c1)).

## V1 and V2

Thresholds and binning were frozen in `gpu/validation/v_config.json` before the first port-against-stock comparison
(sha256 `7058b254019718dd6d98f269a6f212398abd7c57006f8b0e2e279aa971c0467a`, recorded in every result).

- **Points:** the eight pi0 points (`mb_pi0` and `he_pi0_cuboid` at 1, 5, 20, 45 MeV). The brem, eta, list and
  off-shell points each have their own copy of the criteria (sections below).
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
| envelope sampler, GPU, new seeds (pre-declared repeat) | G1 | `91c2c3e` | 1.51 / +0.22 / 0.71 | 1.00 / 0.0082 / 2.24 | **pass** |
| stock sampler, GPU, new seeds (control) | G1 | `91c2c3e` | 1.67 / +0.76 / 0.48 | 0.24 / 0.0039 / 1.93 | **pass** |

"New seeds" adds 1,000,000 to every reference card's seed (`SEED_OFFSET` in `validation/run_v_set.sh`). The stock
reference sets are the same for every row.

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

### The envelope sampler

`parent_sampler="envelope"` samples the same parent law as stock's proposal, with a piecewise bound on a (p, v) grid
instead of one bar (see `gpu/src/bdnmc_jax/parents.py`). It draws different random numbers.

**First run (`ed09f8d`): V2 failed.**
- Parent theta at `he_pi0` 45 MeV had p = 5.6e-5 (and 0.018 at 1 MeV).
- V1 passed, but with chi2 p = 0.031 (pulls +2.06, +0.07, +0.58, -0.76 for `mb_pi0` and +2.51, -0.27, +2.30,
  +0.19 for `he_pi0`).

**High-statistics check: no difference in the parent law.** `validation/compare_parent_samplers.py` on G1 drew
1.34e8 parents from each sampler:
- `he_pi0` (BMPT): p, theta (log bins down to 1e-5 rad), pT and v = theta / theta*(p) give p = 0.54, 0.40, 0.88 and
  0.14;
- `mb_pi0` (Sanford-Wang): p, theta and pT give p = 0.64, 0.53 and 0.54;
- no proposal exceeded its bound.

**Pre-declared repeat (`gpu/validation/v_rerun_envelope_20261029.json`, committed before the jobs ran).**
- Rule: the envelope is validated only if a run with new seeds
  1. passes every V1 and V2 criterion,
  2. gives p > 0.01 for the histogram that failed (`he_pi0` 45 MeV parent theta), and
  3. has no bound violation.
- No further repeats are allowed under the rule; the stock sampler runs on the same new seeds as a control.
- **Result: all three conditions hold.** The rerun passes V1 (max |pull| 1.51, chi2 p 0.71) and V2 (KS 1.00, min p
  0.0082 for `he_pi0` 45 MeV parent momentum, tails 2.24). The failed histogram has p = 0.019, and there were no
  bound violations. **The envelope is validated.** The control passes too.

The failed histogram, parent theta at `he_pi0` 45 MeV, gives p = 0.20 (stock sampler, original seeds), 5.6e-5
(envelope, first run), 0.019 (envelope, new seeds) and 0.023 (stock sampler, new seeds). The stock reference sample
is the same in every run, so p-values of one histogram are correlated across port runs. A reference that sits a
little off in this histogram, plus one unlucky port run, accounts for the first failure. Both envelope runs stay on
record.

### Proton bremsstrahlung

Thresholds, binning and variables are copied unchanged from `v_config.json` into
`gpu/validation/v_config_brem.json`, frozen before the first brem comparison (sha256
`3a8d5c3eef3c602c1400d5aa710f4c2ef3979de29ad498ea36c3275bb46cbff4`). Only the points differ.

- **Points:** `mb_brem` and `he_brem_cuboid` at 1, 5, 20, 45 MeV, against the same stock reference sets (`ref`,
  `ref_w`). The 5 MeV unweighted references are the 2e7-trial timing runs of the base cards.
- **Port runs:** 2e7 trials per point, the reference card's seed, 4 shards (`V_SET=brem validation/run_v_set.sh`;
  `compare_v.py --config validation/v_config_brem.json`).
- **V2:** the parent is the V (stock writes no meson line), so `parent_p` and `parent_theta` are the V's.

| Configuration | Env | Commit | V1: max \|pull\| / mean pull / chi2 p | V2: KS p / min p / max tail pull | Result |
|---|---|---|---|---|---|
| stock sampler, GPU | G1 | `9127b57` | 1.13 / +0.06 / 0.90 | 0.54 / 6.3e-4 / 2.70 | **pass** |

Per point, V1:

| Point (1, 5, 20, 45 MeV) | Port / stock weighted - 1 | Pull |
|---|---|---|
| `mb_brem` | +0.07, +0.01, -0.08, +0.01 % | +0.83, +0.14, -0.96, +0.28 |
| `he_brem_cuboid` | -0.00, +0.07, -0.05, -0.02 % | -0.00, +1.13, -0.64, -0.26 |

- **Secondary:** within 0.09 % of stock unweighted `se` at every point. Brem has no measurable running-pMax bias
  (+0.03 % on both cards, above), so the two references agree.
- **V2:** no histogram failed closed. 4 of the 112 p-values are below 0.01, where 1.1 are expected; the KS test of
  all 112 against uniform gives 0.54. Three of the four are at `mb_brem` 20 MeV, histograms of the same events
  and so not independent: parent momentum (6.3e-4, the smallest), theta_e (0.002) and theta_e in the lowest E_e
  slice (0.0019). The fourth is theta_e in the third E_e slice at `mb_brem` 45 MeV (0.0039). The largest tail pull is
  E_e at `he_brem` 1 MeV (+2.70). `mb_brem` 20 MeV is the point to watch in a repeat.
- **No proposal exceeded the fixed bar** at any point.
- **Scope:** the envelope sampler is not ported for brem; `setup` refuses it.

### Eta decay

`gpu/validation/v_config_eta.json` (sha256 `38c688d1a72f38fb4be6263c03b67924fa9de46ade27c3718c82f68f6c6ef9b6`):
criteria copied unchanged from `v_config.json`, frozen before the first eta comparison.
- **Points:** `mb_eta` (`k0_sanfordwang`) and `he_eta_cuboid` (`bpmt_k0`) at 1, 5, 20, 45 MeV. The parent in V2 is
  the eta.
- **Port runs:** 5e8 trials per `mb_eta` point, 2e8 per `he_eta_cuboid` point (`V_SET=eta`).
- **References:** the `mb_eta` unweighted references hold about 2.3e5 electrons per point, because stock's
  `max_trials` wraps ([above](#stock-behaviours-that-matter-for-the-comparison)); the others hold about 1e6.
- **Scope:** `eta_decay` with any other parent distribution (`bmpt`, `pi0_sanfordwang`, `proton_brem`) is refused.

| Configuration | Env | Commit | V1: max \|pull\| / mean pull / chi2 p | V2: KS p / min p / max tail pull | Result |
|---|---|---|---|---|---|
| stock sampler, GPU | G1 | `69468ea` | 1.89 / -0.30 / 0.49 | 0.29 / 0.0074 / 2.91 | **pass** |
| envelope sampler, GPU | G1 | `69468ea` | 2.48 / -0.13 / 0.12 | 0.24 / 0.0025 / 2.32 | **pass** |
| envelope sampler, GPU, new seeds (pre-declared repeat) | G1 | `852a259` | 1.05 / -0.21 / 0.90 | 0.70 / 0.0053 / 2.18 | **pass**; the rule's condition 2 not met (below) |
| stock sampler, GPU, new seeds (control) | G1 | `852a259` | 2.61 / -0.31 / 0.29 | 0.85 / 0.0040 / **3.09** | **fail** (tail pull > 3); a reference fluctuation (below) |

Per point, stock sampler, first seeds, V1:

| Point (1, 5, 20, 45 MeV) | Port / stock weighted - 1 | Pull |
|---|---|---|
| `mb_eta` | -0.14, -0.17, -0.46, +0.31 % | -0.56, -0.61, -1.89, +0.91 |
| `he_eta_cuboid` | +0.07, -0.10, -0.03, +0.02 % | +0.94, -1.15, -0.26, +0.18 |

Secondary: within 0.8 % of stock unweighted `se` at every point.

**The envelope repeat (`gpu/validation/v_rerun_envelope_eta_20261029.json`, committed before the jobs ran).** The
first envelope run passed, but 4 of its 112 p-values were below 0.01 (1.1 expected), all `theta_e` histograms at
`mb_eta` 5 MeV. The rule: the envelope is validated for eta only if a run with new seeds
1. passes every V1 and V2 criterion,
2. gives p > 0.01 for each of those four histograms,
3. has no bound violation, and
4. a high-statistics parent check on both K0 laws gives p > 0.001 for every histogram, with no bound violation.

Result: conditions 1, 3 and 4 hold. The parent check (`validation/compare_parent_samplers.py`, 1.34e8 parents per
sampler) gives p = 0.83, 0.89, 0.048 for p, theta and pT on `k0_sanfordwang`, and 0.26, 0.61, 0.11, 0.65 for p,
theta, pT and v on `bpmt_k0`. Condition 2 fails: `theta_e` in the lowest E_e slice at `mb_eta` 5 MeV has p = 0.0053.
Under the rule the envelope is therefore **not validated for eta**. **On review, the envelope was accepted for eta
anyway (2026-10-30)**, overriding the rule; this record keeps both.

**Second stock reference (`gpu/validation/eta_second_reference_20261029.json`, declared before it ran).** The
control's failure (`theta_e` tail pull -3.09 at `he_eta_cuboid` 5 MeV) and the low `theta_e|E_e_q0` p-values at
`mb_eta` 5 MeV recur across independent port runs against the same stock sample, so a second stock reference was
run at those two points (`ref_r2`, new seeds, on C1).
- Test A, first reference against second (`validation/compare_stock_refs.py`): `he_eta_cuboid` 5 MeV `theta_e`
  tail pull -2.10, so the first reference has more tail than the second.
- Test B, the four port runs against the second reference (`compare_v.py --stock-dir validation/out/stock_r2`): the
  `he_eta_cuboid` `theta_e` tail pulls are -0.26, +0.21, +0.34, -0.70 (mean -0.10), and `mb_eta` `theta_e|E_e_q0`
  has p = 0.21, 0.68, 0.33, 0.32.
- Verdict under the declared rule: **"reference"**. The control's V2 failure is a fluctuation of the first
  reference, and the stock-sampler eta configuration stays validated on its first run.
- The two stock references disagree at 6 of their 28 histograms at p < 0.05 (1.4 expected). They also differ in
  build (L1 against C1), which may contribute; not checked further.

### particle_list

`gpu/validation/v_config_list.json` (sha256 `0f90e4a2...`): criteria copied unchanged from `v_config.json`, frozen
before the first list comparison.
- **Lists:** `phase0/make_lists.py` writes two synthetic lists of 1e6 pi0 each: Sanford-Wang parents at 8.9 GeV for
  `mb_list`, BMPT parents at 120 GeV on carbon for `he_list_cuboid` (drawn with the port's envelope sampler, fixed
  seed). Each row has a synthetic decay vertex (x, y Gaussian with sigma 0.2 m, z uniform in [0, 50] m, t = z / c),
  so `particle_list_position` is exercised. The lists are not tracked; the script regenerates them and checks them
  against the sha256 in `lists/MANIFEST.json`.
- **Port runs:** 5e8 trials per `mb_list` point, 4e7 per `he_list_cuboid` point (`V_SET=list`).
- **Parser:** `tests/data/plist_formats.dat` covers the separators and column counts stock accepts; the port's
  parents and positions match stock's own echo of that file (`v0/plist_echo.sh`).

| Configuration | Env | Commit | V1: max \|pull\| / mean pull / chi2 p | V2: KS p / min p / max tail pull | Result |
|---|---|---|---|---|---|
| GPU | G1 | `eeb7003` | 1.52 / +0.40 / 0.78 | **0.0054 / 4.8e-9** / 2.26 | V1 pass; **V2 fails** on two parent histograms |

- **V1:** pulls +0.81, -0.13, -0.48, -0.46 (`mb_list`) and +0.81, +0.70, +1.52, +0.43 (`he_list_cuboid`); the port
  is within 0.18 % of stock weighted `se` at every point.
- **V2:** only two of 112 histograms have p < 0.01, both at `mb_list` 45 MeV: `parent_p` (3.3e-8) and `parent_theta`
  (4.8e-9). They pull the KS test below 0.01. Every electron, DM and vertex histogram passes.
- **Why the parent histograms fail:** both codes draw their parents from the same rows of the same list, so the two
  samples share their inputs and the two-sample test does not apply to them. Every row is used many times (500 times
  in the port's 5e8 trials, 1,200-2,000 times in stock's runs), so both samples are clumped on the same values.
  Stock also writes the parent at 10 significant digits, which piles repeated values onto the equal-population bin
  edges (`validation/diagnose_ties.py`). An amendment that would drop these two histograms for list points was
  drafted but not declared, so this is recorded as a V2 fail confined to the shared inputs.

### Off-shell decays

`gpu/validation/v_config_offshell.json` (sha256 `b729ebda...` after its amendment): criteria copied unchanged from
`v_config.json`, frozen before the first off-shell comparison and amended, still before it, for the reference runs
(split parts, C1 container build).
- **Points:** `mb_offshell` and `he_offshell_cuboid` at 1, 5, 20, 45 MeV (m_V = 200 MeV); stock sampler only.
- **Port runs:** 5e8 trials per `mb_offshell` point, 4e7 per `he_offshell_cuboid` point (`V_SET=offshell`).
- **References:** split unweighted references (`ref_s`) and 8 weighted replicas per point, all on C1
  ([above](#stock-reference-sets)); 1.00-1.10e6 electrons per point.

| Configuration | Env | Commit | V1: max \|pull\| / mean pull / chi2 p | V2: KS p / min p / max tail pull | Result |
|---|---|---|---|---|---|
| stock sampler, GPU | G1 | `c4ce8d5` | 1.94 / -0.18 / 0.51 | 0.91 / 0.016 / 2.04 | **pass** |

Per point, V1:

| Point (1, 5, 20, 45 MeV) | Port / stock weighted - 1 | Pull |
|---|---|---|
| `mb_offshell` | +0.04, +0.00, -0.30, +0.04 % | +0.20, +0.00, -1.94, +0.19 |
| `he_offshell_cuboid` | -0.05, -0.26, +0.04, +0.22 % | -0.22, -1.22, +0.21, +1.35 |

- **V2:** no histogram failed closed; 10 of 112 p-values are below 0.05 (5.6 expected), none below 0.01.
- **Secondary:** the port is 0.7-1.4 % below stock unweighted `se`, outside max(1 %, 2 sigma) at three
  `he_offshell` points. This is a larger running-pMax excess than the single-run references show (0.03-0.62 % at the
  pi0 points; 0.2-0.3 % at three `he_list_cuboid` points, 1.27 % at the fourth). Each split part starts again from
  a 1000-DM burn-in, so a larger share of the sample is drawn under a low pMax; the amendment noted the extra
  burn-ins before the comparison. The secondary comparison is not a pass criterion.
- **Unit tests** check the off-shell mode choice and `Vnum` against the stock logs, the sampler against its density,
  and the decay kinematics, including stock's x-z-plane V* direction.

### V1 and V2 on a V100

The five V sets rerun unchanged (same points, trials and seeds, frozen configs; commit `51d982f`, float64, stock
sampler) on a V100-SXM2-32GB (G2), one GPU per set, and compared with the same stock references:

| Set | V1: max \|pull\|, mean pull, chi2 p | V2: KS p, smallest p, max \|tail pull\| | Verdict |
|---|---|---|---|
| pi0 | 2.15, +0.80, 0.47 | 0.50, 0.0028, 2.51 | pass |
| brem | 1.13, +0.06, 0.90 | 0.54, 0.00063, 2.70 | pass |
| eta | 1.89, -0.30, 0.49 | 0.29, 0.0074, 2.91 | pass |
| off-shell | 1.94, -0.18, 0.51 | 0.91, 0.016, 2.04 | pass |
| `particle_list` | 1.52, +0.40, 0.78 | 0.01, 4.8e-9, 2.26 | V1 pass; V2 fails on the parent histograms only, as on G1 |

The V100 rounds differently from the A100, so its events differ; these are independent repeats of the G1 results.

## float32

`BDNMC_JAX_PRECISION=float32`, set before `bdnmc_jax` is first imported, runs the device code in float32 (JAX's
64-bit mode off); host-side setup, pass 2 and the sums stay in float64. It is meant for endpoints without fast
float64, and it is not a validated configuration. Its cost is measured three ways:
`validation/precision_v0.py` (the V0 function checks as a per-function error report), the five V sets against
stock with the frozen configs, and `validation/compare_precision.py` (float32 against float64 run with the same
points and seeds; the two precisions draw different random bits, so the samples are independent).

Plain float32 of stock's expressions fails in four places, each fixed for float32 only (float64 keeps stock's
expressions):
- **`acos` of a cosine just above 1.** pz/|p| of a DM along the beam axis (0.99999997 in float64) rounds past 1 on
  the GPU, and `theta_ee`'s argument does so near the recoil endpoint. Both stopped most `mb` runs with a
  non-finite electron; both are clipped to the endpoint.
- **The Lorentz boost.** Stock goes gamma -> beta = sqrt(1 - 1/gamma^2) -> 1/sqrt(1 - beta^2), which loses about
  2 eps gamma^2 of gamma (several % at gamma ~ 700). float32 boosts from the parent's four-momentum (gamma = E/m,
  gamma beta = |p|/m). Before this, `E_chi` failed V2 against float64 at every 120 GeV point.
- **`sigma2`**, the scatter cross section, a difference of two large F2 values: written as the difference of its
  1/x and log terms (worst float32 error against the stock dump 110 % -> 1e-4).
- **The sphere's path length:** the discriminant from the closest approach instead of B^2 - 4AC (worst error
  0.29 m -> 9e-5 m; no hit/miss flips in 2,000 rays).

Results with these fixes are being collected.

## Reproducibility

- **Per seed:** results do not depend on the batch size (blocks per device call), the candidate cap or the split
  into shards; tests check this bit for bit on CPU. Each 2^16-trial block's random numbers depend only on (seed,
  channel, block).
- **GPU against CPU:** with the stock sampler, the GPU runs give the same number of detector hits and accepted
  events at all eight pi0 points as the CPU run, and `se` within 6e-14 relative. (Brem, eta, `particle_list` and
  off-shell have run on G1 only.)
  - Per candidate (64 blocks at two points), the candidate sets are identical and values agree within 3.6e-6
    relative (`prob`) and 2.8e-6 (`recoil_dm`, a difference of nearly equal momenta).
  - The two backends round transcendental functions differently, so this is not bit-identical.
- **Code changes:** making masses runtime inputs (`7c3fbe0`) moved values by up to 1.6e-8 relative (`prob` near the
  `sigma2` threshold) with unchanged candidate sets (`validation/check_shard_repro.py`).

## Unit tests

`python -m pytest` in `gpu/` (about 2 min on G1, where compilation dominates):
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
- **Output:**
  - number formatting matches stock's 10-digit lines token for token;
  - event structure (`Run`, `event N`, 5 or 8 particle lines, `endevent N`, ` Summary`);
  - the text reads back through the same parser used on stock event files, giving the accepted rows;
  - each DM's interaction point lies between its detector entry and exit;
  - weighted output writes one DM per event;
  - HDF5 weights sum to the predicted signal in both modes.
- **Proton brem:**
  - `Vnum` against the eight stock brem logs (5e-6);
  - the Simpson cubature is exact for cubics;
  - the fixed bar covers the density, and stock's starting bar is the window maximum on both cards;
  - V kinematics, and at most one DM in the detector per event;
  - one compile across masses;
  - text events start with the V line (no meson).

- **Eta:**
  - branching ratio and `Vnum` against the stock logs;
  - the K0 laws differ from the pi0 ones and keep stock's choices;
  - eta kinematics, text output with the eta as parent;
  - other parent distributions are refused.
- **`particle_list`:**
  - the parser against stock's echo of a test list, with and without positions;
  - the refusals; rows cycling across blocks;
  - a list of a sampler's parents reproduces that sampler's run bit for bit;
  - positions move the decay vertex; an eta list uses the eta mass.
- **Off-shell:**
  - mode choice and `Vnum` against the stock logs, at several (m_V, m_chi) regimes;
  - the (s, theta) sampler follows its density, and the bar covers a V pole inside the window;
  - decay kinematics in the meson frame (with stock's x-z-plane V* direction) and momentum conservation in the lab;
  - on-shell cards are unchanged.
- **Split references:** the part cards differ from their reference card only in seed, samplesize and max_trials;
  their seeds are unique; the merge and the combination of the parts' logs.

106 tests in all: on L1, 105 pass and the off-shell V0 test is skipped (harness not rebuilt) (2 min). On G1, in the
off-shell V-set job at `c4ce8d5`, 87 pass and 19 skip: those read stock run logs under `phase0/out`, which were not
copied into that job's code tree.

## Not validated yet

- the envelope sampler with off-shell decays or with `particle_list` (not run), and for `proton_brem` (not ported);
- off-shell eta decay in V1/V2 (only pi0 has reference points);
- V2 of the parent histograms for `particle_list` (see [above](#particle_list));
- cylinder detectors, timing cuts, `dist_mod`, signal channels other than `NCE_electron`, and models other than
  `Dark_Photon`.
