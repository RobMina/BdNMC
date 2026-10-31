# GPU port: performance

Throughput of stock BdNMC and of the JAX port, measured on the environments in
[test-environments.md](test-environments.md). Physics validation, which every configuration quoted as a speed-up
must pass, is in [gpu-validation.md](gpu-validation.md).

**Target (P1):** at least 100x one core of stock `-O2`, in accepted events per second, on each of the four 5 MeV
generic cards. Measured on a full A100 in float64, with binary output included and compile time reported
separately.
**Status (2026-10-31, commit `85db916`, HDF5 written):** met on three of the four cards and borderline on the fourth:

| Card | stock sampler | envelope sampler |
|---|---|---|
| `mb_pi0` | 158x | 250x |
| `mb_brem` | **90-116x** (median of interleaved repeats 97x, 90x, 104x on three nodes) | (not ported) |
| `he_pi0_cuboid` | 120x | 873x |
| `he_brem_cuboid` | 293x | (not ported) |

`mb_brem` writes 5.9 M rows (1.4 GB) per 6.7e7 trials, and its rate depends on how busy the shared node's host is
([below](#accepted-rows-gathered-on-the-device-commits-3f3993a-to-85db916)).

Also not yet done for P1: the stock baseline is a laptop core rather than a core of the GPU node, and the MIG and
whole-node figures are not measured (the whole-node job is submitted).

## Stock BdNMC (L1, one core)

20 M trials per run, CPU time (user + system) from `time -l`:

| Card (m_chi = 5 MeV) | `-O0` CPU | `-O2` CPU | `-O2` speed-up | Trials per CPU-second, `-O0` / `-O2` | DM reaching the detector per trial | Accepted per trial | Accepted events per CPU-second, `-O0` / `-O2` |
|---|---|---|---|---|---|---|---|
| `mb_pi0` | 68.9 s | 23.0 s | 3.0x | 290 k / 868 k | 2.1e-3 | 7.7e-4 | 223 / 667 |
| `mb_brem` | 135.9 s | 43.7 s | 3.1x | 147 k / 457 k | 0.143 | 8.9e-2 | 13.0 k / 40.5 k |
| `he_pi0_cuboid` | 699.1 s | 131.0 s | 5.3x | 29 k / 153 k | 2.6e-2 | 1.2e-2 | 334 / 1,784 |
| `he_brem_cuboid` | 1,340 s | 193.3 s | 6.9x | 15 k / 103 k | 0.161 | 8.4e-2 | 1.26 k / 8.71 k |

- `build/makefile` sets no optimisation flag, so every stock build is `-O0` unless the user adds one. `-O2` gives
  the same events at the same seed (printed values within 2.3e-9 relative).
- Brem runs have high system time (15-25 s): stock keeps every accepted event in memory until the end of the run
  (about 1 GB resident for 1.7 M events).
- Speed-ups below are quoted against both builds: `-O0` is how BdNMC is built as shipped.

## Stock BdNMC on a Linux cluster node (C1)

One core of C1 runs stock 3-8x slower than one core of L1, in CPU time per trial:

| Card | Trials | L1 `-O2` | C1 `-O2` | C1 / L1 |
|---|---|---|---|---|
| `mb_pi0_m005` (Sanford-Wang) | 2e7 | 1.15 µs (23.0 s) | 9.21 µs (184.2 s) | 8.0x |
| `mb_eta_m005` (K0 Sanford-Wang) | 7.05e8 | 1.10 µs | 4.37 µs | 4.0x |
| `he_eta_cuboid_m005` (BMPT kaon) | 1.7e8 | 7.45 µs | 22.3 µs | 3.0x |

All runs were CPU-bound (99 % of a core). That is more than the hardware difference between the two cores accounts for.

**Where the time goes on C1** (`valgrind --tool=callgrind`, `mb_pi0_m005`):
- `pow` from glibc 2.28's `libm` takes 59 % of all instructions, about 230 instructions per call. `libm` as a whole
  takes about 70 %.
- The calls come from `sanfordwang::swpip` and `swpim`. The Sanford-Wang density calls `pow` about 8 times per
  evaluation, and the sampler evaluates it about 14 times per accepted parent, so a trial makes about 115 `pow` calls.
- The exponents are fit parameters held at run time, so the compiler cannot replace these calls with
  multiplications, whatever the optimisation flags. `libm` is precompiled and already picks its FMA/AVX2 code paths on
  this CPU, so stock's compiler flags do not change its cost either.

**Three stock builds on one C1 node**, one after another, CPU time per trial (`mb_pi0_m005` 2e7 trials,
`he_pi0_cuboid_m005` 2e6 trials, `output_mode summary`):

| Build | `mb_pi0` | `he_pi0` |
|---|---|---|
| gcc 14.2.0, node glibc 2.28 | 8.43 µs | 36.0 µs |
| Intel icpx `-fp-model=precise` (`libimf`) | 7.87 µs | 45.3 µs |
| gcc 14.2.0 in a Debian bookworm container, glibc 2.36 | 4.97 µs | 25.6 µs |
| L1 `-O2`, for comparison | 1.15 µs | 6.55 µs |

- All three give the same `se` on both cards. Their V0 dumps pass `tests/test_v0.py`; against the gcc build, the Intel
  dump differs by up to 4.1e-14 relative and the container dump by 1.8e-15 (`v0/compare_dumps.py`).
- The container build is the fastest, 1.7x the node build, from glibc 2.36's faster `pow`, but still about 4x slower
  per trial than L1. Stock reference runs on C1 use it.

**The four P1 cards with the container build**, one after another on one core (two jobs on the same node type,
agreeing to 1 %; `output_mode summary`), CPU time per trial: `mb_pi0` 5.02 µs, `mb_brem` 2.44 µs, `he_pi0_cuboid`
25.9 µs, `he_brem_cuboid` 22.9 µs. Against L1 `-O2` that is 4.4x, 1.1x, 4.0x and 2.4x slower. C1 is nearly as fast
as L1 on `mb_brem`; the gap is largest on `mb_pi0`, whose time goes to `pow` (above). The brem cards have not been
profiled on C1.

**Not tried yet:**
- `-ffast-math`. It would at most skip `libm`'s error-handling wrappers. Its other effects on stock's results are not
  checked: it allows reordering floating-point arithmetic, assumes no NaN or infinity (stock can produce a NaN in
  `ThetaEe`) and flushes subnormal numbers to zero.
- the same profile on L1, where callgrind does not run, so the per-call cost of `pow` there is not measured.

The port does not call `libm` (its maths runs in XLA), so none of this affects the port's throughput. Every speed-up on
this page is against L1, the faster stock baseline.

## JAX port

Measured with `gpu/validation/bench.py` on the 5 MeV reference cards (`phase0/cards/ref/*_m005.dat`).
- **Steady state:** the first call (compilation and candidate-cap growth) is reported separately.
- **End to end:** pass 1 (all trials, with host-side checks) plus pass 2 (two-pass unweighting); no output file is
  written.
- **Device only:** the device kernel alone, without host transfer or checks.
- **Batch:** 2^16-trial blocks per device call.

### Stock parent sampler (validated configuration), commit `7c3fbe0`

| Card | Env | Batch | Trials/s | Accepted/s | Device only, trials/s | First call |
|---|---|---|---|---|---|---|
| `mb_pi0` | L1 (one core) | 1 | 1.03 M | 0.8 k | 1.07 M | 1.3 s |
| `mb_pi0` | G1 | 1 / 4 / 16 / 64 | 18.5 / 50.8 / 89.5 / **112.7 M** | 13.8 / 37.9 / 66.7 / **83.9 k** | 15.2 / 47.9 / 100.0 / 147.0 M | 3.5-5.0 s |
| `he_pi0_cuboid` | L1 (one core) | 1 | 0.05 M | 0.7 k | 0.07 M | 6.7 s |
| `he_pi0_cuboid` | G1 | 1 / 4 / 16 / 64 | 1.7 / 5.3 / 10.4 / **12.9 M** | 20.8 / 63.0 / 123.7 / **154.1 k** | 1.7 / 5.7 / 12.0 / 15.5 M | 8.7-11.2 s |

Speed-up of G1 at batch 64 over one L1 core of stock, in accepted events per second:

| Card | vs `-O2` | vs `-O0` |
|---|---|---|
| `mb_pi0` | **126x** | 376x |
| `he_pi0_cuboid` | **86x** | 461x |

### With HDF5 output (P1 configuration), commit `efc2607`

`bench.py --write hdf5`: the accepted DM written to an HDF5 file on the cluster's shared scratch file system (one
row per DM, about 260 bytes), the write timed and included; 1.34e8 trials per point.

| Card | Sampler | Batch | Trials/s | Accepted/s | HDF5 write | vs stock `-O2` / `-O0` |
|---|---|---|---|---|---|---|
| `mb_pi0` | stock | 64 / 256 | 97.2 / 112.4 M | 72.5 / **83.8 k** | 0.05-0.13 s, 26 MB | **126x** / 376x |
| `he_pi0_cuboid` | stock | 64 / 256 | 14.1 / 16.3 M | 167.4 / **193.7 k** | 0.61-0.64 s, 422 MB | **109x** / 580x |
| `mb_pi0` | envelope | 64 / 256 | 141.0 / 167.3 M | 105.8 / **125.5 k** | 0.05-0.09 s, 27 MB | **188x** / 563x |
| `he_pi0_cuboid` | envelope | 64 / 256 | 65.9 / 74.9 M | 784.8 / **892.0 k** | 0.61-0.64 s, 422 MB | **500x** / 2,670x |

- **First calls:** 4.7-5.7 s (`mb_pi0`) and 9.7-12.5 s (`he_pi0`), excluded.
- **Cost of the write:** on `he_pi0` with the envelope, writing is about a third of the end-to-end time (device
  only: 257 M trials/s).
- **Text output** (`--write text`, stock's format, `he_pi0` envelope, 3.4e7 trials) takes 18.3 s for 0.4 M events
  (272 MB): 21 k accepted/s end to end. That is a 12x speed-up over stock `-O2`, so text is for small samples and
  compatibility, not throughput.
- **`he_pi0` stock-sampler variation:** the stock sampler on `he_pi0` reached 167 k accepted/s at batch 64 in this
  job against 154 k in the earlier one (`7c3fbe0`, no output). The generation code is unchanged, so this is
  run-to-run variation (a different node).

### Envelope parent sampler without output, commit `ed09f8d`

| Card | Env | Batch | Trials/s | Accepted/s | Device only, trials/s | vs stock `-O2` / `-O0` |
|---|---|---|---|---|---|---|
| `mb_pi0` | L1 (one core) | 1 | 4.9 M (profile) | | | |
| `mb_pi0` | G1 | 16 / 64 / 256 | 118.2 / 158.2 / **175.3 M** | 88.6 / 118.6 / **131.5 k** | 143.8 / 240.4 / 275.3 M | 197x / 590x |
| `he_pi0_cuboid` | L1 (one core) | 1 | 4.5 M (profile) | | | |
| `he_pi0_cuboid` | G1 | 16 / 64 / 256 | 74.1 / 89.6 / **111.8 M** | 883 / 1,067 / **1,332 k** | 125.2 / 213.6 / 251.4 M | 747x / 3,990x |

The envelope configuration has since passed V1 and V2 (pre-declared repeat, [gpu-validation.md](gpu-validation.md#the-envelope-sampler)).
The P1 figures are the ones with output in the table above.

### Proton brem with HDF5 output, commit `2c9e3c7`

G1, stock sampler, `bench.py --write hdf5 --cap 16384` (a brem block has about 9,400-10,500 candidates), 6.7e7
trials per card:

| Card | Batch | Trials/s | Accepted/s | Generate / HDF5 write | Device only, trials/s | vs stock `-O2` / `-O0` |
|---|---|---|---|---|---|---|
| `mb_brem` | 64 / 256 | 17.9 / 19.8 M | 1,580 / **1,749 k** | 1.41 s / 1.98 s, 1.57 GB | 208 / 229 M | **43x** / 135x |
| `he_brem_cuboid` | 64 / 256 | 13.7 / 14.5 M | 1,155 / **1,220 k** | 2.59 s / 2.04 s, 1.49 GB | 48 / 54 M | **140x** / 968x |

- **First calls:** 4.4-8.4 s, excluded.
- **`mb_brem` misses P1** (100x is 4.05 M accepted/s). Stock accepts 8.9 % of `mb_brem` trials, so the file is large
  (5.9 M rows, 265 bytes each), and writing it to the cluster's shared scratch file system at about 0.8 GB/s takes
  58 % of the run. Generation alone runs at 4.2 M accepted/s (104x). The host side between device and file
  (transfer, checks) is the next cost: device only is 229 M trials/s against 47.5 M for generation end to end.
- **L1, one core** (commit `9127b57`, no output): `he_brem_cuboid` at 0.32 M trials/s, 27.3 k accepted/s, 3.1x one
  core of stock `-O2`.

### Accepted rows gathered on the device, commits `3f3993a` to `85db916`

The host side above (copying every candidate's 4-vectors off the device, then the write) was the `mb_brem` limit.
`pipeline.events` now keeps the candidates' large fields on the device until pass 2 has chosen the accepted rows,
gathers those there in output order, and copies only them; with `stream=True` the HDF5 writer takes each batch as
its copy arrives, so the copies overlap the write and the rows are never joined on the host. The HDF5 file is
identical to the one from `finish(candidates(...))` (tests). A brem event's parent is its V, so `parent` is written
as an HDF5 hard link to `mediator`.

`mb_brem` on G1, batch 256, 6.7e7 trials, HDF5 to the shared scratch file system. The A100 nodes are shared with
other jobs, and this host-heavy run varies by about ±30 % with the node's load, so code versions are compared in
one job on one node, interleaved, five repeats each (medians):

| Node | before streaming (`476abbe`) | streaming (`50017f0`) | vs stock `-O2` |
|---|---|---|---|
| A | 3.24 M accepted/s | 3.93 M/s | 97x |
| A (busier) | 2.93 M/s | 3.66 M/s | 90x |
| B | 3.21 M/s | 4.22 M/s | 104x |

The remaining host time is NumPy in pass 2 and the output ordering (the boolean selection of the pass-2 fields,
the sort key and the event numbering): about 1.4 s of a run whose device time is 0.28 s
(`validation/profile_host.py --events --stream`). Moving them to the device is the next step.

### All four cards on several GPU types, commit `85db916`

`bench.py --write hdf5`, stock sampler (and envelope for pi0), best of batch 64 and 256, one job per GPU type and
card pair (G1 and G2, a single run each, so node load adds the variation above). Accepted events per second, and the
speed-up over one L1 core of stock `-O2`:

| GPU | `mb_pi0` | `mb_pi0` envelope | `mb_brem` | `he_pi0_cuboid` | `he_pi0` envelope | `he_brem_cuboid` |
|---|---|---|---|---|---|---|
| A100-SXM4-80GB | 105 k (158x) | 167 k (250x) | 4.23 M (104x) | 215 k (120x) | 1.56 M (873x) | 2.55 M (293x) |
| A100-SXM4-40GB | 102 k (153x) | 146 k (218x) | 3.86 M (95x) | 195 k (109x) | 1.42 M (797x) | 2.27 M (261x) |
| V100-SXM2-32GB | 74 k (111x) | 108 k (163x) | 2.79 M (69x) | 120 k (67x) | 833 k (467x) | 1.43 M (164x) |
| RTX 3090 | 18 k (27x) | 63 k (94x) | 2.25 M (56x) | 27 k (15x) | 622 k (349x) | 543 k (62x) |
| RTX A6000 | 18 k (26x) | 60 k (90x) | 1.62 M (40x) | 25 k (14x) | 568 k (318x) | 524 k (60x) |
| A40 | 16 k (24x) | 54 k (82x) | 2.39 M (59x) | 24 k (13x) | 622 k (349x) | 507 k (58x) |
| RTX 2080 Ti (batch 64) | 13 k (20x) | 47 k (70x) | 1.77 M (44x) | 20 k (11x) | 467 k (262x) | 424 k (49x) |

- The port runs in float64, and the A40, A6000 and RTX cards run FP64 at 1/32 of FP32: their device-only rate is
  7-9x below the A100 on `he_pi0` and about 8x on `he_brem`. They hold up only where the host side, or the cheap
  envelope sampler, sets the pace.
- The V100 (full-rate FP64) runs at 55-75 % of the A100.
- The RTX 2080 Ti (11 GB) runs out of device memory at batch 256.
- A float32 mode exists for GPUs without fast FP64 (`BDNMC_JAX_PRECISION=float32`, see
  [gpu-validation.md](gpu-validation.md#float32)); none of these figures use it.

## Where the time goes

`gpu/validation/profile_port.py` times the parent sampler alone against the whole pass-1 block:

| Card | Sampler | Env, batch | Block | Parent sampler share | Proposals per accepted parent |
|---|---|---|---|---|---|
| `he_pi0_cuboid` (BMPT) | stock | L1, 1 | 691 ms | 99 % | 54 (plus 13-15 theta-shrink iterations per round) |
| `mb_pi0` (Sanford-Wang) | stock | L1, 1 | 37 ms | 95 % | 14 |
| `he_pi0_cuboid` | envelope | L1, 1 | 14.5 ms | 86 % | 1.6 |
| `mb_pi0` | envelope | L1, 1 | 13.5 ms | 86 % | 1.4 |
| `he_pi0_cuboid` | envelope | G1, 16 / 64 / 256 | 6.6 / 16.8 / 53.1 ms | about 100 / 65 / 53 % | 1.6 |
| `mb_pi0` | envelope | G1, 16 / 64 / 256 | 5.4 / 16.0 / 55.2 ms | about 100 / 62 / 50 % | 1.4 |

At small batches the two timings come from separately compiled programs, so a share near or above 100 % means
"almost all".

Proton brem (stock sampler): the parent sampler is 93-97 % of the block on L1 (batch 1), with 8 proposals per
accepted V on `mb_brem` and 97 on `he_brem_cuboid`, whose density peaks sharply inside a wide (z, pT) window. On
G1 at batch 256 it is 80 % (`mb_brem`, 438 M trials/s per block) and 96 % (`he_brem`, 63 M trials/s). An envelope
for brem would cut the `he_brem` cost as it did for pi0; on `mb_brem` the output, not the sampler, is the limit.

With the stock samplers the parent rejection loop is the whole cost. With the envelope it falls to about half on
the GPU at large batches, and the host side (transfer of candidates, checks, concatenation) becomes the next cost:
18-36 % of the end-to-end time on `mb_pi0` and 41-58 % on `he_pi0` at batches 16-256 (device-only against
end-to-end), against 17-23 % with the stock sampler at batch 64.

## What is not measured yet

- A stock baseline on a core of the GPU node itself, and the whole-node comparison (an exclusive G1 job is
  submitted: stock on one core and on every core of the node, the port on one GPU and on all eight).
- A MIG slice.

## P2: a mass grid in one GPU-hour

`validation/run_grid.py`, commit `ff1acf2`, one A100-SXM4-80GB (G1), float64, stock sampler: the ten generic base
cards (mb and he; pi0, eta, proton brem, `particle_list`, off-shell pi0) x 20 masses (m_chi 1-45 MeV, log-spaced; only
`dark_matter_mass` changes) x at least 80,000 accepted events each, HDF5 written, in one process timed from before
the JAX import:

| Samples | End to end | XLA compilation | Fewest events | Per sample after the first mass |
|---|---|---|---|---|
| 200 | **321 s (0.089 GPU-hours; P2 asks for <= 1)** | 218 compilations, 103 s | 82,499 | 0.75-2.3 s |

- Each base card compiles on its first mass. The 5 later compilations each come where that card's candidate cap
  doubled (a larger fixed-shape buffer); a change of mass alone never recompiled.
- Trials per sample are sized from the previous mass's acceptance; one `mb_eta` sample came out short and was rerun
  (included in the time).

## Reproducing

```bash
cd gpu
python validation/bench.py phase0/cards/ref/mb_pi0_m005.dat 134217728 --batch 16 64 256 [--parent-sampler envelope] [--write hdf5 --out-dir DIR]
python validation/profile_port.py phase0/cards/ref/he_pi0_cuboid_m005.dat --batch 16 64 256 [--parent-sampler envelope]
python validation/profile_host.py phase0/cards/mb_brem_m005.dat 67108864 --events --stream --out-dirs DIR
python validation/run_grid.py phase0/cards/*_m005.dat --out-dir DIR --n-masses 20 --events 80000 --json grid.json
```

On CPU, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to keep one process near one core.
