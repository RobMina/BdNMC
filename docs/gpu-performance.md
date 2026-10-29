# GPU port: performance

Throughput of stock BdNMC and of the JAX port, measured on the environments in
[test-environments.md](test-environments.md). Physics validation, which every configuration quoted as a speed-up
must pass, is in [gpu-validation.md](gpu-validation.md).

**Target (P1):** at least 100x one core of stock `-O2`, in accepted events per second, on each of the four 5 MeV
generic cards. Measured on a full A100 in float64, with binary output included and compile time reported
separately.
**Status (2026-10-29):** met on three of the four cards. On the two pi0 cards with HDF5 output included, for both
validated parent samplers:
- stock sampler: 126x on `mb_pi0`, 109x on `he_pi0`;
- envelope sampler: 188x on `mb_pi0`, 500x on `he_pi0`.

On the brem cards (stock sampler, the only one ported for brem): met on `he_brem_cuboid` (140x), **not met on
`mb_brem` (43x)**, where writing the HDF5 file takes 58 % of the run ([below](#proton-brem-with-hdf5-output-commit-2c9e3c7)).

Also not yet done for P1: the stock baseline is a laptop core rather than a core of the GPU node, and the MIG and
whole-node figures are not measured.

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
- **Ways to close it, none tried yet:** write to node-local disk; overlap the write with generation; drop columns
  that can be recomputed (`recoil_dm` is the DM minus the electron).
- **L1, one core** (commit `9127b57`, no output): `he_brem_cuboid` at 0.32 M trials/s, 27.3 k accepted/s, 3.1x one
  core of stock `-O2`.

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

- A stock baseline on a core of the GPU node itself, and the whole-node comparison.
- A MIG slice.
- P2 (a 20-mass x all-channel grid in one GPU-hour): masses are already runtime inputs, so a mass scan reuses the
  compiled kernel, but the grid itself has not been run.

## Reproducing

```bash
cd gpu
python validation/bench.py phase0/cards/ref/mb_pi0_m005.dat 134217728 --batch 16 64 256 [--parent-sampler envelope] [--write hdf5 --out-dir DIR]
python validation/profile_port.py phase0/cards/ref/he_pi0_cuboid_m005.dat --batch 16 64 256 [--parent-sampler envelope]
```

On CPU, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to keep one process near one core.
