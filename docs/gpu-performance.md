# GPU port: performance

Throughput of stock BdNMC and of the JAX port, measured on the environments in
[test-environments.md](test-environments.md). Physics validation, which every configuration quoted as a speed-up
must pass, is in [gpu-validation.md](gpu-validation.md).

**Target (P1):** at least 100x one core of stock `-O2`, in accepted events per second, on each of the four 5 MeV
generic cards, on a full A100 in float64, with binary output included and compile time reported separately.
**Status (2026-10-29):** provisional, pi0 cards only, no output written. The validated configuration reaches 126x on
`mb_pi0` and 86x on `he_pi0`. The faster envelope sampler (197x and 747x) has not passed V2. The brem cards are not
ported.

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

### Envelope parent sampler (not validated), commit `ed09f8d`

| Card | Env | Batch | Trials/s | Accepted/s | Device only, trials/s | vs stock `-O2` / `-O0` |
|---|---|---|---|---|---|---|
| `mb_pi0` | L1 (one core) | 1 | 4.9 M (profile) | | | |
| `mb_pi0` | G1 | 16 / 64 / 256 | 118.2 / 158.2 / **175.3 M** | 88.6 / 118.6 / **131.5 k** | 143.8 / 240.4 / 275.3 M | 197x / 590x |
| `he_pi0_cuboid` | L1 (one core) | 1 | 4.5 M (profile) | | | |
| `he_pi0_cuboid` | G1 | 16 / 64 / 256 | 74.1 / 89.6 / **111.8 M** | 883 / 1,067 / **1,332 k** | 125.2 / 213.6 / 251.4 M | 747x / 3,990x |

These numbers cannot be quoted as P1 until the envelope configuration passes V1 and V2.

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

With the stock samplers the parent rejection loop is the whole cost. With the envelope it falls to about half on
the GPU at large batches, and the host side (transfer of candidates, checks, concatenation) becomes the next cost:
18-36 % of the end-to-end time on `mb_pi0` and 41-58 % on `he_pi0` at batches 16-256 (device-only against
end-to-end), against 17-23 % with the stock sampler at batch 64.

## What is not measured yet

- Binary output (part of the P1 definition) and the text writer.
- A stock baseline on a core of the GPU node itself, and the whole-node comparison.
- A MIG slice.
- The brem cards (not ported).
- P2 (a 20-mass x all-channel grid in one GPU-hour): masses are already runtime inputs, so a mass scan reuses the
  compiled kernel, but the grid itself has not been run.

## Reproducing

```bash
cd gpu
python validation/bench.py phase0/cards/ref/mb_pi0_m005.dat 134217728 --batch 16 64 256 [--parent-sampler envelope]
python validation/profile_port.py phase0/cards/ref/he_pi0_cuboid_m005.dat --batch 16 64 256 [--parent-sampler envelope]
```

On CPU, set `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false` to keep one process near one core.
