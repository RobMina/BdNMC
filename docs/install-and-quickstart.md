# Install and quickstart

Building and running BdNMC. Every parameter-card option is in [parameter-card.md](parameter-card.md).

## Build and run

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

The sample card ends with a comment giving the result to expect, 284.31 signal events, varying by about ±10 at
its sample size of 1,000 events. That number is out of date: the current code gives about 195 (187-199 in five
unseeded runs, 198.05 with the card's commented `seed 114234`), the same with and without optimisation:

```
Predicted number of signal events = 198.052
Predicted number of signal events from channel 1 pi0_decay = 198.052
```

**Optimised build.** `build/makefile` builds with `-O2`. Before this change it set no optimisation flag, so
older builds are `-O0`. Changing the flags does not trigger a rebuild, so in a checkout that was built before,
remove the old objects first:

```bash
cd build
rm -f *.o main && rm -rf .d
make main
cd ..
```

With the same seed, `-O2` gives the same events as `-O0`, with printed values within 2.3e-9 relative. CPU time of
20 M trials on one core (Apple M5, Apple clang 21), m_chi = 5 MeV:

| Card | `-O0` | `-O2` | Speed-up |
|---|---|---|---|
| 8.9 GeV beam, pi0 decay (Sanford-Wang), sphere detector | 68.9 s | 23.0 s | 3.0x |
| 8.9 GeV beam, proton brem, sphere detector | 135.9 s | 43.7 s | 3.1x |
| 120 GeV beam, pi0 decay (BMPT), cuboid detector | 699.1 s | 131.0 s | 5.3x |
| 120 GeV beam, proton brem, cuboid detector | 1,340 s | 193.3 s | 6.9x |

**Card generator.** `Parameter_Card_Generator.py` copies a card, replaces the values of the keys you give
(`run_output_file(card, new_card, {"dark_photon_mass": 0.1, ...})`) and, by default, runs `./bin/BDNMC` on the
copy; its `test()` scans three dark-photon masses with `parameter.dat`. Run it from the repository root.

**Outputs:**
- `output_mode comprehensive` writes every particle of every event to `output_file`;
- `summary` writes only the summary line;
- `particle_list` writes parent four-momenta for reuse as `production_distribution particle_list`.
- `dm_detector_distribution` writes one line per DM that reaches the detector.

All of them, with the pitfalls of the card format, are in [parameter-card.md](parameter-card.md#output).
