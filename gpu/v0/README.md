# V0 harness

`v0_dump.cpp` links the stock `-O2` library and prints stock function values, with their inputs, at `%.17g`:
`EeTMax`, `EeTMin`, `sigma2`, `dsigmadEe`, `ThetaEe`, `Ldet` (sphere, cuboid, rotated cuboid), the pi0 branching
ratios, the Sanford-Wang density and the BMPT invariant cross section. `tests/test_v0.py` compares the JAX port with
it at the V0 tolerances: (1e-12 + 64 eps x condition number) relative for closed-form functions (the condition number
covers cancellation, where clang's default FMA contraction and XLA round differently), 1e-9 m for `Ldet`. It skips
when `v0_dump.txt` is absent. The committed `v0_dump.txt` comes from a stock build of `RobMina/BdNMC` `bf45d75`.

Build, with `BDNMC_SRC` the BdNMC source tree and `STOCK` the directory of its `-O2` build (`libBdNMC.so`):

```bash
c++ -std=c++14 -O2 -I"$BDNMC_SRC/src" v0/v0_dump.cpp "$STOCK/libBdNMC.so" -o v0/v0_dump
```

Run from the BdNMC source root (stock opens `data/*.dat` relative to the working directory); on Linux use
`LD_LIBRARY_PATH`:

```bash
(cd "$BDNMC_SRC" && DYLD_LIBRARY_PATH="$STOCK" "$OLDPWD/v0/v0_dump") > v0/v0_dump.txt
```
