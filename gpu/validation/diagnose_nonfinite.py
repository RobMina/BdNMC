"""Rows of one pass-1 block whose recoil electron is not finite (the "non-finite recoil electron" failure), with
the inputs of each step of pipeline._recoil recomputed on the host from the block's DM row: pz/|p| of the DM (the
argument of kinematics.theta), the electron energy window, and the electron, Recoil_DM and probability as the
device returned them. For a precision study (BDNMC_JAX_PRECISION) on the backend that shows the failure.
Usage: [BDNMC_JAX_PRECISION=float32] diagnose_nonfinite.py <card> <n_trials> <block> [--seed S] [--cap 1024]
"""
import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bdnmc_jax  # noqa: E402
from bdnmc_jax import card as cardmod, electron, pipeline, rng  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("n_trials", type=int)
    ap.add_argument("block", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--cap", type=int, default=1024)
    args = ap.parse_args()
    card = cardmod.parse_card(args.card)
    seed = card.seed if args.seed is None else args.seed
    det, sc, (ch,) = pipeline.setup(card)
    out = jax.device_get(pipeline.blocks_batch(det, sc, ch, card.e_density, seed, 0, jnp.asarray([args.block]),
                                               jnp.asarray([rng.BLOCK]), args.cap))
    out = jax.tree.map(lambda x: x[0], out)
    n = int(out["count"])
    bad = np.flatnonzero((np.arange(args.cap) < n) & (out["prob"] > 0) & ~out["e_finite"])
    print(f"precision {bdnmc_jax.PRECISION} backend {jax.default_backend()} device {jax.devices()[0].device_kind}; "
          f"block {args.block}: {n} candidates, {len(bad)} with a non-finite electron")
    for r in bad:
        dm = out["dm"][r]
        p32 = np.sqrt(np.float32(dm[1]) ** 2 + np.float32(dm[2]) ** 2 + np.float32(dm[3]) ** 2)
        p64 = np.sqrt(np.float64(dm[1]) ** 2 + np.float64(dm[2]) ** 2 + np.float64(dm[3]) ** 2)
        lo, hi = (float(np.asarray(f(sc, jnp.asarray(dm[0])))) for f in (electron.scat_min, electron.scat_max))
        print(f"  row {r}: dm {dm} (dtype {dm.dtype}); pz/|p| {np.float32(dm[3]) / p32:.9g} (float32), "
              f"{dm[3] / p64:.17g} (float64); Ee window [{lo:.6g}, {hi:.6g}]")
        print(f"         electron {out['electron'][r]}; recoil_dm {out['recoil_dm'][r]}; prob {out['prob'][r]:.4g}; "
              f"L {out['L'][r]:.6g}; recoil_done {out['recoil_done'][r]}")


if __name__ == "__main__":
    main()
