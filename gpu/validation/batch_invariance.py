"""Run the same blocks of a card at two batch/cap settings and report, per candidate array, whether they are
bit-identical and otherwise how many elements differ and by how much (relative).

Usage: batch_invariance.py <card> [<card> ...] [--blocks N] [--seed S] [--parent-sampler stock|envelope]
The settings are those of the unit tests: batch 1 / cap 64 against batch 3 / cap 8192, on the default backend.
Exit 1 if the candidate sets differ (counts or index arrays); differences in float values are reported, not judged.
"""
import argparse
from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import PRECISION, pipeline, rng  # noqa: E402

INDEX_KEYS = ("block", "trial", "slot", "ang_ok")


def compare(card, blocks, seed, sampler):
    n = blocks * rng.BLOCK - 77
    a = pipeline.candidates(card, n, seed=seed, batch=1, cap=64, parent_sampler=sampler)["candidates"]
    b = pipeline.candidates(card, n, seed=seed, batch=3, cap=8192, parent_sampler=sampler)["candidates"]
    same_set = a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in INDEX_KEYS if k in a)
    print(f"== {card.name} ({sampler}, {n} trials, {len(a['prob'])} candidates): candidate sets "
          f"{'identical' if same_set else 'DIFFER'}")
    print(f"   {'key':<14}{'shape':<16}{'differing':>11}{'fraction':>11}{'max rel':>12}{'max abs':>12}")
    for k in sorted(a):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            print(f"   {k:<14}{str(x.shape):<16} shapes differ: {y.shape}")
            continue
        if np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
            print(f"   {k:<14}{str(x.shape):<16}{'0':>11}{'0':>11}{'identical':>12}{'':>12}")
            continue
        d = (x != y) & ~(np.isnan(x) & np.isnan(y)) if x.dtype.kind == "f" else x != y
        rows = d.reshape(len(d), -1).any(axis=1) if d.ndim > 1 else d
        ad = np.abs(x - y)[d].astype(float) if x.dtype.kind == "f" else np.array([np.inf])
        rel = ad / np.maximum(np.abs(x), np.abs(y))[d] if x.dtype.kind == "f" else ad
        print(f"   {k:<14}{str(x.shape):<16}{int(rows.sum()):>11}{rows.mean():>11.2e}{rel.max():>12.2e}{ad.max():>12.2e}")
    return same_set


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cards", type=Path, nargs="+")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--parent-sampler", default="stock")
    args = ap.parse_args()
    print(f"backend {jax.default_backend()} ({jax.devices()[0].device_kind}), precision {PRECISION}")
    ok = [compare(c, args.blocks, args.seed, args.parent_sampler) for c in args.cards]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
