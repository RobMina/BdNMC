"""Stock's parent-sampler bars against the densities they bound: for each card, the grid maximum of the parent
density, the bar stock uses (Sanford-Wang: fixed; BMPT: running maximum, so the port's fixed bar is shown), their
ratio, and for a fixed bar below the maximum the fraction of the density mass above the bar (which stock clips).

Usage: parent_bars.py <card> [<card> ...] [--n 2000] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import card as cardmod, parents, pipeline  # noqa: E402


def grid(dist, n):
    """Density on an n x n midpoint grid of the stock proposal rectangle in (p, v), and the cell area."""
    p = (np.arange(n) + 0.5) / n * dist.p_range
    v = (np.arange(n) + 0.5) / n
    P, V = np.meshgrid(p, v, indexing="ij")
    th = dist.theta_of(jnp.asarray(P), jnp.asarray(V))
    f = np.asarray(dist.density(jnp.asarray(P), th))
    if isinstance(dist, parents.BMPT):
        f = np.where(np.asarray(dist.xR(jnp.asarray(P), th)) <= 1, f, 0.0)
    return np.where(np.isfinite(f), f, 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cards", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = []
    for c in args.cards:
        _, _, (ch,) = pipeline.setup(cardmod.parse_card(c))
        f = grid(ch.dist, args.n)
        fmax_grid = float(f.max())
        row = {"card": str(c), "dist": type(ch.dist).__name__, "kind": getattr(ch.dist, "kind", None) or
               getattr(ch.dist, "choice", None), "grid_max": fmax_grid, "bar": ch.fmax,
               "grid_max_over_bar": fmax_grid / ch.fmax}
        if isinstance(ch.dist, parents.SanfordWang):  # fixed stock bar: how much of the law it clips
            row["clipped_fraction"] = float(np.clip(f - ch.fmax, 0, None).sum() / f.sum())
            row["fraction_of_cells_above_bar"] = float((f > ch.fmax).mean())
        out.append(row)
        print(json.dumps(row))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
