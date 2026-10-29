"""How much repeated values in a V2 variable can matter: for a stock electrons file (extract_stock.py), the number of
distinct values of each parent variable, the largest multiplicity, and how many of compare_v.py's quantile bin edges
fall exactly on a value that occurs more than once (then which side of the edge that value lands on decides a whole
cluster's bin). With --port-dir, the same for the port's shards of that point, and the share of port values that
differ from their own rounding to 10 significant digits (stock's text precision).

Usage: diagnose_ties.py <stock.npz> [--port-dir validation/out/v/<stem>] [--n-bins 40]
"""
import argparse
from pathlib import Path

import numpy as np

import compare_v as cv
from bdnmc_jax import pipeline, shards


def report(name, x, edges):
    vals, counts = np.unique(x, return_counts=True)
    rep = set(vals[counts > 1])
    on_edge = [e for e in edges[1:-1] if e in rep]
    print(f"  {name}: {len(x)} values, {len(vals)} distinct, max multiplicity {counts.max()}, "
          f"{(counts > 1).sum()} repeated values holding {counts[counts > 1].sum() / len(x):.1%}; "
          f"{len(on_edge)} of {len(edges) - 2} inner edges on a repeated value")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stock", type=Path)
    ap.add_argument("--port-dir", type=Path)
    ap.add_argument("--n-bins", type=int, default=40)
    args = ap.parse_args()
    s = np.load(args.stock)
    vs = cv.variables(s["electron"], s["dm"], s["parent"], s["vertex"])
    print(f"stock {args.stock.name}")
    edges = {}
    for name in ("parent_p", "parent_theta", "E_e"):
        edges[name] = np.quantile(vs[name], np.linspace(0, 1, args.n_bins + 1))
        report(name, vs[name], edges[name])
    if args.port_dir:
        res = pipeline.merge(shards.load_all(args.port_dir.glob("shard*.npz")))
        vp, w = cv.port_v2_sample(res)
        print(f"port {args.port_dir}")
        for name in ("parent_p", "parent_theta", "E_e"):
            report(name, vp[name], edges[name])
            r = np.array([float("%.10g" % v) for v in vp[name][:200000]])
            print(f"    {np.mean(r != vp[name][:200000]):.1%} of the first 200000 port values change when rounded "
                  "to 10 significant digits")
        # the parent four-momentum itself, as stock prints it
        p = res["candidates"]["parent"]
        r = np.vectorize(lambda v: float("%.10g" % v))(p[:20000])
        print(f"    parent four-momentum components changed by 10-digit rounding: {np.mean(r != p[:20000]):.1%}")


if __name__ == "__main__":
    main()
