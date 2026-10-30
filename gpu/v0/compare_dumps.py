"""Compare two V0 dumps (v0_dump.cpp output) from different stock builds: per entry name, the number of values, how
many differ at all, and the largest relative difference, over the names both dumps have. A build that changes
stock's function values shows up here before any statistical test.

Usage: compare_dumps.py <reference_dump.txt> <other_dump.txt> [--rel 1e-12]
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path):
    rows = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        name, *vals = line.split()
        rows[name].append([float(v) for v in vals])
    return {k: np.array(v) for k, v in rows.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ref", type=Path)
    ap.add_argument("other", type=Path)
    ap.add_argument("--rel", type=float, default=1e-12, help="report entries whose largest relative difference exceeds this")
    args = ap.parse_args()
    a, b = load(args.ref), load(args.other)
    common = sorted(set(a) & set(b))
    only = sorted(set(a) ^ set(b))
    worst_all = 0.0
    for name in common:
        x, y = a[name], b[name]
        if x.shape != y.shape:
            print(f"{name}: shapes differ {x.shape} vs {y.shape}")
            continue
        both_nan = np.isnan(x) & np.isnan(y)
        x, y = np.where(both_nan, 0, x), np.where(both_nan, 0, y)
        if not np.array_equal(x[:, :-1], y[:, :-1]):
            print(f"{name}: inputs differ")  # the grids are the same in both builds; only the last column is a value
        d = np.abs(x[:, -1] - y[:, -1]) / np.maximum(np.abs(x[:, -1]), np.finfo(float).tiny)
        worst = float(np.nanmax(d)) if len(d) else 0.0
        worst_all = max(worst_all, worst)
        flag = "  <-- above --rel" if worst > args.rel else ""
        print(f"{name}: {len(d)} values, {int((d > 0).sum())} differ, max rel diff {worst:.2e}{flag}")
    print(f"{len(common)} entries compared, max rel diff {worst_all:.2e}; only in one dump: {only or 'none'}")


if __name__ == "__main__":
    main()
