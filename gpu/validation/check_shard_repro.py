"""Rerun the first blocks of a saved shard with the current code and backend, and compare the candidates with the
saved ones: identical count and indices, and the largest relative difference of each float array.

Usage: check_shard_repro.py <shard.npz> [--blocks N] [--rtol R]
Use after a code change that should not change results, or across backends. Exit 1 if the candidate sets differ or
a relative difference exceeds R (default 0: bit-identical). NaN in both (the electron of a dead, prob = 0, row)
counts as equal.
"""
import argparse
from pathlib import Path
import sys

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import pipeline, shards  # noqa: E402

INDEX_KEYS = ("block", "trial", "slot", "ang_ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shard", type=Path)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--rtol", type=float, default=0.0)
    args = ap.parse_args()
    old = shards.load(args.shard)
    b0 = old["blocks"][0]
    blocks = range(b0, min(b0 + args.blocks, old["blocks"][1]))
    root = Path(__file__).resolve().parents[1]
    card = root / "phase0" / "cards" / "ref" / Path(old["card"]).name  # the saved path may be from another host
    new = pipeline.candidates(card, old["n_trials"], old["seed"], blocks,
                              parent_sampler=old.get("parent_sampler", "stock"))
    oc = {k: v[np.isin(old["candidates"]["block"], list(blocks))] for k, v in old["candidates"].items()}
    nc = new["candidates"]
    if len(nc["prob"]) != len(oc["prob"]) or any((nc[k] != oc[k]).any() for k in INDEX_KEYS):
        print(f"{args.shard}: candidate sets differ ({len(nc['prob'])} vs {len(oc['prob'])})")
        sys.exit(1)
    worst = {}
    for k in pipeline.CANDIDATE_KEYS:
        if k in INDEX_KEYS or k not in oc:  # keys added after the shard was written are not compared
            continue
        a, b = np.asarray(nc[k], float), np.asarray(oc[k], float)
        both_nan = np.isnan(a) & np.isnan(b)
        if (np.isnan(a) != np.isnan(b)).any():
            worst[k] = float("inf")
            continue
        a, b = np.where(both_nan, 0.0, a), np.where(both_nan, 0.0, b)
        scale = np.maximum(np.abs(b), np.finfo(float).tiny)
        worst[k] = float(np.max(np.abs(a - b) / scale)) if a.size else 0.0
    identical = all(v == 0 for v in worst.values())
    print(f"{args.shard}: {len(blocks)} blocks, {len(nc['prob'])} candidates, backend {jax.default_backend()} "
          f"(saved: {old.get('backend', 'cpu')}); " + ("bit-identical" if identical else
          "max rel diff " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items() if v)))
    sys.exit(0 if max(worst.values()) <= args.rtol else 1)


if __name__ == "__main__":
    main()
