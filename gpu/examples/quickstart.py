"""Run one parameter card through the JAX port and print the signal prediction.

Usage: python examples/quickstart.py <card> [--trials N] [--parent-sampler stock|envelope] [--weighted]
                                     [--text events.dat] [--summary summary.dat] [--hdf5 events.h5] [--save out.npz]

Prints the backend, the weighted and two-pass unweighted predicted signal events (`se`, as stock prints
"Predicted number of signal events"), the accepted-event count and the throughput.
- --text writes the accepted events in stock's comprehensive format, --summary appends stock's summary lines,
  --hdf5 writes them as HDF5 (needs h5py); --weighted writes weighted events (one per DM) instead of unweighted.
- --save writes the pass-1 candidates (every DM that reaches the detector, with its recoil electron) as a shard
  that `bdnmc_jax.shards.load` can read.
"""
import argparse
from pathlib import Path
import sys
import time

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import output, pipeline, shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("card", type=Path)
    ap.add_argument("--trials", type=int, default=1 << 22)
    ap.add_argument("--parent-sampler", default="stock", choices=pipeline.PARENT_SAMPLERS)
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--text", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--hdf5", type=Path)
    ap.add_argument("--save", type=Path)
    args = ap.parse_args()
    t = time.perf_counter()
    cand = pipeline.candidates(args.card, args.trials, parent_sampler=args.parent_sampler)
    res = pipeline.finish(cand, weighted=args.weighted)
    dt = time.perf_counter() - t
    print(f"backend {jax.default_backend()} ({jax.devices()[0].device_kind}), {res['trials']} trials in {dt:.1f} s "
          f"(compile included), parent sampler {args.parent_sampler}")
    print(f"DM reaching the detector: {res['n_hits']}; accepted (two-pass unweighted): {res['accepted']}")
    print(f"Predicted number of signal events (weighted) = {res['se_weighted']:.6g}")
    print(f"Predicted number of signal events (unweighted, two-pass) = {res['se']:.6g}")
    if args.text:
        print(f"{output.write_text(res, args.text, summary_path=args.summary)} events written to {args.text}")
    if args.hdf5:
        print(f"{output.write_hdf5(res, args.hdf5)} accepted DM written to {args.hdf5}")
    if args.save:
        shards.save(cand, args.save)
        print(f"candidates saved to {args.save}")


if __name__ == "__main__":
    main()
