"""Run pass 1 of the JAX port over a block range of one card and save the shard.

Usage: run_port.py <card> <n_trials> <block_start> <block_stop> <out.npz> [--seed S]
On CPU, run with XLA_FLAGS=--xla_cpu_multi_thread_eigen=false so one shard uses about one core.
"""
import argparse
from pathlib import Path
import sys
import time

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import pipeline, shards  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("n_trials", type=int)
    ap.add_argument("block_start", type=int)
    ap.add_argument("block_stop", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"output exists, not overwriting: {args.out}")
    t0 = time.process_time()
    w0 = time.time()
    res = pipeline.candidates(args.card, args.n_trials, args.seed, range(args.block_start, args.block_stop))
    res["cpu_s"] = time.process_time() - t0
    res["wall_s"] = time.time() - w0
    res["backend"] = jax.default_backend()
    res["device"] = jax.devices()[0].device_kind
    res["jax"] = jax.__version__
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shards.save(res, args.out)
    print(f"done {args.out}: {res['trials']} trials, {res['n_hits']} hits, cpu {res['cpu_s']:.1f} s, "
          f"wall {res['wall_s']:.1f} s on {res['backend']} ({res['device']})")


if __name__ == "__main__":
    main()
