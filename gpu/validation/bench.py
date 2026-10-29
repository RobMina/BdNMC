"""Throughput of pass 1 + pass 2 on one card: compile time, steady-state trials/s, candidates/s and accepted
(unweighted, two-pass) events/s, for a grid of blocks-per-call batch sizes; and the device-only rate (no host
transfer or checks) at each batch size. Output writing is not included (goal P1 excludes it).

Usage: bench.py <card> <n_trials> [--batch 1 4 16] [--cap C] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import card as cardmod, pipeline, rng  # noqa: E402


def device_only(card_path, batch, cap, n_calls, parent_sampler):
    card = cardmod.parse_card(card_path)
    det, sc, (ch,) = pipeline.setup(card, parent_sampler)
    nv = jnp.full(batch, rng.BLOCK)
    call = lambda i: pipeline.blocks_batch(det, sc, ch, card.e_density, card.seed, 0,  # noqa: E731
                                           jnp.arange(i * batch, (i + 1) * batch), nv, cap)
    jax.block_until_ready(call(0))
    t = time.perf_counter()
    outs = [call(i) for i in range(1, n_calls + 1)]
    jax.block_until_ready(outs)
    return n_calls * batch * rng.BLOCK / (time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("n_trials", type=int)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--cap", type=int, default=1024)
    ap.add_argument("--parent-sampler", default="stock", choices=pipeline.PARENT_SAMPLERS)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    kw = {"parent_sampler": args.parent_sampler}
    out = {"card": str(args.card), "n_trials": args.n_trials, "parent_sampler": args.parent_sampler,
           "backend": jax.default_backend(),
           "device": jax.devices()[0].device_kind, "jax": jax.__version__, "runs": []}
    for b in args.batch:
        t = time.perf_counter()
        warm = pipeline.candidates(args.card, b * rng.BLOCK, batch=b, cap=args.cap, **kw)  # compile, cap growth
        t_warm = time.perf_counter() - t
        cap = warm["cap"]
        t = time.perf_counter()
        res = pipeline.finish(pipeline.candidates(args.card, args.n_trials, batch=b, cap=cap, **kw))
        dt = time.perf_counter() - t
        dev = device_only(args.card, b, cap, max(2, min(50, args.n_trials // (b * rng.BLOCK))), args.parent_sampler)
        r = {"batch": b, "cap": cap, "first_call_s": t_warm, "wall_s": dt, "trials_per_s": res["trials"] / dt,
             "candidates_per_s": res["n_hits"] / dt, "accepted": res["accepted"], "accepted_per_s": res["accepted"] / dt,
             "device_only_trials_per_s": dev}
        out["runs"].append(r)
        print(f"batch {b:3d} cap {cap:6d}: first call {t_warm:6.1f} s; {r['trials_per_s'] / 1e6:8.2f} M trials/s, "
              f"{r['accepted_per_s'] / 1e3:8.1f} k accepted/s end to end; device only {dev / 1e6:8.2f} M trials/s",
              flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
