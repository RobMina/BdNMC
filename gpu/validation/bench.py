"""Throughput of pass 1 + pass 2 on one card: compile time, steady-state trials/s, candidates/s and accepted
(unweighted, two-pass) events/s, for a grid of blocks-per-call batch sizes; and the device-only rate (no host
transfer or checks) at each batch size. With --write hdf5 (goal P1's binary output) or --write text, writing the
accepted events is timed and included in the end-to-end rate; the file is deleted afterwards. Runs that write use
`pipeline.events` (accepted rows gathered on the device) with `stream` (the write takes each batch as its copy
arrives, so the copies are timed in the write), unless --no-stream or --host-path (`finish(candidates(...))`).

Usage: bench.py <card> <n_trials> [--batch 1 4 16] [--cap C] [--write none|hdf5|text] [--host-path] [--no-stream]
       [--out-dir D] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import card as cardmod, output, pipeline, rng  # noqa: E402


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
    ap.add_argument("--write", choices=["none", "hdf5", "text"], default="none",
                    help="also write the accepted events (to --out-dir) and include the write in the end-to-end rate")
    ap.add_argument("--host-path", action="store_true", help="write through finish(candidates(...)), not events()")
    ap.add_argument("--no-stream", action="store_true",
                    help="events() joins the rows on the host before the write, rather than the write taking each "
                         "batch as its copy arrives")
    ap.add_argument("--out-dir", type=Path, default=Path(tempfile.gettempdir()))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    kw = {"parent_sampler": args.parent_sampler}
    device_path = args.write != "none" and not args.host_path

    def generate(n, b, cap):
        if device_path:
            return pipeline.events(args.card, n, batch=b, cap=cap, stream=not args.no_stream, **kw)
        return pipeline.finish(pipeline.candidates(args.card, n, batch=b, cap=cap, **kw))
    out = {"card": str(args.card), "n_trials": args.n_trials, "parent_sampler": args.parent_sampler,
           "path": ("events" + ("" if args.no_stream else "+stream")) if device_path else "candidates+finish",
           "backend": jax.default_backend(),
           "device": jax.devices()[0].device_kind, "jax": jax.__version__, "runs": []}
    for b in args.batch:
        t = time.perf_counter()
        cap = generate(b * rng.BLOCK, b, args.cap)["cap"]  # compile, cap growth
        t_warm = time.perf_counter() - t
        t = time.perf_counter()
        res = generate(args.n_trials, b, cap)
        t_gen = time.perf_counter() - t
        t_write, size = 0.0, 0
        if args.write != "none":
            path = args.out_dir / f"bench_{args.card.stem}_b{b}.{'h5' if args.write == 'hdf5' else 'dat'}"
            t = time.perf_counter()
            (output.write_hdf5 if args.write == "hdf5" else output.write_text)(res, path)
            t_write = time.perf_counter() - t
            size = path.stat().st_size
            path.unlink()
        dt = t_gen + t_write
        dev = device_only(args.card, b, cap, max(2, min(50, args.n_trials // (b * rng.BLOCK))), args.parent_sampler)
        r = {"batch": b, "cap": cap, "first_call_s": t_warm, "generate_s": t_gen, "write": args.write,
             "write_s": t_write, "file_bytes": size, "wall_s": dt, "trials_per_s": res["trials"] / dt,
             "candidates_per_s": res["n_hits"] / dt, "accepted": res["accepted"], "accepted_per_s": res["accepted"] / dt,
             "device_only_trials_per_s": dev}
        out["runs"].append(r)
        print(f"batch {b:3d} cap {cap:6d}: first call {t_warm:6.1f} s; {r['trials_per_s'] / 1e6:8.2f} M trials/s, "
              f"{r['accepted_per_s'] / 1e3:8.1f} k accepted/s end to end"
              + (f" (write {args.write} {t_write:.2f} s, {size / 1e6:.1f} MB)" if args.write != "none" else "")
              + f"; device only {dev / 1e6:8.2f} M trials/s", flush=True)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
