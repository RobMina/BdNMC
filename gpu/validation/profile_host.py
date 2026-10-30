"""Where the end-to-end time goes on the host for one card: pass 1 (`pipeline.candidates`), pass 2 (`finish`) and
the HDF5 write, each timed after a warm-up call that compiles, with a cProfile table of the functions that take the
time. The write is timed into each of --out-dirs (for example shared scratch against node-local disk), and the
file is deleted afterwards. Also reports the bytes copied from the device per batch and written per accepted row.

Usage: profile_host.py <card> <n_trials> [--batch 256] [--cap 16384] [--out-dirs D ...] [--top 25] [--json out.json]
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import card as cardmod, output, pipeline, rng  # noqa: E402


def profiled(f, top):
    prof = cProfile.Profile()
    t = time.perf_counter()
    prof.enable()
    out = f()
    prof.disable()
    dt = time.perf_counter() - t
    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(top)
    return out, dt, s.getvalue()


def batch_bytes(card_path, batch, cap):
    card = cardmod.parse_card(card_path)
    det, sc, (ch,) = pipeline.setup(card)
    out = pipeline.blocks_batch(det, sc, ch, card.e_density, card.seed, 0, jnp.arange(batch),
                                jnp.full(batch, rng.BLOCK), cap)
    return sum(x.nbytes for x in jax.tree.leaves(out)), {k: v.nbytes for k, v in out.items() if hasattr(v, "nbytes")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("n_trials", type=int)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap", type=int, default=16384)
    ap.add_argument("--out-dirs", type=Path, nargs="+", default=[Path("/tmp")])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    kw = {"batch": args.batch, "cap": args.cap}
    t = time.perf_counter()
    pipeline.finish(pipeline.candidates(args.card, args.batch * rng.BLOCK, **kw))
    rep = {"card": str(args.card), "n_trials": args.n_trials, "batch": args.batch, "cap": args.cap,
           "device": jax.devices()[0].device_kind, "warm_s": time.perf_counter() - t}
    total, per_key = batch_bytes(args.card, args.batch, args.cap)
    rep["device_get_bytes_per_batch"] = total
    rep["device_get_bytes_per_key"] = per_key
    res, rep["candidates_s"], prof_c = profiled(lambda: pipeline.candidates(args.card, args.n_trials, **kw), args.top)
    t = time.perf_counter()
    res = pipeline.finish(res)
    rep["finish_s"] = time.perf_counter() - t
    t = time.perf_counter()
    rows, _ = output.accepted_events(res)
    rep["accepted_events_s"] = time.perf_counter() - t
    rep["n_hits"], rep["accepted"] = res["n_hits"], res["accepted"]
    rep["write"] = []
    for d in args.out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"profile_{args.card.stem}.h5"
        _, dt, prof_w = profiled(lambda: output.write_hdf5(res, path), args.top)
        size = path.stat().st_size
        path.unlink()
        rep["write"].append({"dir": str(d), "write_s": dt, "bytes": size, "bytes_per_row": size / max(res["accepted"], 1)})
        print(f"==== write to {d}: {dt:.2f} s, {size / 1e9:.2f} GB, {size / max(res['accepted'], 1):.0f} B/row\n{prof_w}")
    print(f"==== candidates {rep['candidates_s']:.2f} s ({res['n_hits']} candidates, "
          f"{total / 1e9:.2f} GB from the device per batch of {args.batch})\n{prof_c}")
    print(json.dumps({k: v for k, v in rep.items() if k != "device_get_bytes_per_key"}, indent=1))
    print("bytes per key per batch:", {k: f"{v / 1e6:.0f} MB" for k, v in per_key.items()})
    if args.json:
        args.json.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
