"""Goal P2: a mass grid of the generic cards, end to end in one process: for each base card and m_chi, write the card,
parse it, generate at least --events accepted (unweighted) events with pipeline.events, and write them to HDF5. Timed
as a whole and per sample; XLA compilations are counted (P2 wants masses as runtime inputs, so a new mass should
not compile), and the summary JSON records per sample the trials, accepted events, se, wall time and compiles.

Trials per sample: --events / (accepted per trial), from the previous mass of the same base card, or from a one-batch
pilot for the first; times --margin. A sample that comes out short is rerun with enough trials (counted in the time).
Only dark_matter_mass changes (as in make_ref_cards.py); the seed is the base card's plus the mass index.
Usage: run_grid.py <base card> [...] --out-dir D [--masses-mev ...] [--n-masses 20] [--events 80000] [--batch 256]
       [--keep] [--json summary.json]
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

T_START = time.perf_counter()  # P2 is end to end: the clock starts before JAX is imported

import jax  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase0"))
import bdnmc_jax  # noqa: E402
from bdnmc_jax import card as cardmod, output, pipeline, rng  # noqa: E402
from make_ref_cards import set_key  # noqa: E402

COMPILES = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, secs, **kw: COMPILES.append(secs) if event == "/jax/core/compile/backend_compile_duration" else None)


def write_card(base, mev, index, outdir):
    lines = base.read_text().splitlines()
    seed = cardmod.parse_card(base).seed + index + 1
    lines = set_key(lines, "dark_matter_mass", f"{mev / 1000:.10g}")
    lines = set_key(lines, "seed", seed)
    stem = f"{base.stem.rsplit('_m', 1)[0]}_grid{index:02d}"
    path = outdir / f"{stem}.dat"
    path.write_text("\n".join(lines) + "\n")
    return path, seed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cards", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--masses-mev", type=float, nargs="+")
    ap.add_argument("--n-masses", type=int, default=20, help="log-spaced from 1 to 45 MeV, when --masses-mev is not given")
    ap.add_argument("--events", type=int, default=80_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--cap", type=int, default=1024, help="starting cap (each base card keeps the largest it needed)")
    ap.add_argument("--margin", type=float, default=1.15)
    ap.add_argument("--parent-sampler", default="stock", choices=pipeline.PARENT_SAMPLERS)
    ap.add_argument("--keep", action="store_true", help="keep the HDF5 files (default: deleted after the size is noted)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    masses = args.masses_mev or list(np.geomspace(1.0, 45.0, args.n_masses))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for base in args.cards:
        cap, per_trial = args.cap, None
        for i, mev in enumerate(masses):
            t0, c0 = time.perf_counter(), len(COMPILES)
            path, seed = write_card(base, mev, i, args.out_dir)
            if per_trial is None:  # pilot: one batch
                pilot = pipeline.events(path, args.batch * rng.BLOCK, batch=args.batch, cap=cap,
                                        parent_sampler=args.parent_sampler, stream=True)
                cap = max(cap, pilot["cap"])
                per_trial = max(pilot["accepted"], 1) / pilot["trials"]
            runs = 0
            while True:
                n = int(math.ceil(args.events / per_trial * args.margin / rng.BLOCK)) * rng.BLOCK
                res = pipeline.events(path, n, batch=args.batch, cap=cap, parent_sampler=args.parent_sampler,
                                      stream=True)
                runs += 1
                cap = max(cap, res["cap"])
                per_trial = max(res["accepted"], 1) / res["trials"]
                if res["accepted"] >= args.events:
                    break
            h5 = args.out_dir / f"{path.stem}.h5"
            output.write_hdf5(res, h5, extra_attrs={"m_chi_MeV": mev})
            size = h5.stat().st_size
            if not args.keep:
                h5.unlink()
            dt = time.perf_counter() - t0
            samples.append({"card": str(base), "m_chi_MeV": mev, "seed": seed, "trials": res["trials"],
                            "accepted": res["accepted"], "se": res["se"], "pmax": res["pmax"], "cap": cap,
                            "runs": runs, "wall_s": dt, "compiles": len(COMPILES) - c0,
                            "compile_s": sum(COMPILES[c0:]), "hdf5_bytes": size})
            print(f"{base.stem} m_chi {mev:7.3f} MeV: {res['trials'] / 1e6:8.1f} M trials, {res['accepted']:7d} "
                  f"accepted, se {res['se']:.4g}, {dt:6.2f} s, {len(COMPILES) - c0} compiles"
                  + (f", {runs} runs" if runs > 1 else ""), flush=True)
    total = time.perf_counter() - T_START
    out = {"precision": bdnmc_jax.PRECISION, "backend": jax.default_backend(), "device": jax.devices()[0].device_kind,
           "jax": jax.__version__, "parent_sampler": args.parent_sampler, "events": args.events, "batch": args.batch,
           "n_samples": len(samples), "total_wall_s": total, "compiles": len(COMPILES), "compile_s": sum(COMPILES),
           "min_accepted": min(s["accepted"] for s in samples), "samples": samples}
    print(f"{len(samples)} samples in {total:.1f} s end to end ({total / 3600:.3f} GPU-hours); {len(COMPILES)} "
          f"compiles ({sum(COMPILES):.1f} s); fewest accepted {out['min_accepted']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
