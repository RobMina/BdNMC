"""The proton-brem envelope per card and m_V: build time, expected acceptance (proposals per accepted V against the
stock bar's), largest bound against the stock bar, and an independent check of the bound on points uniform over the
whole window (the builder's own check draws points cell by cell).

Usage: brem_envelope_info.py <card> [...] [--mv 0.01 0.1 0.5] [--n-check 2000000] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import brem, card as cardmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cards", type=Path, nargs="+")
    ap.add_argument("--mv", type=float, nargs="+", default=[0.01, 0.1, 0.5])
    ap.add_argument("--n-check", type=int, default=2_000_000)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    rows = []
    for path in args.cards:
        card = cardmod.parse_card(path)
        (ch,) = card.channels
        base = brem.ProtonBrem(card.beam_energy, ch.zmin, ch.zmax, ch.ptmin, ch.ptmax)
        env = brem.BremEnvelope(base)
        for mv in args.mv:
            t0 = time.perf_counter()
            t = env.tables(mv)
            dt = time.perf_counter() - t0
            fmax = base.find_fmax(mv)
            area = (base.zmax - base.zmin) * (base.ptmax - base.ptmin)
            bound, cdf = np.asarray(t.bound), np.asarray(t.cdf)
            dz, dpt = np.asarray(t.dz), np.asarray(t.dpt)
            cell_area = (dz[:, None] * dpt[None, :]).ravel()
            integral_bound = float((bound * cell_area).sum())
            # stock acceptance = integral(shape) / (fmax x area); envelope = integral(shape) / integral(bound)
            rs = np.random.default_rng(1)
            u = rs.random((args.n_check, 2))
            z = base.zmin + (base.zmax - base.zmin) * u[:, 0]
            pt = base.ptmin + (base.ptmax - base.ptmin) * u[:, 1]
            f = np.asarray(base.shape(jnp.asarray(z), jnp.asarray(pt), mv))
            iz = np.clip(np.searchsorted(np.asarray(t.z_lo), z, side="right") - 1, 0, env.n_z - 1)
            ip = np.clip(np.searchsorted(np.asarray(t.pt_lo), pt, side="right") - 1, 0, env.n_pt - 1)
            ratio = f / bound[iz * env.n_pt + ip]
            integral_f = float(f.mean() * area)
            row = {"card": str(path), "m_v": mv, "build_s": dt, "efficiency_table": float(t.efficiency),
                   "efficiency_mc": integral_f / integral_bound, "stock_efficiency_mc": integral_f / (fmax * area),
                   "max_bound_over_stock_bar": float(bound.max() / fmax), "uniform_check_max_ratio": float(ratio.max()),
                   "uniform_check_points": args.n_check, "n_cells": int(len(cdf))}
            row["proposals_per_accept"] = {"envelope": 1 / row["efficiency_mc"], "stock": 1 / row["stock_efficiency_mc"]}
            rows.append(row)
            print(f"{path.stem} m_V {mv}: build {dt:.2f} s, efficiency {row['efficiency_mc']:.3f} (table "
                  f"{row['efficiency_table']:.3f}), stock {row['stock_efficiency_mc']:.4f} "
                  f"({1 / row['stock_efficiency_mc']:.1f} proposals per V); max bound / stock bar "
                  f"{row['max_bound_over_stock_bar']:.3f}; uniform check max f / bound {row['uniform_check_max_ratio']:.3f}")
            if not ratio.max() <= 1:
                print("  BOUND EXCEEDED")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
