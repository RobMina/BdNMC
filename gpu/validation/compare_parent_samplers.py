"""High-statistics check that the envelope parent sampler draws the same law as the stock-proposal sampler: N
blocks of parents from each (independent seeds), with a two-sample chi2 per histogram and the largest bin pulls.
Meson cards: p, theta (log bins), pT and, for BMPT, v = theta / theta*(p). Proton-brem cards (the parent is the V):
the sampled z and pT (linear bins), pT in log bins, the V's p and theta (log bins), and the joint (z, pT) law in
20 x 20 cells of equal width in z and in log pT.

Usage: compare_parent_samplers.py <card> [--blocks N] [--batch B] [--json out.json]
"""
import argparse
import json
import math
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import brem, card as cardmod, parents, pipeline, rng  # noqa: E402
from bdnmc_jax.constants import MASS_PROTON as mp  # noqa: E402


def sampler(ch, seed, batch):
    if ch.kind == "brem":
        def one(b):
            return brem.sample_block(ch.dist, rng.block_key(seed, 0, b), rng.BLOCK, ch.fmax, ch.m_v)
    else:
        def one(b):
            return parents.sample_block(ch.dist, rng.block_key(seed, 0, b), rng.BLOCK, ch.fmax, ch.m_meson)

    @jax.jit
    def call(b0):
        par, info = jax.vmap(one)(b0 + jnp.arange(batch))
        par = par.reshape(-1, 4)
        mom = jnp.linalg.norm(par[:, 1:], axis=1)
        return {"p": mom, "theta": jnp.arccos(par[:, 3] / mom), "pT": jnp.hypot(par[:, 1], par[:, 2]),
                "pz": par[:, 3]}, info
    return call


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("--blocks", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    card = cardmod.parse_card(args.card)
    chans = {s: pipeline.setup(card, s)[2][0] for s in ("stock", "envelope")}
    base = chans["stock"].dist
    if isinstance(base, brem.ProtonBrem):
        p_beam = math.sqrt(base.beam_energy ** 2 - mp * mp)
        pt_lo = max(base.ptmin, 1e-4 * (base.ptmax - base.ptmin))
        log_pt = np.concatenate([[base.ptmin], np.geomspace(pt_lo, base.ptmax, 200)])
        th_max = math.atan2(base.ptmax, base.zmin * p_beam)
        edges = {"z": np.linspace(base.zmin, base.zmax, 201), "pT": np.linspace(base.ptmin, base.ptmax, 201),
                 "pT_log": log_pt,
                 "p": np.linspace(base.zmin * p_beam, math.hypot(base.zmax * p_beam, base.ptmax), 201),
                 "theta": np.concatenate([[0], np.geomspace(1e-4 * th_max, th_max, 200)])}
        z2, pt2 = np.linspace(base.zmin, base.zmax, 21), log_pt[::10]
    else:
        edges = {"p": np.linspace(0, base.p_range, 201),
                 "theta": np.concatenate([[0], np.geomspace(1e-5, np.pi / 2, 200)]), "pT": None,
                 "v": np.linspace(0, 1, 201)}
    hists, over, proposed = {}, {}, {}
    for s, ch in chans.items():
        call = sampler(ch, 1000 + len(hists), args.batch)
        h, n_over, n_prop = {}, 0, 0
        for i in range(0, args.blocks, args.batch):
            d, info = jax.device_get(call(i))
            assert (info["filled"] == rng.BLOCK).all()
            n_over += int(info["over_bar"].sum())
            n_prop += int(info["proposed"].sum())
            if isinstance(base, parents.BMPT):
                d["v"] = d["theta"] / np.asarray(base.theta_of(jnp.asarray(d["p"]), 1.0))
            if isinstance(base, brem.ProtonBrem):
                d["z"], d["pT_log"] = d["pz"] / p_beam, d["pT"]
                iz = np.clip(np.searchsorted(z2, d["z"], side="right") - 1, 0, len(z2) - 2)
                ip = np.clip(np.searchsorted(pt2, d["pT"], side="right") - 1, 0, len(pt2) - 2)
                h["z_pT"] = h.get("z_pT", 0) + np.bincount(iz * (len(pt2) - 1) + ip,
                                                           minlength=(len(z2) - 1) * (len(pt2) - 1))
            if edges["pT"] is None:
                edges["pT"] = np.linspace(0, np.quantile(d["pT"], 0.9999), 201)
            for k, e in edges.items():
                if k in d and e is not None:
                    h[k] = h.get(k, 0) + np.histogram(d[k], e)[0]
        hists[s], over[s], proposed[s] = h, n_over, n_prop
    n_par = args.blocks * rng.BLOCK
    out = {"card": str(args.card), "dist": type(base).__name__, "parents_per_sampler": n_par, "over_bar": over,
           "proposals_per_parent": {s: proposed[s] / n_par for s in proposed}, "backend": jax.default_backend(),
           "device": jax.devices()[0].device_kind, "tests": {}}
    for k in hists["stock"]:
        a, b = hists["stock"][k].astype(float), hists["envelope"][k].astype(float)
        keep = (a + b) > 0
        a, b = a[keep], b[keep]
        pull = (a - b * a.sum() / b.sum()) / np.sqrt(a + b * (a.sum() / b.sum()) ** 2)
        chi2 = float((pull ** 2).sum())
        ndf = int(keep.sum()) - 1
        top = np.argsort(-np.abs(pull))[:3]
        if k == "z_pT":
            cz, cp = (z2[:-1] + z2[1:]) / 2, (pt2[:-1] + pt2[1:]) / 2
            centers = [f"z {cz[j // len(cp)]:.3g} pT {cp[j % len(cp)]:.3g}" for j in np.flatnonzero(keep)]
        else:
            centers = [float(c) for c in ((edges[k][:-1] + edges[k][1:]) / 2)[keep]]
        out["tests"][k] = {"chi2": chi2, "ndf": ndf, "p": float(stats.chi2.sf(chi2, ndf)),
                           "top_pulls": [[centers[j], float(pull[j])] for j in top]}
        print(f"{k:6s}: chi2 {chi2:7.1f} / {ndf}, p {out['tests'][k]['p']:.3g}; largest pulls "
              + ", ".join(f"{pull[j]:+.1f} at " + (centers[j] if isinstance(centers[j], str) else f"{centers[j]:.4g}")
                          for j in top))
    print(f"over bar: {over}; proposals per parent: {out['proposals_per_parent']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
