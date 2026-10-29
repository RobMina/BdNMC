"""High-statistics check that the envelope parent sampler draws the same law as the stock-proposal sampler: N
blocks of parents from each (independent seeds), histograms of p, theta (log bins), pT and, for BMPT, v = theta /
theta*(p), with a two-sample chi2 per histogram and the largest bin pulls.

Usage: compare_parent_samplers.py <card> [--blocks N] [--batch B] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import card as cardmod, parents, pipeline, rng  # noqa: E402


def sampler(ch, seed, batch):
    @jax.jit
    def call(b0):
        par, info = jax.vmap(lambda b: parents.sample_block(ch.dist, rng.block_key(seed, 0, b), rng.BLOCK, ch.fmax,
                                                            ch.m_meson))(b0 + jnp.arange(batch))
        par = par.reshape(-1, 4)
        mom = jnp.linalg.norm(par[:, 1:], axis=1)
        return {"p": mom, "theta": jnp.arccos(par[:, 3] / mom), "pT": jnp.hypot(par[:, 1], par[:, 2])}, info
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
    edges = {"p": np.linspace(0, base.p_range, 201), "theta": np.concatenate([[0], np.geomspace(1e-5, np.pi / 2, 200)]),
             "pT": None, "v": np.linspace(0, 1, 201)}
    hists, over = {}, {}
    for s, ch in chans.items():
        call = sampler(ch, 1000 + len(hists), args.batch)
        h, n_over = {}, 0
        for i in range(0, args.blocks, args.batch):
            d, info = jax.device_get(call(i))
            assert (info["filled"] == rng.BLOCK).all()
            n_over += int(info["over_bar"].sum())
            if isinstance(base, parents.BMPT):
                d["v"] = d["theta"] / np.asarray(base.theta_of(jnp.asarray(d["p"]), 1.0))
            if edges["pT"] is None:
                edges["pT"] = np.linspace(0, np.quantile(d["pT"], 0.9999), 201)
            for k, e in edges.items():
                if k in d:
                    h[k] = h.get(k, 0) + np.histogram(d[k], e)[0]
        hists[s], over[s] = h, n_over
    out = {"card": str(args.card), "dist": type(base).__name__, "parents_per_sampler": args.blocks * rng.BLOCK,
           "over_bar": over, "backend": jax.default_backend(), "tests": {}}
    for k in hists["stock"]:
        a, b = hists["stock"][k].astype(float), hists["envelope"][k].astype(float)
        keep = (a + b) > 0
        a, b = a[keep], b[keep]
        pull = (a - b * a.sum() / b.sum()) / np.sqrt(a + b * (a.sum() / b.sum()) ** 2)
        chi2 = float((pull ** 2).sum())
        ndf = int(keep.sum()) - 1
        top = np.argsort(-np.abs(pull))[:3]
        centers = (edges[k][:-1] + edges[k][1:])[keep] / 2
        out["tests"][k] = {"chi2": chi2, "ndf": ndf, "p": float(stats.chi2.sf(chi2, ndf)),
                           "top_pulls": [[float(centers[j]), float(pull[j])] for j in top]}
        print(f"{k:6s}: chi2 {chi2:7.1f} / {ndf}, p {out['tests'][k]['p']:.3g}; largest pulls "
              + ", ".join(f"{pull[j]:+.1f} at {centers[j]:.4g}" for j in top))
    print(f"over bar: {over}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
