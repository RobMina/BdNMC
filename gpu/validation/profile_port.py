"""Where pass 1 spends its time on one card: the parent sampler alone against the whole block (per batch of
blocks, steady state), the parent rejection efficiency (rounds of 2^16 proposals per block), and for BMPT the
number of theta-shrink iterations per round.

Usage: profile_port.py <card> [--batch 1 16 64] [--calls N] [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bdnmc_jax import brem, card as cardmod, parents, pipeline, rng  # noqa: E402


def timed(f, n_calls):
    jax.block_until_ready(f(0))
    t = time.perf_counter()
    jax.block_until_ready([f(i) for i in range(1, n_calls + 1)])
    return (time.perf_counter() - t) / n_calls


def shrink_iterations(dist, key, pool, n_rounds):
    """Iterations of BMPT's theta-shrink loop per round of `pool` proposals (the loop runs until every proposal
    has xR <= 1, so a round costs its slowest proposal)."""
    @jax.jit
    def one(k):
        k0, k1, k2 = jax.random.split(k, 3)
        p = jax.random.uniform(k0, (pool,), dtype=jnp.float64) * dist.p_max
        th = jax.random.uniform(k1, (pool,), dtype=jnp.float64) * dist.theta_max
        per = jnp.zeros(pool, dtype=jnp.int32)

        def cond(st):
            th, per, i = st
            return jnp.any(dist.xR(p, th) > 1) & (i < dist.max_shrinks)

        def body(st):
            th, per, i = st
            bad = dist.xR(p, th) > 1
            u = jax.random.uniform(jax.random.fold_in(k2, i), (pool,), dtype=jnp.float64)
            return jnp.where(bad, th * u, th), per + bad, i + 1

        _, per, i = jax.lax.while_loop(cond, body, (th, per, 0))
        return i, jnp.mean(per), jnp.mean(per > 0)
    out = [jax.device_get(one(jax.random.fold_in(key, r))) for r in range(n_rounds)]
    return {"iterations_per_round": [int(o[0]) for o in out], "mean_shrinks_per_proposal": float(np.mean([o[1] for o in out])),
            "fraction_needing_a_shrink": float(np.mean([o[2] for o in out]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", type=Path)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    ap.add_argument("--calls", type=int, default=10)
    ap.add_argument("--parent-sampler", default="stock", choices=pipeline.PARENT_SAMPLERS)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    card = cardmod.parse_card(args.card)
    det, sc, (ch,) = pipeline.setup(card, args.parent_sampler)
    n = rng.BLOCK
    out = {"card": str(args.card), "backend": jax.default_backend(), "device": jax.devices()[0].device_kind,
           "dist": type(ch.dist).__name__, "parent_sampler": args.parent_sampler, "runs": []}

    def one_block(b):
        key = rng.block_key(card.seed, 0, b)
        if ch.kind == "brem":
            return brem.sample_block(ch.dist, key, n, ch.fmax, ch.m_v)
        return parents.sample_block(ch.dist, key, n, ch.fmax, ch.m_meson)

    @jax.jit
    def parents_only(bi):
        return jax.vmap(one_block)(bi)

    for b in args.batch:
        bi = lambda i: jnp.arange(i * b, (i + 1) * b)  # noqa: E731
        info = jax.device_get(parents_only(bi(0))[1])
        t_par = timed(lambda i: parents_only(bi(i)), args.calls)
        nv = jnp.full(b, n)
        t_blk = timed(lambda i: pipeline.blocks_batch(det, sc, ch, card.e_density, card.seed, 0, bi(i), nv, 1024
                                                      if isinstance(ch.dist, parents.SanfordWang) else 2048),
                      args.calls)
        r = {"batch": b, "parents_s_per_call": t_par, "block_s_per_call": t_blk, "parent_share": t_par / t_blk,
             "block_trials_per_s": b * n / t_blk,
             "proposals_per_accepted": float(np.mean(info["proposed"]) / n),
             "rounds_per_block_max": int(np.max(info["proposed"]) // n)}
        out["runs"].append(r)
        print(f"batch {b:3d}: block {t_blk * 1e3:8.1f} ms/call ({r['block_trials_per_s'] / 1e6:.2f} M trials/s), "
              f"parents {t_par * 1e3:8.1f} ms ({100 * r['parent_share']:.0f} %); "
              f"{r['proposals_per_accepted']:.1f} proposals per parent, up to {r['rounds_per_block_max']} rounds/block",
              flush=True)
    if isinstance(ch.dist, parents.BMPT):
        out["shrink"] = shrink_iterations(ch.dist, jax.random.key(1), n, 20)
        s = out["shrink"]
        print(f"BMPT shrink loop: iterations per round {min(s['iterations_per_round'])}-{max(s['iterations_per_round'])}, "
              f"{s['mean_shrinks_per_proposal']:.2f} shrinks per proposal, "
              f"{100 * s['fraction_needing_a_shrink']:.1f} % of proposals need one")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
