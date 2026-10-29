"""Counter-based random numbers and a batched rejection sampler.

Every draw is keyed by (seed, channel, block, stage, round), with threefry (JAX's default PRNG, identical on CPU and
GPU). A block is a fixed number of trials, so the numbers a trial sees do not depend on how many blocks one call
processes: results are reproducible for a given seed whatever the batch size or device split (goal P3).
"""
from functools import partial

import jax
import jax.numpy as jnp

BLOCK = 1 << 16  # trials per reproducibility unit; changing it changes every random stream

# Stage ids (fold_in constants); new stages get new ids, existing ids never change meaning.
CHANNEL, PARENT, PARENT_PHI, DECAY, ACCEPT, RECOIL, RECOIL_PHI, POSITION = range(8)


def block_key(seed, channel, block):
    key = jax.random.key(seed)
    return jax.random.fold_in(jax.random.fold_in(key, channel), block)


def uniform(key, stage, shape):
    """U[0, 1) float64 for one stage of one block."""
    return jax.random.uniform(jax.random.fold_in(key, stage), shape, dtype=jnp.float64)


@partial(jax.jit, static_argnames=("propose", "n", "pool", "max_rounds"))
def rejection(key, propose, fmax, n, pool, max_rounds=10_000, params=None):
    """Draw n accepted samples, in proposal order, from `propose(key, pool) -> (x (pool, d), f (pool,))`, or
    `propose(key, pool, params)` when `params` (traced, e.g. a mass) is given.

    A proposal is accepted when f > u * fmax (stock's test). Rounds of `pool` proposals repeat, each with its own
    key, until n are accepted. Returns (x (n, d), filled, n_proposed, n_over) where n_over counts proposals with
    f > fmax: the target is then clipped, and the caller must treat a nonzero count as a failure unless clipping
    is what it means to reproduce (stock Sanford-Wang's fixed bar). The caller must also check filled == n
    (fewer means max_rounds ran out).
    """
    def cond(state):
        _, filled, _, _, r = state
        return (filled < n) & (r < max_rounds)

    def body(state):
        out, filled, nprop, nover, r = state
        kr = jax.random.fold_in(key, r)
        x, f = propose(jax.random.fold_in(kr, 0), pool, *(() if params is None else (params,)))
        u = jax.random.uniform(jax.random.fold_in(kr, 1), (pool,), dtype=jnp.float64)
        acc = f > u * fmax
        idx = jnp.where(acc, filled + jnp.cumsum(acc) - 1, n)
        out = out.at[idx].set(x, mode="drop")
        filled = jnp.minimum(filled + jnp.sum(acc), n)
        return out, filled, nprop + pool, nover + jnp.sum(f > fmax), r + 1

    x0, _ = jax.eval_shape(lambda k: propose(k, pool, *(() if params is None else (params,))), key)
    out = jnp.zeros((n,) + x0.shape[1:], dtype=x0.dtype)
    zero = jnp.zeros((), dtype=jnp.int64)
    out, filled, nprop, nover, _ = jax.lax.while_loop(cond, body, (out, zero, zero, zero, zero))
    return out, filled, nprop, nover
