"""The envelope parent sampler samples the same law as stock's proposal (BMPT and Sanford-Wang)."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import bdnmc_jax  # noqa: F401
from bdnmc_jax import card as cardmod, parents, pipeline, rng

CARDS = Path(__file__).resolve().parents[1] / "phase0" / "cards" / "ref"


def _dist(stem, sampler):
    return pipeline.setup(cardmod.parse_card(CARDS / f"{stem}.dat"), sampler)[2][0]


def test_bmpt_theta_star_is_where_xr_reaches_one():
    d = _dist("he_pi0_cuboid_m005", "stock").dist
    p = jnp.linspace(0.5, d.p_max * 0.999, 200)
    th = d.theta_of(p, 1.0)
    inside = th < d.theta_max
    assert inside.any()
    np.testing.assert_allclose(np.asarray(d.xR(p, th))[inside], 1.0, rtol=0, atol=1e-9)
    assert (np.asarray(d.xR(p, th * 0.999)) <= 1).all()
    assert (np.asarray(d.xR(p, jnp.minimum(th * 1.001, d.theta_max)))[inside] > 1).all()


@pytest.mark.parametrize("stem", ["mb_pi0_m005", "he_pi0_cuboid_m005"])
def test_envelope_samples_the_stock_law(stem):
    stock, env = _dist(stem, "stock"), _dist(stem, "envelope")
    assert isinstance(env.dist, parents.Envelope) and env.dist.efficiency > 0.5
    n = rng.BLOCK
    draws = {}
    for name, ch, seed in (("stock", stock, 1), ("envelope", env, 2)):
        blocks = [parents.sample_block(ch.dist, rng.block_key(seed, 0, b), n, ch.fmax, ch.m_meson) for b in range(2)]
        for _, info in blocks:
            assert int(info["filled"]) == n
            if name == "envelope":
                assert int(info["over_bar"]) == 0
        par = np.concatenate([np.asarray(p) for p, _ in blocks])
        mom = np.linalg.norm(par[:, 1:], axis=1)
        draws[name] = {"p": mom, "theta": np.arccos(par[:, 3] / mom), "pT": np.hypot(par[:, 1], par[:, 2])}
    for var in ("p", "theta", "pT"):
        assert stats.ks_2samp(draws["stock"][var], draws["envelope"][var]).pvalue > 1e-3, var


def test_envelope_bound_violation_is_reported():
    env = _dist("mb_pi0_m005", "envelope").dist
    tight = parents.Envelope(env.base, env.digest + "-tight", env.efficiency, env.p_lo, env.dp, env.v_lo, env.dv,
                             env.bound / 2, env.cdf)
    _, info = parents.sample_block(tight, jax.random.key(0), 1024, 1.0, 0.135)
    assert int(info["over_bar"]) > 0
