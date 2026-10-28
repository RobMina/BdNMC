import math

import jax.numpy as jnp
import numpy as np
import pytest

import bdnmc_jax  # noqa: F401  (enables float64)
from bdnmc_jax import detector


def rays(n, seed=1, spread=0.02):
    rng = np.random.default_rng(seed)
    p = np.column_stack([rng.normal(0, spread, n), rng.normal(0, spread, n), np.ones(n)]) * rng.uniform(0.1, 5, (n, 1))
    return jnp.zeros((n, 3)), jnp.asarray(p)


def test_sphere_central_chord():
    det = detector.Sphere((0.0, 0.0, 100.0), 5.0)
    L, t0, t1 = detector.ldet_sphere(det, jnp.zeros((1, 3)), jnp.asarray([[0.0, 0.0, 2.0]]))
    assert float(L[0]) == pytest.approx(10.0, rel=1e-14)
    assert float(t0[0]) * 2.0 == pytest.approx(95.0, rel=1e-14)


def test_sphere_matches_independent_chord():
    det = detector.Sphere((0.0, -1.9, 491.0), 5.0)
    o, p = rays(20000, spread=0.03)
    L = np.asarray(detector.ldet_sphere(det, o, p)[0])
    u = np.asarray(p) / np.linalg.norm(np.asarray(p), axis=1, keepdims=True)
    c = np.asarray(det.center)
    b = u @ c
    d2 = c @ c - b ** 2
    ref = np.where(d2 < 25.0, 2 * np.sqrt(np.clip(25.0 - d2, 0, None)), 0.0)
    assert (L > 0).sum() > 100
    np.testing.assert_allclose(L, ref, rtol=1e-9, atol=1e-9)


def slab(center, half, o, p):
    """Textbook slab test for an axis-aligned box, independent of the stock algorithm."""
    lo, hi = np.asarray(center) - half, np.asarray(center) + half
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (lo - o) / p, (hi - o) / p
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    tmin = np.maximum(tmin, 0)
    return np.where(tmax > tmin, (tmax - tmin) * np.linalg.norm(p, axis=1), 0.0)


def test_unrotated_cuboid_matches_slab():
    det = detector.make_cuboid(0.0, -2.0, 500.0, 3.0, 3.0, 10.0, 0.0, 0.0, 0.0)
    o, p = rays(20000, spread=0.01)
    L = np.asarray(detector.ldet_cuboid(det, o, p)[0])
    ref = slab((0.0, -2.0, 500.0), np.array([1.5, 1.5, 5.0]), np.asarray(o), np.asarray(p))
    assert (L > 0).sum() > 100
    np.testing.assert_allclose(L, ref, rtol=1e-9, atol=1e-9)


def test_rotation_by_quarter_turn_swaps_axes():
    # det-theta = pi/2 rotates z into x: a 10 m long box along z becomes 10 m along x.
    det = detector.make_cuboid(0.0, 0.0, 100.0, 3.0, 3.0, 10.0, 0.0, math.pi / 2, 0.0)
    L = detector.ldet_cuboid(det, jnp.zeros((1, 3)), jnp.asarray([[0.0, 0.0, 1.0]]))[0]
    assert float(L[0]) == pytest.approx(3.0, rel=1e-9)
