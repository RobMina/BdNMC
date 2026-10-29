"""V0: the port against stock C++ function values (v0/v0_dump.txt, written by v0/v0_dump.cpp)."""
from collections import defaultdict
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import branching, brem, detector, electron, parents
from bdnmc_jax.constants import MASS_ELECTRON as Me

DUMP = Path(__file__).resolve().parents[1] / "v0" / "v0_dump.txt"
REL = 1e-12  # closed-form functions
LDET_ABS = 1e-9  # metres
M, MV, KAP, ALD = 0.005, 0.1, 1e-3, 0.1


@pytest.fixture(scope="module")
def dump():
    if not DUMP.exists():
        pytest.skip("v0/v0_dump.txt not built yet (see v0/README.md)")
    rows = defaultdict(list)
    for line in DUMP.read_text().splitlines():
        name, *vals = line.split()
        rows[name].append([float(v) for v in vals])
    return {k: np.array(v) for k, v in rows.items()}


EPS = np.finfo(float).eps


def close(got, ref, cond=1.0, rel=REL):
    """|got - ref| <= (rel + 64 eps cond) |ref|, with cond the condition number of the stock expression at that
    input. Stock is compiled with clang's default FMA contraction and XLA rounds differently, so where an expression
    cancels (cond >> 1) the two codes legitimately differ by about eps x cond; elsewhere the bound is `rel`."""
    got, ref = np.asarray(got), np.asarray(ref)
    both_nan = np.isnan(got) & np.isnan(ref)
    got, ref = np.where(both_nan, 0, got), np.where(both_nan, 0, ref)
    tol = (rel + 64 * EPS * np.broadcast_to(cond, ref.shape)) * np.abs(ref)
    bad = ~(np.abs(got - ref) <= tol)
    assert not bad.any(), (f"{bad.sum()} of {bad.size} beyond tolerance; worst rel diff "
                           f"{np.max(np.abs(got - ref)[bad] / np.abs(ref[bad])):.3g} at cond {np.asarray(cond)[bad].max() if np.ndim(cond) else cond:.3g}")


def cond_ee_theta(E, c):
    """EeTheta's denominator (E+Me)^2 - (E^2-m^2) cos^2 cancels at high E."""
    a, b = (E + Me) ** 2, (E * E - M * M) * c * c
    return (a + b) / np.abs(a - b)


def test_electron_kinematics(dump):
    E = dump["EeTMax"][:, 0]
    close(electron.ee_tmax(jnp.asarray(E), M), dump["EeTMax"][:, 1], cond_ee_theta(E, 1.0))
    close(electron.ee_tmin(jnp.asarray(E), M), dump["EeTMin"][:, 1], cond_ee_theta(E, electron.COS_HALF_PI))
    t = dump["ThetaEe"]
    got = np.asarray(electron.theta_ee(jnp.asarray(t[:, 0]), jnp.asarray(t[:, 1]), M))
    # At Ee = EeTMax(E) the acos argument is 1 within an ulp: stock itself gives 0 or NaN there, the port 0,
    # ~1.5e-8 (acos(1 - eps)) or NaN. Both codes share that hazard; at the endpoint only require that each value
    # is one of those rounding outcomes.
    stock_tmax = dict(zip(dump["EeTMax"][:, 0], dump["EeTMax"][:, 1]))
    end = np.isclose(t[:, 0], np.array([stock_tmax[e] for e in t[:, 1]]), rtol=1e-14, atol=0)
    edge = lambda v: np.isnan(v) | (np.abs(v) <= 3e-8)
    assert edge(got[end]).all() and edge(t[end, 2]).all()
    x = np.sqrt((t[:, 0] - Me) / (t[:, 0] + Me)) * (t[:, 1] + Me) / np.sqrt(t[:, 1] ** 2 - M * M)
    with np.errstate(divide="ignore", invalid="ignore"):  # acos is ill-conditioned near x = 1
        cond = np.abs(x) / (np.sqrt(1 - x * x) * np.arccos(x))
    close(got[~end], t[~end, 2], np.nan_to_num(cond[~end], nan=1e16, posinf=1e16))


def test_sigma2_and_dsigma(dump):
    s = dump["sigma2"]
    hi = np.asarray(electron.F2(jnp.asarray(s[:, 1]), jnp.asarray(s[:, 0]), M, MV))
    lo = np.asarray(electron.F2(jnp.asarray(s[:, 2]), jnp.asarray(s[:, 0]), M, MV))
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.nan_to_num((np.abs(hi) + np.abs(lo)) / np.abs(hi - lo), nan=1.0, posinf=1e16)
    close(electron.sigma2(jnp.asarray(s[:, 0]), M, MV, KAP, ALD, jnp.asarray(s[:, 1]), jnp.asarray(s[:, 2])),
          s[:, 3], cond)
    d = dump["dsigmadEe"]
    Ee, E = d[:, 0], d[:, 1]
    num = 2 * E * Me * (-Ee + E + Me) + M * M * (Me - Ee)
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = np.nan_to_num((2 * E * Me * (Ee + E + Me) + M * M * (Me + Ee)) / np.abs(num), posinf=1e16)
    close(electron.dsigma_dEe(jnp.asarray(Ee), jnp.asarray(E), M, MV, KAP, ALD), d[:, 2], cond)


@pytest.mark.parametrize("name,det", [
    ("Ldet_sphere", detector.Sphere((0.0, -1.9, 491.0), 5.0)),
    ("Ldet_cuboid", detector.make_cuboid(0.0, -2.0, 500.0, 3.0, 3.0, 10.0, 0.0, 0.0, 0.0)),
    ("Ldet_rotated", detector.make_cuboid(0.5, -2.0, 500.0, 3.0, 4.0, 10.0, 0.3, 0.05, -0.2)),
])
def test_ldet(dump, name, det):
    r = dump[name]
    L = np.asarray(detector.ldet(det, jnp.zeros((len(r), 3)), jnp.asarray(r[:, :3]))[0])
    assert ((L > 0) == (r[:, 3] > 0)).all(), "hit/miss disagrees"
    np.testing.assert_allclose(L, r[:, 3], rtol=0, atol=LDET_ABS)


def test_branching_ratios(dump):
    for mx, br in dump["brpi0_to_gamma_dm_dm"]:
        assert branching.brpi0_to_gamma_dm_dm(MV, mx, KAP, ALD) == pytest.approx(br, rel=REL)
    for mx, br in dump["brpi0toVgamma"]:
        assert branching.brpi0toVgamma(MV, mx, KAP, ALD) == pytest.approx(br, rel=REL)
    for mx, br in dump["brV_to_dm_dm"]:
        assert branching.brV_to_dm_dm(MV, mx, KAP, ALD) == pytest.approx(br, rel=REL)


def _cond_sum(*terms):
    """Condition number of a sum: sum of |terms| / |sum|."""
    terms = [np.asarray(t, float) for t in terms]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(sum(np.abs(t) for t in terms) / np.abs(sum(terms)), nan=1e16, posinf=1e16)


def _wpp_cond(z, pt2, mA):
    mp = 0.938272
    H = pt2 + (1 - z) * mA * mA + (z * mp) ** 2
    return 50 + _cond_sum((1 + (1 - z) ** 2) / z, -2 * z * (1 - z) * (2 * mp * mp + mA * mA) / H,
                          4 * z * (1 - z) * (z * mp * mp / H) ** 2, 2 * z * (1 - z) * (z + (1 - z) ** 2) * (mp * mA / H) ** 2,
                          2 * z * ((1 - z) * mA * mA / H) ** 2)


def test_proton_brem(dump):
    if "wpp" not in dump:
        pytest.skip("v0_dump.txt predates the proton-brem entries (rebuild the harness)")
    w = dump["wpp"]
    close(brem.wpp(w[:, 0], w[:, 1], w[:, 2]), w[:, 3], _wpp_cond(w[:, 0], w[:, 1], w[:, 2]))
    d = dump["d2N"]
    E, mA, z, pt = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    mp = 0.938272
    EV = np.sqrt(mA * mA + pt + z * z * (E * E - mp * mp))
    # sigma_pp(s') with s' = 2 mp (E - E_V): the difference cancels as z -> 1; log^2 and the powers add a few units.
    cond = _wpp_cond(z, pt, mA) + 50 * (1 + E / np.abs(E - EV))
    got = np.array([float(brem.d2N(zi, pi, Ei, KAP, mi)) for Ei, mi, zi, pi in zip(E, mA, z, pt)])
    close(got, d[:, 4], cond)
    for E_, zmin, zmax, ptmax, mA_, rate in dump["V_prod_rate"]:
        r = brem.ProtonBrem(E_, zmin, zmax, 0.0, ptmax).v_prod_rate(KAP, mA_)
        # 40,401 positive terms summed in a different order (numpy pairwise against stock's loops)
        assert r == pytest.approx(rate, rel=1e-10)


def test_parent_densities(dump):
    sw = parents.SanfordWang()
    s = dump["sanfordwang"]
    # (1 - p / (pB - c9)) is exactly 0 at p = ppimax, leaving only rounding; the exponent (up to ~100) adds
    # |argument| to the conditioning.
    with np.errstate(divide="ignore"):
        cond = np.nan_to_num(1 / np.abs(1 - s[:, 0] / (sw.pB - sw.c[9])), posinf=1e16) + 200
    close(sw.density(jnp.asarray(s[:, 0]), jnp.asarray(s[:, 1])), s[:, 2], cond)
    bm = parents.BMPT(120.0, 12.0)
    b = dump["bmpt_ics"]
    p, th = b[:, 0], b[:, 1]
    # xR's numerator cancels large terms, and (1 - xR)^alpha amplifies its error by alpha xR / (1 - xR).
    E, Mm = bm.beam_energy, bm.meson_mass
    Ep = np.sqrt(p * p + Mm ** 2)
    mp = 0.938272
    terms = (bm.s + 2 * mp * mp + Mm ** 2 + 2 * E * mp + 2 * E * Ep + 2 * np.sqrt(E ** 2 - mp * mp) * p * np.abs(np.cos(th))
             + 2 * Ep * mp + Mm ** 2)
    xr = np.asarray(bm.xR(jnp.asarray(p), jnp.asarray(th)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cond_xr = terms / np.abs(xr * (bm.s - 1.88 ** 2 + Mm ** 2))
        cond = 200 + cond_xr * (1 + 3.45 * np.abs(xr) / np.abs(1 - xr))
    close(bm.ics(jnp.asarray(p), jnp.asarray(th)), b[:, 2], np.nan_to_num(cond, nan=1e16, posinf=1e16))
