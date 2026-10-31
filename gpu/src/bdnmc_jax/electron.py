"""DM-electron elastic scattering, `NCE_electron` (stock src/DMscattering.cpp and src/Electron_Scatter.cpp).

Elementwise in jax.numpy; expressions keep the stock operation order so float64 results can be compared at the
1e-12 level (V0). Energies are total electron energies. The scatter-energy window comes from the card's
`min_scatter_energy`/`max_scatter_energy` as total energies: BdNMCSession calls `set_energy_limits`, which
overwrites the `+Me` the Electron_Scatter constructor adds.
"""
from dataclasses import dataclass, fields
import math

import jax
import jax.numpy as jnp

from .constants import MASS_ELECTRON as Me, alphaEM, convGeV2cm2, convmcm, pi

COS_HALF_PI = math.cos(pi / 2.0)  # EeTMin uses cos(Pi/2.0) = 6.1e-17, not 0


@dataclass(frozen=True)
class ElectronScatter:
    m_dm: float
    m_v: float
    kappa: float
    alpha_d: float
    e_min: float  # total electron energy window
    e_max: float
    min_angle: float = 0.0
    max_angle: float = 2.1 * pi


# Every field is a traced leaf under jit, so a new mass or coupling does not recompile (goal P2).
jax.tree_util.register_dataclass(ElectronScatter, data_fields=[f.name for f in fields(ElectronScatter)],
                                 meta_fields=[])


def ee_theta(E, cos_t, m):
    """EeTheta: electron energy for scattering angle theta (passed as cos theta)."""
    c2 = cos_t * cos_t
    num = Me * ((E + Me) * (E + Me) + (E * E - m * m) * c2)
    den = (E + Me) * (E + Me) - (E * E - m * m) * c2
    return num / den


def ee_tmax(E, m):
    return ee_theta(E, 1.0, m)


def ee_tmin(E, m):
    return ee_theta(E, COS_HALF_PI, m)


def theta_ee(Ee, E, m):
    """ThetaEe: electron lab angle for recoil energy Ee."""
    x = jnp.sqrt((Ee - Me) / (Ee + Me)) * (E + Me) / jnp.sqrt(E * E - m * m)
    if x.dtype == jnp.float32:
        # The argument is at most 1 (Ee <= EeTMax), but float32 rounding near the endpoint takes it above 1 often
        # enough to fail most runs; clipped, those events get the endpoint angle 0 exactly. float64 is not clipped:
        # there, as in stock (which writes NaN), it is rare, and the pipeline fails on it.
        x = jnp.minimum(x, 1.0)
    return jnp.arccos(x)


def F2(Ee, E, m, mv):
    return -((4 * E * E * Me * Me + 2 * E * Me * mv * mv + mv * mv * m * m) / (2 * Me * (Ee - Me) + mv * mv)
             + (2 * E * Me + m * m) * jnp.log(2 * Me * (Ee - Me) + mv * mv)) / (Me * Me * (E - m) * (E + m))


def sigma2(E, m, mv, kappa, alpha_d, emax, emin):
    if jnp.result_type(E, emax, emin) == jnp.float32:
        diff = _F2_difference(emax, emin, E, m, mv)
    else:
        diff = F2(emax, E, m, mv) - F2(emin, E, m, mv)
    s = pi * kappa * kappa * alphaEM * alpha_d * diff
    return jnp.where(emax < emin, 0.0, s)


def _F2_difference(hi, lo, E, m, mv):
    """F2(hi) - F2(lo) for float32 without subtracting the two large F2 values: with x = 2 Me (Ee - Me) + mv^2, the
    1/x terms differ by -2 Me (hi - lo) / (x_hi x_lo) and the logs by log1p(2 Me (hi - lo) / x_lo). Stock's
    difference cancels to ~100 % error in float32 at narrow windows."""
    x_hi, x_lo = 2 * Me * (hi - Me) + mv * mv, 2 * Me * (lo - Me) + mv * mv
    a = 4 * E * E * Me * Me + 2 * E * Me * mv * mv + mv * mv * m * m
    d = 2 * Me * (hi - lo)
    return -(a * (-d / (x_hi * x_lo)) + (2 * E * Me + m * m) * jnp.log1p(d / x_lo)) / (Me * Me * (E - m) * (E + m))


def dsigma_dEe(Ee, E, m, mv, kappa, alpha_d):
    d = 2 * Me * (Ee - Me) + mv * mv
    return (4 * pi * kappa * kappa * alphaEM * alpha_d * (2 * E * Me * (-Ee + E + Me) + m * m * (Me - Ee))
            / (E - m) / (E + m) / (d * d))


def scat_max(sc, E):
    return jnp.minimum(sc.e_max, ee_tmax(E, sc.m_dm))


def scat_min(sc, E):
    return jnp.maximum(sc.e_min, ee_tmin(E, sc.m_dm))


def scatter_probability(sc, E, L, e_density):
    """Electron_Scatter::probscatter `prob`: sigma2 x ENtot x Ldet[cm] x convGeV2cm2; non-finite or < 0 -> 0."""
    lx = L * convmcm
    prob = sigma2(E, sc.m_dm, sc.m_v, sc.kappa, sc.alpha_d, scat_max(sc, E), scat_min(sc, E)) * e_density * lx \
        * convGeV2cm2
    return jnp.where(jnp.isfinite(prob) & (prob >= 0), prob, 0.0)
