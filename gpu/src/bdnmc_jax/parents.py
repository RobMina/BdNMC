"""Parent-meson momentum distributions (stock src/sanfordwang.cpp, src/BMPT_dist.cpp).

Stock samples both by rejection, and the bar differs:
- Sanford-Wang uses a fixed bar, `fpi0max = 54.25`, unless a distribution-parameter file is given (then it is
  burned in). Values above the bar are accepted with probability 1, so stock's sampled density is
  min(f, 54.25). The port uses the same bar, so it reproduces that density; `rejection` reports how often f
  exceeds it.
- BMPT keeps a running maximum (10 000 burn-in draws, then raised whenever exceeded), so its early draws are
  slightly clipped. The port uses a fixed bar from a grid search of the density times a safety margin and fails if
  any proposal exceeds it, i.e. it samples the unclipped density. Its theta proposal (shrink theta by a uniform
  factor until xR <= 1) is stock's and is kept.

Each sampler returns momenta (n, 3) = (px, py, pz); parents start at the origin with t = 0.
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from .constants import MASS_PROTON as mp, MASS_PION_CHARGED, pi
from . import rng


def polar(mom, th, ph):
    """Particle::ThreeMomentumPolar."""
    return jnp.stack([mom * jnp.cos(ph) * jnp.sin(th), mom * jnp.sin(ph) * jnp.sin(th), mom * jnp.cos(th)], axis=-1)


# --- Sanford-Wang ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SanfordWang:
    """Stock defaults (sanfordwang.h); the card's beam_energy does not enter."""
    pB: float = 8.89
    ppimax: float = 7.8900
    fmax: float = 54.25
    c: tuple = (0, 220.7000, 1.080000, 1.000000, 1.978000, 1.320000, 5.572000, 0.086800, 9.686000, 1.000000)
    d: tuple = (0, 213.7000, 0.937900, 5.454000, 1.210000, 1.284000, 4.781000, 0.073380, 8.329000, 1.000000)

    def _sw(self, c, p, th):
        pB = self.pB
        return c[1] * p ** c[2] * (1 - p / (pB - c[9])) * jnp.exp(
            -c[3] * p ** c[4] / pB ** c[5] - c[6] * th * (p - c[7] * pB * jnp.cos(th) ** c[8]))

    def density(self, p, th):
        """sin(theta) (swpim + swpip), the quantity pi0sample compares with fpi0max."""
        return jnp.sin(th) * (self._sw(self.d, p, th) + self._sw(self.c, p, th))

    def propose(self, key, pool):
        u = jax.random.uniform(key, (pool, 2), dtype=jnp.float64)
        th = 0 + (pi / 2.0 - 0) * u[:, 0]
        p = 0 + (self.ppimax - 0) * u[:, 1]
        return jnp.stack([p, th], axis=-1), self.density(p, th)


# --- BMPT --------------------------------------------------------------------------------------------------------

_A, _B, _alpha, _beta, _a, _gamma, _delta, _r0, _r1, _MXBAR = 62.3, 1.57, 3.45, 0.517, 6.10, 0.153, 0.478, 1.05, 2.65, 1.88
_BE_MASS_NUMBER = 8.0


@dataclass(frozen=True)
class BMPT:
    beam_energy: float
    mass_number: float
    margin: float = 1.02
    max_shrinks: int = 400

    @property
    def meson_mass(self):
        return MASS_PION_CHARGED  # stock uses the charged-pion mass for the pi0 choice

    @property
    def s(self):
        return 2 * (mp * mp + mp * self.beam_energy)

    @property
    def beta_cm(self):
        return math.sqrt(self.beam_energy ** 2 - mp * mp) / (self.beam_energy + mp)

    @property
    def p_max(self):
        return self.beam_energy - 2 * mp - self.meson_mass

    theta_max = pi / 2.0

    def xR(self, p, th):
        E, M, s = self.beam_energy, self.meson_mass, self.s
        return (s - (2 * mp * mp + M ** 2 + 2 * E * mp - 2 * E * jnp.sqrt(p * p + M ** 2)
                     + 2 * math.sqrt(E ** 2 - mp * mp) * p * jnp.cos(th) - 2 * jnp.sqrt(p * p + M ** 2) * mp)
                + M ** 2) / (s - _MXBAR ** 2 + M ** 2)

    def xF(self, p, th):
        b = self.beta_cm
        g = 1.0 / math.sqrt(1 - b ** 2)
        return g * (p * jnp.cos(th) - b * jnp.sqrt(p * p + self.meson_mass ** 2)) / math.sqrt(self.s)

    def ics(self, p, th):
        """BMPT::Invariant_Cross_Section."""
        x = self.xR(p, th)
        pT = p * jnp.sin(th)
        ap = _a / x ** _gamma
        bp = _a * _a / (2 * x ** _delta)
        xf = self.xF(p, th)
        alphaf = (0.74 - 0.55 * xf + 0.26 * xf ** 2) * (0.98 + 0.21 * pT)
        return (_A * (1.0 - x) ** _alpha * (1.0 + _B * x) * x ** (-1.0 * _beta) * (1 + ap * pT + bp * pT * pT)
                * jnp.exp(-ap * pT) * (self.mass_number / _BE_MASS_NUMBER) ** alphaf)

    def density(self, p, th):
        """The `prob` of BMPT::sample_momentum for the pi0 choice."""
        M = self.meson_mass
        pratio = _r0 * (1 + self.xR(p, th)) ** _r1
        return 2 * pi * jnp.sin(th) * p ** 2 / jnp.sqrt(p * p + M ** 2) * self.ics(p, th) * (1.0 + 1.0 / pratio)

    def propose(self, key, pool):
        k0, k1, k2 = jax.random.split(key, 3)
        p = jax.random.uniform(k0, (pool,), dtype=jnp.float64) * self.p_max
        th = jax.random.uniform(k1, (pool,), dtype=jnp.float64) * self.theta_max

        def cond(st):
            th, i = st
            return jnp.any(self.xR(p, th) > 1) & (i < self.max_shrinks)

        def body(st):
            th, i = st
            u = jax.random.uniform(jax.random.fold_in(k2, i), (pool,), dtype=jnp.float64)
            return jnp.where(self.xR(p, th) > 1, th * u, th), i + 1

        th, _ = jax.lax.while_loop(cond, body, (th, 0))
        f = self.density(p, th)
        # A proposal still at xR > 1 after max_shrinks gets f = inf, which `rejection` counts as over the bar.
        f = jnp.where(self.xR(p, th) > 1, jnp.inf, f)
        return jnp.stack([p, th], axis=-1), f

    def find_fmax(self, n_p=2000, n_th=2000):
        """Grid maximum of the density over p in (0, p_max], theta in (0, pi/2] with xR <= 1, times `margin`."""
        p = np.linspace(self.p_max / n_p, self.p_max, n_p)
        best = 0.0
        lo, hi = 1e-9, self.theta_max
        for _ in range(3):  # zoom the theta grid towards the peak: BMPT peaks at small angles
            th = np.geomspace(lo, hi, n_th)
            P, T = np.meshgrid(p, th, indexing="ij")
            f = np.asarray(self.density(jnp.asarray(P), jnp.asarray(T)))
            f = np.where((np.asarray(self.xR(jnp.asarray(P), jnp.asarray(T))) <= 1) & np.isfinite(f), f, 0.0)
            i, j = np.unravel_index(np.argmax(f), f.shape)
            best = max(best, f[i, j])
            lo, hi = th[max(j - 20, 0)], th[min(j + 20, n_th - 1)]
            p = np.linspace(p[max(i - 20, 0)], p[min(i + 20, n_p - 1)], n_p)
        return best * self.margin


# --- driver ------------------------------------------------------------------------------------------------------

def sample_block(dist, key, n, fmax, m_meson, pool=None):
    """n parent four-momenta (n, 4) for one block, with diagnostics."""
    pool = pool or n
    x, filled, nprop, nover = rng.rejection(jax.random.fold_in(key, rng.PARENT), dist.propose, fmax, n, pool)
    ph = 2 * pi * rng.uniform(key, rng.PARENT_PHI, (n,))  # Flat(0, 2 pi), drawn after acceptance in stock
    mom = polar(x[:, 0], x[:, 1], ph)
    E = jnp.sqrt(m_meson * m_meson + mom[:, 0] ** 2 + mom[:, 1] ** 2 + mom[:, 2] ** 2)
    return jnp.concatenate([E[:, None], mom], axis=-1), {"filled": filled, "proposed": nprop, "over_bar": nover}
