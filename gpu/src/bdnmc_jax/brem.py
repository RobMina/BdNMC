"""Proton bremsstrahlung of a dark photon, `production_distribution proton_brem` (stock src/Proton_Brem_Distribution.cpp;
wpp in src/branchingratios.cpp).

Stock behaviours kept, because they set the rate and the kinematics:
- `d2N_proton_brem_to_V(z, pt2)` is called with pT, not pT^2, both by the rate integral (`SimpsonCubature` over
  [zmin, zmax] x [ptmin, ptmax]) and by the sampler. The port evaluates the same function at the same arguments.
- the channel's DM count is `V_prod_rate x POT`, with no V -> chi chi branching ratio (Kinetic_Mixing.cpp:195).
- the V carries z x the beam momentum along z and pT transverse (no rotation to the proton direction).
Not kept: stock's sampler bar starts at d2N(zmin, ptmin) and is raised whenever exceeded (a running maximum, like
BMPT's); the port uses a fixed bar from a grid search x a margin and fails if a proposal exceeds it.

The density's constant factors (|F1(m_A^2)|^2, epsilon^2 alpha, 1/sigma_pp(s)) cancel in the accept test, so the
device sampler uses `shape` = sigma_pp(s') x wpp, with m_A traced (no recompile per mass).
"""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import numpy as np

from . import rng
from .constants import MASS_PROTON as mp, alphaEM, pi

# sigma_pp fit (Proton_Brem_Distribution.h)
_Hpp, _Mpp, _eta1pp, _eta2pp, _R1pp, _R2pp, _Ppp = 0.2704, 2.2127, 0.451, 0.549, 12.98, 7.38, 34.49
_sppM = (2 * mp + _Mpp) ** 2

# Timelike proton form factor of arXiv:0910.5589, constants as in Proton_Brem_Distribution.cpp
_mrh, _mome, _mrhop, _mrhopp, _momep, _momepp = 0.77, 0.77, 1.25, 1.45, 1.25, 1.45
_grho, _gome, _grhop, _gomep, _grhopp, _gomepp = 0.15, 0.0085, 0.3, 0.3, 0.5, 0.5
_f1ra, _f1rb, _f1rc = 0.6165340033101271, 0.22320420111672623, -0.33973820442685326
_f1wa, _f1wb, _f1wc = 1.0117544786579074, -0.8816565944110686, 0.3699021157531611


def F1_proton(q2):
    """F_1_proton(q2) = F1r + F1w (complex)."""
    def bw(a, m, g):
        return a * m * m / (m * m - complex(q2, m * g))
    f1r = bw(_f1ra, _mrh, _grho) + bw(_f1rb, _mrhop, _grhop) + bw(_f1rc, _mrhopp, _grhopp)
    f1w = bw(_f1wa, _mome, _gome) + bw(_f1wb, _momep, _gomep) + bw(_f1wc, _momepp, _gomepp)
    return f1r + f1w


def sigmapp(s):
    return _Hpp * jnp.log(s / _sppM) ** 2 + _Ppp + _R1pp * (s / _sppM) ** (-_eta1pp) - _R2pp * (s / _sppM) ** (-_eta2pp)


def wpp(z, pt2, mA):
    H = pt2 + (1 - z) * mA * mA + (z * mp) ** 2
    return 1.0 / 4.0 / (2 * pi * H) * ((1 + (1 - z) ** 2) / z - 2 * z * (1 - z) * ((2 * mp * mp + mA * mA) / H
            - 2 * (z * mp * mp / H) ** 2) + 2 * z * (1 - z) * (z + (1 - z) ** 2) * (mp * mA / H) ** 2
            + 2 * z * ((1 - z) * mA * mA / H) ** 2)


def _sprime(z, pt2, E, mA):
    return 2 * mp * (E - jnp.sqrt(mA * mA + pt2 + z * z * (E * E - mp * mp)))


def d2N(z, pt2, E, kappa, mA):
    """Proton_Brem_Distribution::d2N_proton_brem_to_V (kinetic mixing; not the baryonic variant)."""
    return (abs(F1_proton(mA * mA)) ** 2 * sigmapp(_sprime(z, pt2, E, mA)) / sigmapp(2 * mp * E) * kappa ** 2 * alphaEM
            * wpp(z, pt2, mA))


def simpson_cubature(f, a, b, P, c, d, Q):
    """Integrator.cpp SimpsonCubature (P, Q = half the number of steps), the same terms in the same groups."""
    hx, hy = (b - a) / (2.0 * P), (d - c) / (2.0 * Q)
    xo, xe = a + (2 * np.arange(P) + 1) * hx, a + 2 * np.arange(1, P) * hx  # odd and interior even abscissae
    yo, ye = c + (2 * np.arange(Q) + 1) * hy, c + 2 * np.arange(1, Q) * hy
    F = lambda x, y: np.asarray(f(*np.meshgrid(x, y, indexing="ij")))  # noqa: E731
    hold = f(a, c) + f(b, c) + f(a, d) + f(b, d)
    hold += 4 * (F(xo, [c]).sum() + F(xo, [d]).sum())
    hold += 4 * (F([a], yo).sum() + F([b], yo).sum())
    hold += 2 * (F(xe, [c]).sum() + F(xe, [d]).sum())
    hold += 2 * (F([a], ye).sum() + F([b], ye).sum())
    hold += 4 * F(xe, ye).sum()
    hold += 16 * F(xo, yo).sum()
    hold += 8 * F(xe, yo).sum()
    hold += 8 * F(xo, ye).sum()
    return float(hold) * hx * hy / 9.0


@dataclass(frozen=True)
class ProtonBrem:
    """The card's beam and (z, pT) window; m_A is passed at sampling time (traced)."""
    beam_energy: float
    zmin: float
    zmax: float
    ptmin: float
    ptmax: float
    margin: float = 1.02

    def check(self):
        # Kinetic_Mixing.cpp:190-193 prints an error and continues with a zero-rate channel; the port refuses.
        if self.ptmax < self.ptmin or self.zmax < 0 or self.zmax < self.zmin or self.zmin < 0:
            raise ValueError(f"invalid proton_brem window z [{self.zmin}, {self.zmax}], pT [{self.ptmin}, {self.ptmax}]")

    def v_prod_rate(self, kappa, mA):
        """V_prod_rate: SimpsonCubature(d2N, zmin, zmax, 100, ptmin, ptmax, 100), in V per proton on target."""
        if self.zmin > self.zmax:
            return 0.0
        return simpson_cubature(lambda z, pt: np.asarray(d2N(np.asarray(z, float), np.asarray(pt, float),
                                                             self.beam_energy, kappa, mA)),
                                self.zmin, self.zmax, 100, self.ptmin, self.ptmax, 100)

    def shape(self, z, pt, mA):
        """d2N up to its constant factors, evaluated like stock (pT in the pT^2 slot)."""
        return sigmapp(_sprime(z, pt, self.beam_energy, mA)) * wpp(z, pt, mA)

    def propose(self, key, pool, mA):
        u = jax.random.uniform(key, (pool, 2), dtype=jnp.float64)
        z = self.zmin + (self.zmax - self.zmin) * u[:, 0]
        pt = self.ptmin + (self.ptmax - self.ptmin) * u[:, 1]
        return jnp.stack([z, pt], axis=-1), self.shape(z, pt, mA)

    def find_fmax(self, mA, n=1000):
        """Grid maximum of `shape` over the window (zoomed three times around the peak), times `margin`."""
        z = np.linspace(self.zmin, self.zmax, n)
        pt = np.linspace(self.ptmin, self.ptmax, n)
        best = 0.0
        for _ in range(3):
            Z, P = np.meshgrid(z, pt, indexing="ij")
            f = np.asarray(self.shape(jnp.asarray(Z), jnp.asarray(P), mA))
            f = np.where(np.isfinite(f), f, 0.0)
            i, j = np.unravel_index(np.argmax(f), f.shape)
            best = max(best, f[i, j])
            z = np.linspace(z[max(i - 5, 0)], z[min(i + 5, n - 1)], n)
            pt = np.linspace(pt[max(j - 5, 0)], pt[min(j + 5, n - 1)], n)
        return best * self.margin


def sample_block(dist, key, n, fmax, mA, pool=None):
    """n dark photons (n, 4) for one block (Proton_Brem_Distribution::sample_momentum), with diagnostics."""
    x, filled, nprop, nover = rng.rejection(jax.random.fold_in(key, rng.PARENT), dist.propose, fmax, n, pool or n,
                                            params=mA)
    z, pt = x[:, 0], x[:, 1]
    phi = 2 * pi * rng.uniform(key, rng.PARENT_PHI, (n,))
    pz = math.sqrt(dist.beam_energy ** 2 - mp * mp) * z
    theta = jnp.arctan2(pt, pz)
    mom = jnp.sqrt(pz * pz + pt * pt)
    p3 = jnp.stack([mom * jnp.cos(phi) * jnp.sin(theta), mom * jnp.sin(phi) * jnp.sin(theta), mom * jnp.cos(theta)], -1)
    E = jnp.sqrt(mA * mA + p3[:, 0] ** 2 + p3[:, 1] ** 2 + p3[:, 2] ** 2)
    return jnp.concatenate([E[:, None], p3], axis=-1), {"filled": filled, "proposed": nprop, "over_bar": nover}
