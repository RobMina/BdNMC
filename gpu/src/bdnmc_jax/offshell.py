"""Off-shell meson -> gamma V*(-> chi chi) (stock src/decay_gen_kinetic_mixing.cpp sample_dist, src/decay.cpp
DecayDM_Off_Shell).

Stock samples (s, theta), s = m_V*^2 and theta the DM angle in the V* frame, uniformly on [4 m_chi^2, m_meson^2] x
[0, pi] against `d2brpi0_to_gamma_dm_dm` (eta: `d2breta_to_gamma_dm_dm`) with a running maximum: started at the
density at (m_V^2, pi/2) when that is inside the window, otherwise at 0, and raised by 1000 burn-in draws and then
whenever exceeded, so its early draws are slightly clipped. The port uses a fixed bar (grid maximum x margin) and
fails if any proposal exceeds it, i.e. it samples the unclipped density, as for BMPT.

The constant factors of the density (alpha_D kappa^2 / (8 m^6 pi), and eta's two-photon ratio) do not change the
accept test, so `shape` leaves them out. m_V Gamma_V depends only on the model and is computed once on the host.

DecayDM_Off_Shell, reproduced step by step:
1. TwoBodyDecay(V* at rest, chi, chi, theta): DM along (theta, phid) in the V* frame, phid = 2 pi Flat(0, 1);
2. V* given momentum TriangleFunc(m_meson, sqrt(s), 0) along +z in the meson frame, and the DM boosted to it;
3. mtheta = acos(Flat(-1, 1)), mphi = 2 pi Flat(0, 1), then Rotate_z(mtheta) and Rotate_y(mphi) of V* and both DM.
   V* starts along z, which Rotate_z leaves unchanged, so its direction in the meson frame is (sin mphi, 0, cos mphi):
   uniform in angle in the x-z plane, not isotropic (a stock bug, kept);
4. all three boosted to the lab with the meson.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from . import branching, rng
from .constants import MASS_ETA, MASS_PION_NEUTRAL, pi
from .decay import triangle
from .kinematics import lorentz_to, rotate_y, rotate_z


@dataclass(frozen=True)
class OffShell:
    meson: str  # "pi0" or "eta"
    margin: float = 1.02

    @property
    def m_meson(self):
        return MASS_PION_NEUTRAL if self.meson == "pi0" else MASS_ETA

    @staticmethod
    def shape(s, th, m, mv, mx, mv_gamma):
        """d2br{pi0,eta}_to_gamma_dm_dm without its constant factor, in stock's order of operations."""
        return (jnp.sqrt(1 - 4 * mx * mx / s) * (m * m - s) ** 3 * (s - 4 * mx * mx) * jnp.sin(th) ** 3
                / ((s - mv * mv) ** 2 + mv_gamma ** 2))

    def propose(self, key, pool, params):
        m, mv, mx, mv_gamma = params
        u = jax.random.uniform(key, (pool, 2), dtype=rng.FLOAT)
        s = 4 * mx * mx + (m * m - 4 * mx * mx) * u[:, 0]  # Random::Flat(4 mx^2, m^2)
        th = pi * u[:, 1]  # Random::Flat(0, pi)
        return jnp.stack([s, th], axis=-1), self.shape(s, th, m, mv, mx, mv_gamma)

    def find_fmax(self, mv, mx, mv_gamma, n=4001):
        """Maximum of `shape` over the window (theta = pi/2), on a grid zoomed three times around the peak, plus a
        fine grid around s = m_V^2 when the resonance is inside, times `margin`."""
        m = self.m_meson
        lo, hi = 4 * mx * mx, m * m
        if not lo < hi:
            raise ValueError(f"{self.meson}: empty off-shell window (2 m_chi >= m_meson)")

        def f(s):
            v = np.asarray(self.shape(jnp.asarray(s), pi / 2, m, mv, mx, mv_gamma))
            return np.where(np.isfinite(v), v, 0.0)

        grids = [np.linspace(lo, hi, n)]
        if lo < mv * mv < hi:
            w = 50 * max(mv_gamma, 1e-12)
            grids.append(np.linspace(max(lo, mv * mv - w), min(hi, mv * mv + w), n))
        best = 0.0
        for s in grids:
            for _ in range(3):
                v = f(s)
                i = int(np.argmax(v))
                best = max(best, float(v[i]))
                s = np.linspace(s[max(i - 5, 0)], s[min(i + 5, len(s) - 1)], n)
        if not best > 0:
            raise ValueError("off-shell density is zero on the whole window")
        return best * self.margin


def mv_gamma(mv, mx, kappa, alpha_d):
    """m_V Gamma_V (branchingratios.cpp GammaV), the width term of the V* propagator."""
    return mv * branching.GammaV(mv, mx, kappa, alpha_d)


def decay(parent, m_meson, m_dm, s, th, u_phid, u_mtheta, u_mphi):
    """DecayDM_Off_Shell: (V*, dm1, dm2) lab four-momenta (..., 4) and the V* mass sqrt(s)."""
    mm = jnp.sqrt(s)  # darkphoton.Set_Mass(sqrt(s))
    phid = u_phid * 2 * pi
    mom = triangle(mm, m_dm, m_dm)
    px, py, pz = mom * jnp.sin(th) * jnp.cos(phid), mom * jnp.sin(th) * jnp.sin(phid), mom * jnp.cos(th)
    E = jnp.sqrt(m_dm * m_dm + px * px + py * py + pz * pz)
    d1 = jnp.stack([E, px, py, pz], axis=-1)
    d2 = jnp.stack([E, -px, -py, -pz], axis=-1)  # V* at rest: the boost to it is the identity
    pm = triangle(m_meson, mm, 0.0)
    zero = jnp.zeros_like(pm)
    med = jnp.stack([jnp.sqrt(mm * mm + zero * zero + zero * zero + pm * pm), zero, zero, pm], axis=-1)
    d1, d2 = lorentz_to(d1, med, mm), lorentz_to(d2, med, mm)
    mtheta = jnp.arccos(-1 + 2 * u_mtheta)
    mphi = u_mphi * 2 * pi
    med, d1, d2 = (rotate_y(rotate_z(p, mtheta), mphi) for p in (med, d1, d2))
    return (lorentz_to(med, parent, m_meson), lorentz_to(d1, parent, m_meson), lorentz_to(d2, parent, m_meson), mm)
