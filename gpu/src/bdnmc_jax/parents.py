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

Both stock samplers are a uniform proposal on a rectangle in (p, v), accepted against a fixed bar:
- Sanford-Wang: v = theta / (pi/2), density min(f, 54.25).
- BMPT: v = theta / theta*(p), density f(p, v theta*(p)). xR is affine in cos(theta), so stock's shrink loop stops
  at the first theta below theta*(p) (where xR = 1); its factors are uniform, so the overshoot of log(theta) is
  Exp(1) and theta given p is exactly Uniform(0, theta*(p)).
`Envelope` samples the same law with a piecewise-constant bound on a (p, v) grid instead of one bar (exact
rejection; much higher acceptance). It draws different random numbers, so it is opt-in.
"""
from dataclasses import dataclass, field
import hashlib
import math

import jax
import jax.numpy as jnp
import numpy as np

from .constants import MASS_PROTON as mp, MASS_KAON_CHARGED, MASS_PION_CHARGED, pi
from . import rng


def polar(mom, th, ph):
    """Particle::ThreeMomentumPolar."""
    return jnp.stack([mom * jnp.cos(ph) * jnp.sin(th), mom * jnp.sin(ph) * jnp.sin(th), mom * jnp.cos(th)], axis=-1)


# --- Sanford-Wang ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SanfordWang:
    """Stock defaults (sanfordwang.h); the card's beam_energy does not enter. kind "pi0" (`pi0_sanfordwang`: pi+ and
    pi- fits, bar fpi0max) or "k0" (`k0_sanfordwang`, stock's eta parents: the K0 fit, bar sigKmax = 0.6506, same
    momentum range); build the latter with `SanfordWang.k0()`."""
    pB: float = 8.89
    ppimax: float = 7.8900
    fmax: float = 54.25
    c: tuple = (0, 220.7000, 1.080000, 1.000000, 1.978000, 1.320000, 5.572000, 0.086800, 9.686000, 1.000000)
    d: tuple = (0, 213.7000, 0.937900, 5.454000, 1.210000, 1.284000, 4.781000, 0.073380, 8.329000, 1.000000)
    kind: str = "pi0"
    e: tuple = (0, 15.130, 1.975, 4.084, 0.928, 0.731, 4.362, 0.048, 13.300, 1.278)

    @classmethod
    def k0(cls):
        return cls(fmax=0.6506, kind="k0")

    def _sw(self, c, p, th):
        pB = self.pB
        return c[1] * p ** c[2] * (1 - p / (pB - c[9])) * jnp.exp(
            -c[3] * p ** c[4] / pB ** c[5] - c[6] * th * (p - c[7] * pB * jnp.cos(th) ** c[8]))

    def density(self, p, th):
        """sin(theta) (swpim + swpip), the quantity pi0sample compares with fpi0max; for "k0", sin(theta) swK
        (K0sample against sigKmax)."""
        if self.kind == "k0":
            return jnp.sin(th) * self._sw(self.e, p, th)
        return jnp.sin(th) * (self._sw(self.d, p, th) + self._sw(self.c, p, th))

    def propose(self, key, pool):
        u = jax.random.uniform(key, (pool, 2), dtype=jnp.float64)
        th = 0 + (pi / 2.0 - 0) * u[:, 0]
        p = 0 + (self.ppimax - 0) * u[:, 1]
        return jnp.stack([p, th], axis=-1), self.density(p, th)

    # (p, v) coordinates of the stock law, for Envelope
    @property
    def p_range(self):
        return self.ppimax

    def theta_of(self, p, v):
        return v * (pi / 2.0)

    def uv_density(self, p, v):
        return jnp.minimum(self.density(p, self.theta_of(p, v)), self.fmax)


# --- BMPT --------------------------------------------------------------------------------------------------------

_A, _B, _alpha, _beta, _a, _gamma, _delta, _r0, _r1, _MXBAR = 62.3, 1.57, 3.45, 0.517, 6.10, 0.153, 0.478, 1.05, 2.65, 1.88
_BE_MASS_NUMBER = 8.0
# namespace Kaon (BMPT_dist.cpp); its A (7.74), r0 and r1 are not used by the K0 sampler (below)
_K_B, _K_alpha, _K_beta, _K_a, _K_gamma = 0.0, 2.45, 0.444, 5.04, 0.121
_K_delta = 2 * _K_gamma


@dataclass(frozen=True)
class BMPT:
    """choice "pi0" (`bmpt`) or "k0" (`bpmt_k0`, stock's eta parents). Stock's K0 choice, kept: the meson mass is the
    charged kaon's; Invariant_Cross_Section_Kaon multiplies by the pion fit's A (62.3, not Kaon::A, a constant factor
    that the bar absorbs); and the K0 density uses the pion ratio pratio, not pratio_Kaon, in (1 + 3/pratio)/2."""
    beam_energy: float
    mass_number: float
    margin: float = 1.02
    max_shrinks: int = 400
    choice: str = "pi0"

    @property
    def meson_mass(self):
        # stock uses the charged-pion mass for the pi0 choice and the charged-kaon mass for K0
        return MASS_KAON_CHARGED if self.choice == "k0" else MASS_PION_CHARGED

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
        """BMPT::Invariant_Cross_Section (pi0), or Invariant_Cross_Section_Kaon (k0)."""
        if self.choice == "k0":
            a, B, alpha, beta, gamma, delta = _K_a, _K_B, _K_alpha, _K_beta, _K_gamma, _K_delta
        else:
            a, B, alpha, beta, gamma, delta = _a, _B, _alpha, _beta, _gamma, _delta
        x = self.xR(p, th)
        pT = p * jnp.sin(th)
        ap = a / x ** gamma
        bp = a * a / (2 * x ** delta)
        xf = self.xF(p, th)
        alphaf = (0.74 - 0.55 * xf + 0.26 * xf ** 2) * (0.98 + 0.21 * pT)
        return (_A * (1.0 - x) ** alpha * (1.0 + B * x) * x ** (-1.0 * beta) * (1 + ap * pT + bp * pT * pT)
                * jnp.exp(-ap * pT) * (self.mass_number / _BE_MASS_NUMBER) ** alphaf)

    def density(self, p, th):
        """The `prob` of BMPT::sample_momentum."""
        M = self.meson_mass
        pratio = _r0 * (1 + self.xR(p, th)) ** _r1
        if self.choice == "k0":
            return (2 * pi * jnp.sin(th) * p ** 2 / jnp.sqrt(p * p + M ** 2) * self.ics(p, th)
                    * (1.0 + 3.0 / pratio) / 2)
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

    # (p, v) coordinates of the stock law, for Envelope
    @property
    def p_range(self):
        return self.p_max

    def cos_theta_star(self, p):
        """cos(theta) at which xR(p, theta) = 1 (xR decreases as cos(theta) grows); > 1 means no valid theta."""
        E, M = self.beam_energy, self.meson_mass
        Ep = jnp.sqrt(p * p + M ** 2)
        return (-2 * mp * mp - 2 * E * mp + 2 * (E + mp) * Ep + _MXBAR ** 2 - M ** 2) / (
            2 * math.sqrt(E ** 2 - mp * mp) * p)

    def theta_of(self, p, v):
        th_star = jnp.where(self.cos_theta_star(p) <= math.cos(self.theta_max), self.theta_max,
                            jnp.arccos(jnp.clip(self.cos_theta_star(p), -1.0, 1.0)))
        return v * th_star

    def uv_density(self, p, v):
        return self.density(p, self.theta_of(p, v))

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


# --- envelope sampler --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Envelope:
    """Exact rejection sampling of a base distribution's stock law with a piecewise-constant bound on a (p, v) grid:
    choose a cell with probability bound x area, propose uniformly in it, accept with density / bound. The tables
    are compile-time constants (they depend on beam and target only); equality and hashing go by their digest.
    `propose` returns f / bound, so `rejection` runs with fmax = 1 and counts any f > bound as over the bar (a
    failure: the bound missed a peak)."""
    base: object
    digest: str
    efficiency: float
    p_lo: np.ndarray = field(compare=False, repr=False)
    dp: np.ndarray = field(compare=False, repr=False)
    v_lo: np.ndarray = field(compare=False, repr=False)
    dv: np.ndarray = field(compare=False, repr=False)
    bound: np.ndarray = field(compare=False, repr=False)
    cdf: np.ndarray = field(compare=False, repr=False)

    def propose(self, key, pool):
        u = jax.random.uniform(key, (pool, 3), dtype=jnp.float64)
        n_v = len(self.v_lo)
        cell = jnp.minimum(jnp.searchsorted(jnp.asarray(self.cdf), u[:, 0], side="right"), len(self.cdf) - 1)
        ip, iv = cell // n_v, cell % n_v
        p = jnp.asarray(self.p_lo)[ip] + u[:, 1] * jnp.asarray(self.dp)[ip]
        v = jnp.asarray(self.v_lo)[iv] + u[:, 2] * jnp.asarray(self.dv)[iv]
        f = self.base.uv_density(p, v) / jnp.asarray(self.bound)[cell]
        return jnp.stack([p, self.base.theta_of(p, v)], axis=-1), jnp.where(jnp.isfinite(f), f, jnp.inf)


def make_envelope(base, n_p=256, n_v=128, v_min=1e-5, n_sub=6, margin=1.2, n_check=2_000_000, seed=0):
    """Build the Envelope of `base` (SanfordWang or BMPT): p in n_p equal bins, v in [0, v_min] plus n_v geometric
    bins up to 1; each cell's bound is the maximum on an n_sub x n_sub sub-grid (edges included) x margin. Checked
    on n_check random points; raises if any exceeds its bound, or if some p has no valid theta (BMPT)."""
    p_e = np.linspace(0.0, base.p_range, n_p + 1)
    v_e = np.concatenate([[0.0], np.geomspace(v_min, 1.0, n_v)])
    if isinstance(base, BMPT) and np.asarray(base.cos_theta_star(jnp.asarray(p_e[1:]))).max() > 1:
        raise NotImplementedError("BMPT: some p below p_max has no theta with xR <= 1 (stock would not terminate)")
    s = np.linspace(0.0, 1.0, n_sub)
    P = (p_e[:-1, None, None, None] + np.diff(p_e)[:, None, None, None] * s[None, None, :, None])
    V = (v_e[None, :-1, None, None] + np.diff(v_e)[None, :, None, None] * s[None, None, None, :])
    P, V = np.broadcast_arrays(P, V)
    f = np.asarray(base.uv_density(jnp.asarray(P), jnp.asarray(V)))
    f = np.where(np.isfinite(f), f, 0.0)
    cmax, cmean = f.max(axis=(2, 3)), f.mean(axis=(2, 3))
    area = np.diff(p_e)[:, None] * np.diff(v_e)[None, :]
    bound = np.maximum(cmax * margin, np.finfo(float).tiny).ravel()
    w = bound * area.ravel()
    cdf = np.cumsum(w) / w.sum()
    env = Envelope(base, "", float((cmean * area).sum() / w.sum()),
                   p_e[:-1].copy(), np.diff(p_e), v_e[:-1].copy(), np.diff(v_e), bound, cdf)
    digest = hashlib.sha256(b"".join(a.tobytes() for a in (env.p_lo, env.dp, env.v_lo, env.dv, bound, cdf))).hexdigest()
    env = Envelope(base, digest, env.efficiency, env.p_lo, env.dp, env.v_lo, env.dv, bound, cdf)
    # Check the bound on random points, cell by cell (uniform within cells, cells chosen like the sampler does).
    rs = np.random.default_rng(seed)
    cell = np.minimum(np.searchsorted(cdf, rs.random(n_check), side="right"), len(cdf) - 1)
    ip, iv = cell // n_v, cell % n_v
    pp = env.p_lo[ip] + rs.random(n_check) * env.dp[ip]
    vv = env.v_lo[iv] + rs.random(n_check) * env.dv[iv]
    ratio = np.asarray(base.uv_density(jnp.asarray(pp), jnp.asarray(vv))) / bound[cell]
    if not (np.nan_to_num(ratio, nan=0.0) <= 1).all():
        raise RuntimeError(f"envelope bound exceeded on {int((ratio > 1).sum())} of {n_check} check points "
                           f"(max ratio {np.nanmax(ratio):.3f}); raise margin or refine the grid")
    return env


# --- driver ------------------------------------------------------------------------------------------------------

def sample_block(dist, key, n, fmax, m_meson, pool=None):
    """n parent four-momenta (n, 4) for one block, with diagnostics. Default pool: n per round for the stock
    samplers (low acceptance: many rounds anyway); for an Envelope, enough that one round usually fills the block
    (a short round is followed by another, so this only sets the cost, not the law)."""
    if pool is None:
        pool = -(-int(1.1 * n / dist.efficiency) // 4096) * 4096 if isinstance(dist, Envelope) else n
    x, filled, nprop, nover = rng.rejection(jax.random.fold_in(key, rng.PARENT), dist.propose, fmax, n, pool)
    ph = 2 * pi * rng.uniform(key, rng.PARENT_PHI, (n,))  # Flat(0, 2 pi), drawn after acceptance in stock
    mom = polar(x[:, 0], x[:, 1], ph)
    E = jnp.sqrt(m_meson * m_meson + mom[:, 0] ** 2 + mom[:, 1] ** 2 + mom[:, 2] ** 2)
    return jnp.concatenate([E[:, None], mom], axis=-1), {"filled": filled, "proposed": nprop, "over_bar": nover}
