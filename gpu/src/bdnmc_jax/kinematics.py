"""Four-vector helpers, vectorised (stock src/Particle.cpp, src/Kinematics.cpp).

Four-momenta are arrays (..., 4) ordered (E, px, py, pz).
"""
import jax.numpy as jnp

from .constants import pi


def three_momentum(m, px, py, pz):
    """Particle::ThreeMomentum: E = sqrt(m^2 + p^2) in stock's summation order."""
    E = jnp.sqrt(m * m + px * px + py * py + pz * pz)
    return jnp.stack([E, px, py, pz], axis=-1)


def momentum(p4):
    return jnp.sqrt(p4[..., 1] * p4[..., 1] + p4[..., 2] * p4[..., 2] + p4[..., 3] * p4[..., 3])


def theta(p4):
    """calculatetheta: acos(pz / |p|)."""
    c = p4[..., 3] / momentum(p4)
    if c.dtype == jnp.float32:
        # |pz / |p|| <= 1, but float32 rounding (GPU division and sqrt) takes it past 1 for particles along the beam
        # axis, and the NaN would lose the particle; clipped, they get angle 0 (or pi). float64 is left as stock.
        c = jnp.clip(c, -1.0, 1.0)
    return jnp.arccos(c)


def phi(p4):
    """Kinematics.cpp phi(), quadrant by quadrant. Its last branch also catches px == 0 or py == 0 (e.g. it gives
    3 pi / 2 for px == 0, py > 0); that is reproduced."""
    px, py = p4[..., 1], p4[..., 2]
    a = jnp.arctan(jnp.abs(py / jnp.where(px == 0, 1.0, px)))
    a = jnp.where(px == 0, pi / 2, a)  # atan(fabs(py/0)) = atan(inf) = pi/2
    out = jnp.where((px > 0) & (py > 0), a,
          jnp.where((px < 0) & (py > 0), pi - a,
          jnp.where((px < 0) & (py < 0), pi + a, 2 * pi - a)))
    return jnp.where((px == 0) & (py == 0), 0.0, out)


def lorentz(p4, beta, bx, by, bz):
    """Particle::Lorentz(beta, betax, betay, betaz); identity where beta == 0."""
    zero = beta == 0.0
    b = jnp.where(zero, 1.0, beta)
    gamma = 1 / jnp.sqrt(1 - b * b)
    nx, ny, nz = bx / b, by / b, bz / b
    g1 = gamma - 1
    E, px, py, pz = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    E2 = gamma * E + gamma * bx * px + gamma * by * py + gamma * bz * pz
    x2 = gamma * bx * E + (1 + g1 * nx * nx) * px + g1 * nx * ny * py + g1 * nx * nz * pz
    y2 = gamma * by * E + g1 * nx * ny * px + (1 + g1 * ny * ny) * py + g1 * ny * nz * pz
    z2 = gamma * bz * E + g1 * nx * nz * px + g1 * ny * nz * py + (1 + g1 * nz * nz) * pz
    out = jnp.stack([E2, x2, y2, z2], axis=-1)
    return jnp.where(zero[..., None], p4, out)


def lorentz_to(p4, parent, m_parent):
    """Particle::Lorentz(Particle& parent): boost p4 by the parent's velocity."""
    if parent.dtype == jnp.float32:
        return _boost_by_momentum(p4, parent, m_parent)
    gamma = parent[..., 0] / m_parent
    beta = jnp.sqrt(1.0 - 1.0 / gamma / gamma)
    E = parent[..., 0]
    return lorentz(p4, beta, parent[..., 1] / E, parent[..., 2] / E, parent[..., 3] / E)


def _boost_by_momentum(p4, parent, m):
    """lorentz_to's boost for float32, from the parent's four-momentum: gamma = E/m, gamma beta = |p|/m and
    gamma - 1 = |p|^2 / (m (E + m)). Stock's route through beta = sqrt(1 - 1/gamma^2) and back to
    1/sqrt(1 - beta^2) loses about 2 eps gamma^2 of gamma, several % at gamma ~ 700 in float32."""
    E, P = parent[..., 0], parent[..., 1:]
    pm = jnp.sqrt(jnp.sum(P * P, axis=-1))
    at_rest = pm == 0
    n = P / jnp.where(at_rest, 1.0, pm)[..., None]
    e, p = p4[..., 0], p4[..., 1:]
    n_p = jnp.sum(n * p, axis=-1)
    gamma_beta, g1 = pm / m, pm * pm / (m * (E + m))
    out = jnp.concatenate([(E / m * e + gamma_beta * n_p)[..., None], p + (g1 * n_p + gamma_beta * e)[..., None] * n],
                          axis=-1)
    return jnp.where(at_rest[..., None], p4, out)


def rotate_y(p4, t):
    px, pz = p4[..., 1], p4[..., 3]
    zr = pz * jnp.cos(t) - px * jnp.sin(t)
    xr = px * jnp.cos(t) + pz * jnp.sin(t)
    return jnp.stack([p4[..., 0], xr, p4[..., 2], zr], axis=-1)


def rotate_z(p4, t):
    px, py = p4[..., 1], p4[..., 2]
    yr = py * jnp.cos(t) + px * jnp.sin(t)
    xr = px * jnp.cos(t) - py * jnp.sin(t)
    return jnp.stack([p4[..., 0], xr, yr, p4[..., 3]], axis=-1)


def lam(a, b, c):
    """Kinematics.cpp lambda()."""
    return jnp.abs(a * a + b * b + c * c - 2 * a * b - 2 * a * c - 2 * b * c)
