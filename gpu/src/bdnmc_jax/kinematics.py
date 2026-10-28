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
    return jnp.arccos(p4[..., 3] / momentum(p4))


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
    gamma = parent[..., 0] / m_parent
    beta = jnp.sqrt(1.0 - 1.0 / gamma / gamma)
    E = parent[..., 0]
    return lorentz(p4, beta, parent[..., 1] / E, parent[..., 2] / E, parent[..., 3] / E)


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
