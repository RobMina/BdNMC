"""Meson -> gamma V, V -> chi chi, on-shell (stock src/decay.cpp DecayDM). The four uniforms are passed in, in the
order stock draws them (ym, zm, yd, zd), so a test can feed both codes the same numbers.
"""
import jax.numpy as jnp

from .constants import pi
from .kinematics import lam, lorentz_to


def decay_dm(parent, m_parent, m_v, m_dm, ym, zm, yd, zd):
    """Return (mediator, dm1, dm2) lab four-momenta (..., 4). Vertices are the parent's (all at the origin for
    Sanford-Wang/BMPT parents, which stock samples at (0, 0, 0, 0))."""
    thetam = jnp.arccos(-1 + 2 * ym)
    phim = 2.0 * pi * zm
    l = lam(1, m_v * m_v / m_parent / m_parent, 0)
    Em = m_parent / 2.0 * (1.0 + m_v * m_v / m_parent / m_parent)
    pm = m_parent / 2.0 * jnp.sqrt(l)
    med = jnp.stack([jnp.broadcast_to(Em, ym.shape), pm * jnp.sin(thetam) * jnp.cos(phim),
                     pm * jnp.sin(thetam) * jnp.sin(phim), pm * jnp.cos(thetam)], axis=-1)
    med = lorentz_to(med, parent, m_parent)

    thetad = jnp.arccos(-1 + 2 * yd)
    phid = 2.0 * pi * zd
    l = lam(1, m_dm * m_dm / m_v / m_v, m_dm * m_dm / m_v / m_v)
    Ed = m_v / 2.0
    pd = m_v / 2.0 * jnp.sqrt(l)
    pdx = pd * jnp.sin(thetad) * jnp.cos(phid)
    pdy = pd * jnp.sin(thetad) * jnp.sin(phid)
    pdz = pd * jnp.cos(thetad)
    Edv = jnp.broadcast_to(Ed, yd.shape)
    d1 = jnp.stack([Edv, pdx, pdy, pdz], axis=-1)
    d2 = jnp.stack([Edv, -pdx, -pdy, -pdz], axis=-1)
    return med, lorentz_to(d1, med, m_v), lorentz_to(d2, med, m_v)
