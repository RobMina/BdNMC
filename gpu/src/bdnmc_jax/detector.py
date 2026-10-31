"""Detector path length `Ldet`, vectorised over rays (stock src/detector.cpp).

A ray is (origin, p): position = origin + t * p, with p the DM three-momentum. Each function returns
(L, t_enter, t_exit): the path length in metres (0 for a miss) and the crossing parameters that stock stores in
`Particle::crossing` (used by `Generate_Position`). DM is stable, so the `END_SET` branches never apply and are
not ported.
"""
from dataclasses import dataclass
import math

import jax.numpy as jnp
import numpy as np

from .card import stod


@dataclass(frozen=True)
class Sphere:
    center: tuple
    radius: float


@dataclass(frozen=True)
class Cuboid:
    center: tuple
    faces: tuple  # 6 x 3: vectors from the centre to the face centres (stock detector_cuboid::face)
    face_dist: tuple  # width, height, length


def xyx_rotate(t1, t2, t3, v):
    """detector.cpp XYX_rotate, in place on a 3-vector (proper Euler XYX)."""
    x, y, z = v
    return (
        x * math.cos(t2) + z * math.cos(t3) * math.sin(t2) + y * math.sin(t2) * math.sin(t3),
        x * math.sin(t1) * math.sin(t2) + z * (-math.cos(t2) * math.cos(t3) * math.sin(t1) - math.cos(t1) * math.sin(t3))
        + y * (math.cos(t1) * math.cos(t3) - math.cos(t2) * math.sin(t1) * math.sin(t3)),
        -x * math.cos(t1) * math.sin(t2) + y * (math.cos(t3) * math.sin(t1) + math.cos(t1) * math.cos(t2) * math.sin(t3))
        + z * (math.cos(t1) * math.cos(t2) * math.cos(t3) - math.sin(t1) * math.sin(t3)),
    )


def make_cuboid(x, y, z, width, height, length, phi, theta, psi):
    """detector_cuboid constructor; the card passes (width, height, length, det-phi, det-theta, det-psi)."""
    axes = [(width / 2.0, 0.0, 0.0), (0.0, height / 2.0, 0.0), (0.0, 0.0, length / 2.0)]
    faces = [xyx_rotate(phi, theta, psi, a) for a in axes]
    faces += [tuple(-c for c in f) for f in faces]
    return Cuboid((x, y, z), tuple(faces), (width, height, length))


def from_card(card):
    """Parameter::Build_Detector. Only sphere and cuboid are ported so far."""
    k = card.keymap
    shape = k.get("detector")
    pos = tuple(stod(k[a]) for a in ("x-position", "y-position", "z-position"))
    if shape == "sphere":
        return Sphere(pos, stod(k["radius"]))
    if shape == "cuboid":
        return make_cuboid(*pos, *(stod(k[a]) for a in ("width", "height", "length", "det-phi", "det-theta",
                                                         "det-psi")))
    raise NotImplementedError(f"detector shape {shape!r} is not ported yet")


def _dot(a, b):
    # Same summation order as detector.cpp ip(): a0*b0 + a1*b1 + a2*b2.
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def ldet_sphere(det, origin, p):
    """detector_sphere::Ldet."""
    r = jnp.asarray(det.center)
    o = r - origin
    A = _dot(p, p)
    B = -2 * _dot(o, p)
    if A.dtype == jnp.float32:
        # B^2 - 4AC cancels terms of order (2 |o| |p|)^2; in float32 that costs up to ~0.3 m of a 10 m chord. The same
        # discriminant from the perpendicular offset w = o - t0 p of the closest approach: 4 A (R^2 - |w|^2).
        t0 = _dot(o, p) / jnp.where(A == 0, 1.0, A)
        w = o - t0[..., None] * p
        disc = 4 * A * (det.radius ** 2 - _dot(w, w))
    else:
        C = _dot(o, o) - det.radius ** 2
        disc = B ** 2 - 4 * A * C
    miss = (disc < 0) | (A == 0)
    sq = jnp.sqrt(jnp.where(miss, 0.0, disc))
    A_safe = jnp.where(A == 0, 1.0, A)
    t_in = jnp.maximum((-B - sq) / (2 * A_safe), 0.0)
    t_out = jnp.maximum((-B + sq) / (2 * A_safe), 0.0)
    hit = ~miss & (t_out > t_in)
    L = jnp.where(hit, (t_out - t_in) * jnp.sqrt(A), 0.0)
    return L, jnp.where(hit, t_in, 0.0), jnp.where(hit, t_out, 0.0)


def ldet_cuboid(det, origin, p):
    """detector_cuboid::Ldet, branch for branch (the three face pairs are processed in stock order)."""
    r = jnp.asarray(det.center)
    o = r - origin
    faces = np.asarray(det.faces)
    shape = origin.shape[:-1]
    entry = jnp.zeros(shape)
    exit_ = jnp.zeros(shape)
    dead = jnp.zeros(shape, dtype=bool)
    for i in range(3):
        f, g = jnp.asarray(faces[i]), jnp.asarray(faces[i + 3])
        ff = faces[i][0] * faces[i][0] + faces[i][1] * faces[i][1] + faces[i][2] * faces[i][2]
        gg = faces[i + 3][0] * faces[i + 3][0] + faces[i + 3][1] * faces[i + 3][1] + faces[i + 3][2] * faces[i + 3][2]
        fd = det.face_dist[i]
        bot1 = _dot(p, f)
        bot2 = _dot(p, g)
        par = bot1 == 0
        # Ray parallel to this face pair: it must lie between the two faces.
        d1 = jnp.abs((ff + _dot(o, f)) / fd * 2)
        d2 = jnp.abs((gg + _dot(o, g)) / fd * 2)
        dead = dead | (par & ((d1 > fd) | (d2 > fd)))

        ft1 = _dot(o, f) + ff
        ft2 = _dot(o, g) + gg
        safe1 = jnp.where(par, 1.0, bot1)
        safe2 = jnp.where(par, 1.0, bot2)
        detc1 = ft1 / safe1
        detc2 = ft2 / safe2
        swap = (ft1 > 0) & (ft2 <= 0)
        enter = (ft1 <= 0) | swap
        detc1, detc2 = jnp.where(swap, detc2, detc1), jnp.where(swap, detc1, detc2)

        live = ~par & ~dead
        dead = dead | (live & enter & (detc1 > detc2))
        live = live & ~dead
        entry = jnp.where(live & enter & (detc1 > entry), detc1, entry)
        upd1 = live & ~enter & (detc1 > 0) & ((detc1 < exit_) | (exit_ == 0))
        exit_ = jnp.where(upd1, detc1, exit_)
        upd2 = live & (detc2 > 0) & ((detc2 < exit_) | (exit_ == 0))
        exit_ = jnp.where(upd2, detc2, exit_)
    hit = ~dead & (entry < exit_) & (entry >= 0)
    L = jnp.where(hit, (exit_ - entry) * jnp.sqrt(_dot(p, p)), 0.0)
    return L, jnp.where(hit, entry, 0.0), jnp.where(hit, exit_, 0.0)


def ldet(det, origin, p):
    if isinstance(det, Sphere):
        return ldet_sphere(det, origin, p)
    if isinstance(det, Cuboid):
        return ldet_cuboid(det, origin, p)
    raise TypeError(det)
