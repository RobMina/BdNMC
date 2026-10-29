"""`production_distribution particle_list`: parent mesons read from a text file (stock src/Particle_List.cpp).

Stock behaviour, reproduced:
- The whole file is read at start-up. Each trial takes the next row; after the last row it starts again from the
  first. The port's trial i of a run takes row i mod N, so a run of k N trials uses every row exactly k times.
  Stock's burn-in draws rows too, so its main loop starts at a random row; for runs much longer than N the two
  orders weight the rows the same.
- The E column is read but not used: `Distribution::Sample_Particle` sets the channel's meson mass and
  `Particle::ThreeMomentum` recomputes E = sqrt(m^2 + p^2).
- With `particle_list_position true`, columns 5-7 (x, y, z, metres) and 8 (t, seconds) set the parent's origin,
  which becomes the origin of the V and both DM (`Set_Origin`, `Set_Creation_Time`, `Set_Time`, `Link_Particles`).
  Fewer than 7 columns leave the position at zero; 7 columns leave t at zero. Without the flag every origin is zero.
- `parse_line`: lines that are empty or start with '#' are skipped. The separator is ' ' if the line has a space,
  else a tab, else a comma, and each single separator character splits a field, so two spaces in a row make an
  empty field.

Where the port differs: stock prints an error and skips a line it cannot parse (an empty or non-numeric field
among those it reads); the port refuses the file and names the line. Lines with fewer than four fields, which stock
reads out of bounds, and zero-momentum rows with positions, which give stock NaN positions, are refused too.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import os

import jax.numpy as jnp
import numpy as np

from .card import stod


@dataclass(frozen=True)
class ParticleList:
    """Static description of a loaded list (the rows themselves are the channel's traced `rows`)."""
    path: str
    position: bool
    n: int
    sha256: str


def _fields(line):
    for sep in (" ", "\t", ","):
        if sep in line:
            return line.split(sep)
    return [line]


def parse(text, position, name="<text>"):
    """Rows (N, 7) = px, py, pz, x, y, z, t from a list's text, with stock's parse_line rules."""
    rows = []
    for lineno, line in enumerate(text.split("\n"), 1):
        if not line or line[0] == "#":
            continue
        f = _fields(line)
        if len(f) < 4:
            raise ValueError(f"{name}:{lineno}: fewer than 4 fields: {line!r}")
        try:
            px, py, pz = stod(f[0]), stod(f[1]), stod(f[2])
            stod(f[3])  # E: parsed (a bad value makes stock skip the line), then unused
            x = y = z = t = 0.0
            if position and len(f) >= 7:
                x, y, z = stod(f[4]), stod(f[5]), stod(f[6])
                if len(f) >= 8:
                    t = stod(f[7])
        except ValueError as e:
            raise ValueError(f"{name}:{lineno}: stock would skip this line ({e}): {line!r}") from None
        if position and px == 0 and py == 0 and pz == 0:
            raise ValueError(f"{name}:{lineno}: zero momentum with particle_list_position (stock: NaN position)")
        rows.append((px, py, pz, x, y, z, t))
    if not rows:
        raise ValueError(f"{name}: no particles (stock: 'File empty')")
    return np.asarray(rows, dtype=np.float64)


@lru_cache(maxsize=8)
def _load(path, position, mtime, size):
    data = open(path, "rb").read()
    return parse(data.decode(), position, path), hashlib.sha256(data).hexdigest()


def load(path, position):
    """(ParticleList, rows). A relative path is opened from the working directory, as stock does."""
    st = os.stat(path)
    rows, digest = _load(str(path), bool(position), st.st_mtime_ns, st.st_size)
    return ParticleList(str(path), bool(position), len(rows), digest), rows


def block(rows, block_index, n, m_meson):
    """Parents (n, 4) and their origins (n, 4: x, y, z, t) for trials block_index * n ... + n - 1."""
    r = rows[(block_index * n + jnp.arange(n)) % rows.shape[0]]
    px, py, pz = r[:, 0], r[:, 1], r[:, 2]
    E = jnp.sqrt(m_meson * m_meson + px * px + py * py + pz * pz)  # Particle::ThreeMomentum
    return jnp.stack([E, px, py, pz], axis=-1), r[:, 3:7]
