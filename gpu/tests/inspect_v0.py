"""Print the V0 rows where the port and stock differ most (diagnostic, not a test)."""
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bdnmc_jax  # noqa: E402,F401
from bdnmc_jax import electron, parents  # noqa: E402

DUMP = Path(__file__).resolve().parents[1] / "v0" / "v0_dump.txt"
rows = {}
for line in DUMP.read_text().splitlines():
    name, *vals = line.split()
    rows.setdefault(name, []).append([float(v) for v in vals])
rows = {k: np.array(v) for k, v in rows.items()}

t = rows["ThetaEe"]
got = np.asarray(electron.theta_ee(jnp.asarray(t[:, 0]), jnp.asarray(t[:, 1]), 0.005))
bad = ~np.isclose(got, t[:, 2], rtol=1e-10, atol=0)
print("ThetaEe: Ee, E, stock, port")
for r, g in zip(t[bad], got[bad]):
    print(f"  {r[0]:.17g} {r[1]:.17g} {r[2]:.17g} {g:.17g}")

b = rows["bmpt_ics"]
bm = parents.BMPT(120.0, 12.0)
got = np.asarray(bm.ics(jnp.asarray(b[:, 0]), jnp.asarray(b[:, 1])))
xr = np.asarray(bm.xR(jnp.asarray(b[:, 0]), jnp.asarray(b[:, 1])))
rel = np.abs(got - b[:, 2]) / np.abs(b[:, 2])
print("BMPT ics: p, theta, xR, stock, port, rel")
for i in np.argsort(-np.nan_to_num(rel))[:8]:
    print(f"  {b[i, 0]:.6g} {b[i, 1]:.6g} {xr[i]:.17g} {b[i, 2]:.17g} {got[i]:.17g} {rel[i]:.3g}")
