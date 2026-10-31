"""The V0 function checks (tests/test_v0.py, against the stock dump v0/v0_dump.txt) as a report rather than a test:
per function, the worst and the 99.9th-percentile relative error, the count above 1e-6 and 1e-3, and NaN/inf
mismatches; for Ldet the worst absolute difference and hit/miss disagreements. Run it with
BDNMC_JAX_PRECISION=float32 to measure what float32 costs each function (float64 gives the V0 levels, ~1e-12).
Functions evaluated on the host in numpy or Python (branching ratios, the brem rate integral) show float64 levels in
either mode.
Usage: [BDNMC_JAX_PRECISION=float32] precision_v0.py [--dump v0/v0_dump.txt] [--json out.json]
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bdnmc_jax  # noqa: E402
from bdnmc_jax import branching, brem, detector, electron, offshell, parents  # noqa: E402
from bdnmc_jax.constants import MASS_ELECTRON as Me  # noqa: E402

M, MV, KAP, ALD = 0.005, 0.1, 1e-3, 0.1  # as in tests/test_v0.py


def load(path):
    rows = defaultdict(list)
    for line in path.read_text().splitlines():
        name, *vals = line.split()
        rows[name].append([float(v) for v in vals])
    return {k: np.array(v) for k, v in rows.items()}


def rel_report(got, ref):
    got, ref = np.asarray(got, dtype=np.float64).ravel(), np.asarray(ref, dtype=np.float64).ravel()
    fin_g, fin_r = np.isfinite(got), np.isfinite(ref)
    ok = fin_g & fin_r & (ref != 0)
    rel = np.abs(got[ok] - ref[ok]) / np.abs(ref[ok])
    return {"n": int(got.size), "max_rel": float(rel.max()) if rel.size else None,
            "p999_rel": float(np.quantile(rel, 0.999)) if rel.size else None,
            "median_rel": float(np.median(rel)) if rel.size else None,
            "n_rel_gt_1e-6": int((rel > 1e-6).sum()), "n_rel_gt_1e-3": int((rel > 1e-3).sum()),
            "nonfinite_mismatch": int((fin_g != fin_r).sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, default=ROOT / "v0" / "v0_dump.txt")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    d = load(args.dump)
    rep = {}
    E = d["EeTMax"][:, 0]
    rep["ee_tmax"] = rel_report(electron.ee_tmax(jnp.asarray(E), M), d["EeTMax"][:, 1])
    rep["ee_tmin"] = rel_report(electron.ee_tmin(jnp.asarray(E), M), d["EeTMin"][:, 1])
    t = d["ThetaEe"]
    stock_tmax = dict(zip(d["EeTMax"][:, 0], d["EeTMax"][:, 1]))
    end = np.isclose(t[:, 0], np.array([stock_tmax[e] for e in t[:, 1]]), rtol=1e-14, atol=0)
    rep["theta_ee (endpoint rows excluded)"] = rel_report(
        np.asarray(electron.theta_ee(jnp.asarray(t[~end, 0]), jnp.asarray(t[~end, 1]), M)), t[~end, 2])
    s = d["sigma2"]
    rep["sigma2"] = rel_report(electron.sigma2(jnp.asarray(s[:, 0]), M, MV, KAP, ALD, jnp.asarray(s[:, 1]),
                                               jnp.asarray(s[:, 2])), s[:, 3])
    q = d["dsigmadEe"]
    rep["dsigma_dEe"] = rel_report(electron.dsigma_dEe(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1]), M, MV, KAP, ALD),
                                   q[:, 2])
    for name, det in (("Ldet_sphere", detector.Sphere((0.0, -1.9, 491.0), 5.0)),
                      ("Ldet_cuboid", detector.make_cuboid(0.0, -2.0, 500.0, 3.0, 3.0, 10.0, 0.0, 0.0, 0.0)),
                      ("Ldet_rotated", detector.make_cuboid(0.5, -2.0, 500.0, 3.0, 4.0, 10.0, 0.3, 0.05, -0.2))):
        r = d[name]
        L = np.asarray(detector.ldet(det, jnp.zeros((len(r), 3)), jnp.asarray(r[:, :3]))[0], dtype=np.float64)
        rep[name] = {"n": len(r), "max_abs_m": float(np.abs(L - r[:, 3]).max()),
                     "hit_miss_disagree": int(((L > 0) != (r[:, 3] > 0)).sum())}
    for key, f in (("brpi0_to_gamma_dm_dm", branching.brpi0_to_gamma_dm_dm), ("brpi0toVgamma", branching.brpi0toVgamma),
                   ("brV_to_dm_dm", branching.brV_to_dm_dm)):
        rep[key] = rel_report([f(MV, mx, KAP, ALD) for mx, _ in d[key]], d[key][:, 1])
    if "wpp" in d:
        w = d["wpp"]
        rep["wpp"] = rel_report(brem.wpp(jnp.asarray(w[:, 0]), jnp.asarray(w[:, 1]), jnp.asarray(w[:, 2])), w[:, 3])
        q = d["d2N"]
        rep["d2N"] = rel_report([float(brem.d2N(z, pt, E_, KAP, m)) for E_, m, z, pt, _ in q], q[:, 4])
        rep["V_prod_rate"] = rel_report([brem.ProtonBrem(E_, zmin, zmax, 0.0, ptmax).v_prod_rate(KAP, mA)
                                         for E_, zmin, zmax, ptmax, mA, _ in d["V_prod_rate"]], d["V_prod_rate"][:, 5])
    for key, sw in (("sanfordwang", parents.SanfordWang()), ("sanfordwang_k0", parents.SanfordWang.k0())):
        if key in d:
            rep[key] = rel_report(sw.density(jnp.asarray(d[key][:, 0]), jnp.asarray(d[key][:, 1])), d[key][:, 2])
    for key, bm in (("bmpt_ics", parents.BMPT(120.0, 12.0)), ("bmpt_k0_ics", parents.BMPT(120.0, 12.0, choice="k0"))):
        if key in d:
            rep[key] = rel_report(bm.ics(jnp.asarray(d[key][:, 0]), jnp.asarray(d[key][:, 1])), d[key][:, 2])
    for meson, d2_f in (("pi0", branching.d2brpi0_to_gamma_dm_dm), ("eta", branching.d2breta_to_gamma_dm_dm)):
        key = f"d2br_offshell_{meson}"
        if key not in d:
            continue
        rows = d[key]
        rep[key] = rel_report([d2_f(mv, mx, KAP, ALD, s_, th) for mv, mx, s_, th, _ in rows], rows[:, 4])
        osh, spread = offshell.OffShell(meson), []
        for mv, mx in {(r[0], r[1]) for r in rows}:
            sel = (rows[:, 0] == mv) & (rows[:, 1] == mx)
            shape = np.asarray(osh.shape(jnp.asarray(rows[sel, 2]), jnp.asarray(rows[sel, 3]), osh.m_meson, mv, mx,
                                         offshell.mv_gamma(mv, mx, KAP, ALD)), dtype=np.float64)
            ratio = rows[sel, 4] / shape
            spread.append(float(np.abs(ratio / ratio[0] - 1).max()))
        rep[f"offshell_shape_{meson} (ratio spread)"] = {"max_rel": max(spread)}
    out = {"precision": bdnmc_jax.PRECISION, "dump": str(args.dump), "functions": rep}
    for k, v in rep.items():
        print(f"{k:38s} " + " ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items()))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
