"""Physics of a float32 run of the port (BDNMC_JAX_PRECISION=float32, see bdnmc_jax) against a float64 run, point by
point, with the V1 and V2 statistics of compare_v.py:
- V1: weighted se of the two runs, pull with both sigmas; the unweighted se and the accepted counts;
- V2: the config's frozen histograms, conditional slices and tail tests, with the float64 run's accepted events
  (unweighted, like a stock sample) as the reference and the float32 run's live DM weighted by probability;
- the samplers' counters (parent proposals over the bar, hits, pMax).
The two precisions draw their uniforms from different random bits, so the runs are independent samples and the
statistics have their usual null distributions. The verdicts use the config's V1/V2 thresholds.
Usage: compare_precision.py --config v_config.json --ref-dir <float64 shards> --test-dir <float32 shards>
       [--json out.json] [--points stem ...]  (shards in <dir>/<stem>/shard*.npz, run_v_set.sh layout)
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "validation"))
from bdnmc_jax import pipeline, shards  # noqa: E402
from compare_v import port_v1, port_v2_sample, v2_tests, variables  # noqa: E402


def load(d, stem):
    res = pipeline.merge(shards.load_all((d / stem).glob("shard*.npz")))
    if res["trials"] != res["n_trials"]:
        raise ValueError(f"{d / stem}: shards cover {res['trials']} of {res['n_trials']} trials")
    return res


def accepted_sample(res):
    """The run's unweighted accepted events (pass 2), as variables."""
    c = res["candidates"]
    acc = pipeline.finish(res)["accepted_mask"]
    return variables(c["electron"][acc], c["dm"][acc], c["parent"][acc], c["vertex"][acc])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ref-dir", type=Path, required=True, help="float64 shards")
    ap.add_argument("--test-dir", type=Path, required=True, help="float32 shards")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--points", nargs="+")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    v1c, v2c = cfg["V1"], cfg["V2"]
    out = {"config": args.config.name, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "ref_dir": str(args.ref_dir), "test_dir": str(args.test_dir), "points": {}}
    pulls, pvals, tails, fails = [], [], [], []
    for stem in args.points or cfg["points"]:
        r64, r32 = load(args.ref_dir, stem), load(args.test_dir, stem)
        dtypes = {k: str(r32["candidates"][k].dtype) for k in ("prob", "electron")}
        a, b = port_v1(r64), port_v1(r32)
        pull = (b["se_weighted"] - a["se_weighted"]) / math.hypot(a["sigma_weighted"], b["sigma_weighted"])
        pulls.append(pull)
        vs = accepted_sample(r64)
        vp, w = port_v2_sample(r32)
        hist, tl = v2_tests(vs, vp, w, v2c)
        for name, h in hist.items():
            if "fail" in h:
                fails.append(f"{stem} {name}: {h['fail']}")
            else:
                pvals.append(h["p"])
        tails += [t["pull"] for t in tl.values()]
        clipped = {"float64": float((vs["theta_e"] == 0).mean()),
                   "float32": float((vp["theta_e"] == 0).mean())}  # electron.theta_ee clips float32 to angle 0
        out["points"][stem] = {"float64": a, "float32": b, "float32_dtypes": dtypes, "pull_se_weighted": pull,
                               "fraction_theta_e_zero": clipped,
                               "rel_diff_se_weighted": b["se_weighted"] / a["se_weighted"] - 1,
                               "rel_diff_se_unweighted": b["se_unweighted"] / a["se_unweighted"] - 1,
                               "rel_diff_pmax": b["pmax"] / a["pmax"] - 1,
                               "rel_diff_hits_per_trial": (b["hits"] / b["trials"]) / (a["hits"] / a["trials"]) - 1,
                               "V2": hist, "tails": tl}
        print(f"{stem}: se_w f64 {a['se_weighted']:.6g} ± {a['sigma_weighted']:.2g}, f32 {b['se_weighted']:.6g} ± "
              f"{b['sigma_weighted']:.2g}, pull {pull:+.2f}; theta_e = 0: {clipped['float32']:.2e}; pMax {b['pmax'] / a['pmax'] - 1:+.2e}; min V2 p "
              f"{min((h.get('p', 0) for h in hist.values())):.3g}; prob dtype {dtypes['prob']}")
    n = len(pulls)
    chi2 = sum(p * p for p in pulls)
    v1 = {"max_abs_pull": max(abs(p) for p in pulls), "mean_pull": statistics.fmean(pulls), "chi2": chi2, "ndf": n,
          "chi2_p": float(stats.chi2.sf(chi2, n))}
    v1["pass"] = bool(v1["max_abs_pull"] <= v1c["max_abs_pull_per_point"]
                      and abs(v1["mean_pull"]) * math.sqrt(n) <= v1c["max_abs_mean_pull_times_sqrt_n"]
                      and v1["chi2_p"] >= v1c["min_chi2_p_value_over_points"])
    ks = stats.kstest(pvals, "uniform").pvalue if pvals else 0.0
    v2 = {"n_tests": len(pvals), "n_failed_closed": len(fails), "failed_closed": fails, "ks_uniform_p": float(ks),
          "min_p": float(min(pvals)) if pvals else None, "max_abs_tail_pull": max(abs(t) for t in tails)}
    v2["pass"] = bool(not fails and ks > v2c["pass_ks_uniform_p_min"] and v2["min_p"] > v2c["pass_min_single_p"]
                      and v2["max_abs_tail_pull"] <= v2c["tails"]["max_abs_pull"])
    out |= {"V1": v1, "V2": v2}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=lambda o: o.item()))
    print(json.dumps({"V1": v1, "V2": {k: v for k, v in v2.items() if k != "failed_closed"}}, indent=1))
    for f in fails:
        print("FAILED CLOSED:", f)


if __name__ == "__main__":
    main()
