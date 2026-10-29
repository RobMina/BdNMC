"""V1 (normalisation) and V2 (distributions) of the JAX port against the stock reference sets, with the thresholds
and binning frozen in v_config.json (its sha256 goes into the result).

Inputs per point <stem>: port shards validation/out/v/<stem>/shard*.npz; stock electrons
validation/out/stock/<stem>.npz (extract_stock.py); stock unweighted log phase0/out/ref/<stem>/O2/log.txt; stock
weighted replica logs phase0/out/ref_w/<stem>_r*/O2/log.txt.
Usage: compare_v.py [--config v_config.json] [--port-dir dir] [--json out.json] [--stock-dir dir] [--points stem ...]
(port shards in <port-dir>/<stem>/, default validation/out/v). A config's "unweighted_dirs" maps a point to its stock
unweighted run directory when that is not phase0/out/ref/<stem>/O2. --stock-dir takes the stock electrons from
another directory (a repeat reference) and --points runs a subset; V1 and the overall verdicts then cover only those
points, so such a result is not a V1/V2 verdict of the config.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bdnmc_jax import pipeline, rng, shards  # noqa: E402

CONFIG = ROOT / "validation" / "v_config.json"
SE = re.compile(r"^Predicted number of signal events = (\S+)", re.M)
ACC = re.compile(r" Events: (\S+) V_num: ", re.M)


# --- V1 ------------------------------------------------------------------------------------------------------------

def stock_unweighted(stem, run_dir=None):
    text = (ROOT / (run_dir or f"phase0/out/ref/{stem}/O2") / "log.txt").read_text()
    se = float(SE.search(text).group(1))
    return se, se / math.sqrt(sum(float(x) for x in ACC.findall(text)))


def stock_weighted(stem):
    logs = sorted((ROOT / "phase0" / "out" / "ref_w").glob(f"{stem}_r*/O2/log.txt"))
    ses = [float(SE.search(p.read_text()).group(1)) for p in logs]
    if len(ses) < 2:
        raise ValueError(f"{stem}: {len(ses)} weighted replicas")
    return statistics.fmean(ses), statistics.stdev(ses) / math.sqrt(len(ses)), len(ses)


def port_v1(res):
    c, n, scale = res["candidates"], res["trials"], res["vnum"] * res["efficiency"]
    w = np.where(c["ang_ok"], c["prob"], 0.0)
    trial_id = c["block"].astype(np.int64) * rng.BLOCK + c["trial"]
    _, inv = np.unique(trial_id, return_inverse=True)
    per_trial = np.bincount(inv, weights=w)
    var = (per_trial ** 2).sum() - per_trial.sum() ** 2 / n
    se_w = w.sum() / n * scale
    out = pipeline.finish(res)
    return {"se_weighted": se_w, "sigma_weighted": math.sqrt(var) / n * scale,
            "se_unweighted": out["se"], "sigma_unweighted": out["se"] / math.sqrt(max(out["accepted"], 1)),
            "accepted": out["accepted"], "pmax": out["pmax"], "trials": n, "hits": res["n_hits"],
            "parent_over_bar": res["parent_over_bar"]}


# --- V2 ------------------------------------------------------------------------------------------------------------

def _theta(p4):
    return np.arccos(p4[:, 3] / np.linalg.norm(p4[:, 1:], axis=1))


def variables(electron, dm, parent, vertex):
    th = _theta(electron)
    return {"E_e": electron[:, 0], "theta_e": th, "E_e_theta_e2": electron[:, 0] * th ** 2, "E_chi": dm[:, 0],
            "theta_chi": _theta(dm), "vertex_x": vertex[:, 0], "vertex_y": vertex[:, 1], "vertex_z": vertex[:, 2],
            "parent_p": np.linalg.norm(parent[:, 1:], axis=1), "parent_theta": _theta(parent)}


def chi2_test(xs, xp, w, nbins, min_expected):
    """Two-sample chi2 of normalised shapes: stock unweighted counts against port weighted sums."""
    if not (np.isfinite(xs).all() and np.isfinite(xp).all() and np.isfinite(w).all()):
        return {"fail": "non-finite values"}
    edges = np.quantile(xs, np.linspace(0, 1, nbins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    if np.any(np.diff(edges) <= 0):
        return {"fail": "degenerate stock quantiles"}
    n = np.histogram(xs, edges)[0].astype(float)
    wi = np.histogram(xp, edges, weights=w)[0]
    si = np.histogram(xp, edges, weights=w * w)[0]
    if n.min() < min_expected:
        return {"fail": f"stock bin with {n.min():.0f} < {min_expected}"}
    if (wi <= 0).any():
        return {"fail": "empty port bin"}
    N, W = n.sum(), wi.sum()
    chi2 = float((((W * n - N * wi) ** 2) / (W * W * n + N * N * si)).sum())
    ndf = nbins - 1
    return {"chi2": chi2, "ndf": ndf, "p": float(stats.chi2.sf(chi2, ndf)), "n_stock": int(N),
            "n_port": int(len(xp)), "port_eff_n": float(W * W / si.sum())}


def tail_test(xs, xp, w, q):
    cut = np.quantile(xs, q)
    fs = (xs > cut).mean()
    ss = math.sqrt(fs * (1 - fs) / len(xs))
    W = w.sum()
    fp = w[xp > cut].sum() / W
    sp = math.sqrt((w[xp > cut] ** 2).sum()) / W
    return {"cut": float(cut), "stock": float(fs), "port": float(fp), "pull": float((fp - fs) / math.hypot(ss, sp))}


def port_v2_sample(res):
    c = res["candidates"]
    live = c["ang_ok"] & (c["prob"] > 0)
    return variables(c["electron"][live], c["dm"][live], c["parent"][live], c["vertex"][live]), c["prob"][live]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--port-dir", type=Path, default=ROOT / "validation" / "out" / "v")
    ap.add_argument("--json", type=Path, default=ROOT / "validation" / "out" / "v_result.json")
    ap.add_argument("--stock-dir", type=Path, default=ROOT / "validation" / "out" / "stock")
    ap.add_argument("--points", nargs="+")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    v1c, v2c = cfg["V1"], cfg["V2"]
    points = args.points or cfg["points"]
    if not set(points) <= set(cfg["points"]):
        raise ValueError(f"points not in {args.config.name}: {sorted(set(points) - set(cfg['points']))}")
    out = {"config": args.config.name, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "port_dir": str(args.port_dir), "stock_dir": str(args.stock_dir), "points": {}}
    if args.points:
        out["subset"] = points
    pulls, pvals, tails, fails = [], [], [], []
    for stem in points:
        res = pipeline.merge(shards.load_all((args.port_dir / stem).glob("shard*.npz")))
        if res["trials"] != res["n_trials"]:
            raise ValueError(f"{stem}: shards cover {res['trials']} of {res['n_trials']} trials")
        p1 = port_v1(res) | {"backend": res.get("backend", "cpu"), "device": res.get("device"),
                             "wall_s": res.get("wall_s")}
        sw, sw_sig, nrep = stock_weighted(stem)
        su, su_sig = stock_unweighted(stem, cfg.get("unweighted_dirs", {}).get(stem))
        pull = (p1["se_weighted"] - sw) / math.hypot(p1["sigma_weighted"], sw_sig)
        pulls.append(pull)
        point = {"port": p1, "stock_weighted": [sw, sw_sig, nrep], "stock_unweighted": [su, su_sig],
                 "pull_main": pull,
                 "rel_diff_secondary": p1["se_weighted"] / su - 1,
                 "secondary_ok": abs(p1["se_weighted"] - su) <= max(0.01 * su, 2 * math.hypot(p1["sigma_weighted"], su_sig)),
                 "port_unweighted_vs_stock_weighted_pull":
                     (p1["se_unweighted"] - sw) / math.hypot(p1["sigma_unweighted"], sw_sig)}
        xs_all = np.load(args.stock_dir / f"{stem}.npz")
        vs = variables(xs_all["electron"], xs_all["dm"], xs_all["parent"], xs_all["vertex"])
        vp, w = port_v2_sample(res)
        hist = {}
        for name in v2c["variables"]:
            hist[name] = chi2_test(vs[name], vp[name], w, v2c["n_bins"], v2c["min_expected_per_bin"])
        cond = v2c["conditional"]
        qs = np.quantile(vs[cond["in_quantiles_of"]], np.linspace(0, 1, cond["n_slices"] + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        for k in range(cond["n_slices"]):
            ms = (vs[cond["in_quantiles_of"]] > qs[k]) & (vs[cond["in_quantiles_of"]] <= qs[k + 1])
            mp = (vp[cond["in_quantiles_of"]] > qs[k]) & (vp[cond["in_quantiles_of"]] <= qs[k + 1])
            hist[f"{cond['variable']}|{cond['in_quantiles_of']}_q{k}"] = chi2_test(
                vs[cond["variable"]][ms], vp[cond["variable"]][mp], w[mp], v2c["n_bins"], v2c["min_expected_per_bin"])
        for name, h in hist.items():
            if "fail" in h:
                fails.append(f"{stem} {name}: {h['fail']}")
            else:
                pvals.append(h["p"])
        tl = {name: tail_test(vs[name], vp[name], w, v2c["tails"]["stock_quantile"]) for name in v2c["tails"]["variables"]}
        tails += [t["pull"] for t in tl.values()]
        point |= {"V2": hist, "tails": tl}
        out["points"][stem] = point
        print(f"{stem}: port se_w {p1['se_weighted']:.5g} ± {p1['sigma_weighted']:.2g}, stock se_w {sw:.5g} ± "
              f"{sw_sig:.2g}, pull {pull:+.2f}; min V2 p {min((h.get('p', 0) for h in hist.values())):.3g}")
    n = len(pulls)
    chi2 = sum(p * p for p in pulls)
    v1 = {"max_abs_pull": max(abs(p) for p in pulls), "mean_pull": statistics.fmean(pulls),
          "chi2": chi2, "ndf": n, "chi2_p": float(stats.chi2.sf(chi2, n))}
    v1["pass"] = (v1["max_abs_pull"] <= v1c["max_abs_pull_per_point"]
                  and abs(v1["mean_pull"]) * math.sqrt(n) <= v1c["max_abs_mean_pull_times_sqrt_n"]
                  and v1["chi2_p"] >= v1c["min_chi2_p_value_over_points"])
    ks = stats.kstest(pvals, "uniform").pvalue if pvals else 0.0
    v2 = {"n_tests": len(pvals), "n_failed_closed": len(fails), "failed_closed": fails, "ks_uniform_p": float(ks),
          "min_p": float(min(pvals)) if pvals else None, "max_abs_tail_pull": max(abs(t) for t in tails)}
    v2["pass"] = (not fails and ks > v2c["pass_ks_uniform_p_min"] and v2["min_p"] > v2c["pass_min_single_p"]
                  and v2["max_abs_tail_pull"] <= v2c["tails"]["max_abs_pull"])
    out |= {"V1": v1, "V2": v2}
    args.json.write_text(json.dumps(out, indent=1, default=lambda o: o.item()))  # numpy scalars
    print(json.dumps({"V1": v1, "V2": {k: v for k, v in v2.items() if k != "failed_closed"}}, indent=1))
    for f in fails:
        print("FAILED CLOSED:", f)


if __name__ == "__main__":
    main()
