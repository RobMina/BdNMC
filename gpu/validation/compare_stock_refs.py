"""Stock reference against a repeat stock reference: the V2 histograms and tail checks of compare_v.py with the repeat
in the port's place (unit weights), binned and cut on the first reference as compare_v.py does.

Usage: compare_stock_refs.py --config v_config_eta.json --repeat-dir validation/out/stock_r2 --points stem ...
       [--json out.json]   (first reference: validation/out/stock/<stem>.npz)
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import compare_v as cv

ROOT = cv.ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repeat-dir", type=Path, required=True)
    ap.add_argument("--points", nargs="+", required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    v2c = cfg["V2"]
    out = {"config": args.config.name, "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
           "repeat_dir": str(args.repeat_dir), "points": {}}
    for stem in args.points:
        a = np.load(ROOT / "validation" / "out" / "stock" / f"{stem}.npz")
        b = np.load(args.repeat_dir / f"{stem}.npz")
        va = cv.variables(a["electron"], a["dm"], a["parent"], a["vertex"])
        vb = cv.variables(b["electron"], b["dm"], b["parent"], b["vertex"])
        w = np.ones(len(b["electron"]))
        hist = {name: cv.chi2_test(va[name], vb[name], w, v2c["n_bins"], v2c["min_expected_per_bin"])
                for name in v2c["variables"]}
        cond = v2c["conditional"]
        q = cond["in_quantiles_of"]
        qs = np.quantile(va[q], np.linspace(0, 1, cond["n_slices"] + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        for k in range(cond["n_slices"]):
            ma, mb = (va[q] > qs[k]) & (va[q] <= qs[k + 1]), (vb[q] > qs[k]) & (vb[q] <= qs[k + 1])
            hist[f"{cond['variable']}|{q}_q{k}"] = cv.chi2_test(va[cond["variable"]][ma], vb[cond["variable"]][mb],
                                                               w[mb], v2c["n_bins"], v2c["min_expected_per_bin"])
        tails = {name: cv.tail_test(va[name], vb[name], w, v2c["tails"]["stock_quantile"])
                 for name in v2c["tails"]["variables"]}
        out["points"][stem] = {"n_first": len(a["electron"]), "n_repeat": len(b["electron"]), "V2": hist,
                               "tails": tails}
        low = {k: round(h["p"], 4) for k, h in hist.items() if "p" in h and h["p"] < 0.05}
        print(f"{stem}: {len(a['electron'])} vs {len(b['electron'])} electrons; p < 0.05: {low or 'none'}; tail pulls "
              + ", ".join(f"{k} {t['pull']:+.2f}" for k, t in tails.items()))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
