"""One line per result file of compare_v.py or compare_precision.py: the V1 and V2 verdicts and statistics; with
--points, one line per point as well (pull, and for a precision result the relative se difference, the fraction of
electrons at theta_e = 0, and the smallest V2 p-value).
Usage: summarize_v.py result.json [...] [--points [--hists]]
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--points", action="store_true")
    ap.add_argument("--hists", action="store_true", help="with --points: every V2 histogram's p-value")
    args = ap.parse_args()
    for f in args.files:
        d = json.loads(f.read_text())
        if "V1" not in d:
            continue  # not a compare_v / compare_precision result
        v1, v2 = d["V1"], d["V2"]
        print(f"{f.name:28s} V1 {'pass' if v1['pass'] else 'FAIL'} max|pull| {v1['max_abs_pull']:.2f} mean "
              f"{v1['mean_pull']:+.2f} chi2 p {v1['chi2_p']:.2f} | V2 {'pass' if v2['pass'] else 'FAIL'} KS p "
              f"{v2['ks_uniform_p']:.2f} min p {v2['min_p']:.2g} max|tail pull| {v2['max_abs_tail_pull']:.2f} "
              f"failed closed {v2['n_failed_closed']}")
        if not args.points:
            continue
        for stem, p in d["points"].items():
            pull = p.get("pull_main", p.get("pull_se_weighted"))
            min_p = min((h.get("p", 0.0) for h in p["V2"].values()), default=None)
            extra = ""
            if "rel_diff_se_weighted" in p:
                extra = (f" se_w f32/f64-1 {p['rel_diff_se_weighted']:+.2e}"
                         f" theta_e=0 {p['fraction_theta_e_zero']['float32']:.2e}")
            print(f"    {stem:24s} pull {pull:+.2f}{extra} min V2 p {min_p:.2g}")
            if args.hists:
                print("        " + "  ".join(f"{k} {h['p']:.1e}" if "p" in h else f"{k} {h['fail']}"
                                             for k, h in p["V2"].items()))


if __name__ == "__main__":
    main()
