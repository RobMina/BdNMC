"""Compare the long stock weighted he_pi0 runs (out/ref_wl) with the short weighted replicas (out/ref_w), the stock
unweighted reference (out/ref) and the port (validation/out/v_result.json).

A single run has no replica spread, so its sigma is the short replicas' spread scaled by sqrt(trials per replica /
trials of the long run) (same per-trial variance). Unweighted sigma: Poisson on the accepted count.
Usage: compare_long_weighted.py  (writes out/long_weighted.json)
"""
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).parent
ROOT = HERE.parent
SE = re.compile(r"^Predicted number of signal events = (\S+)", re.M)
TRIALS = re.compile(r"^Number of trials = (\d+)", re.M)
ACC = re.compile(r" Events: (\S+) V_num: ", re.M)
MASSES = ["001", "005", "020", "045"]


def read(log):
    text = log.read_text()
    return float(SE.search(text).group(1)), int(TRIALS.search(text).group(1)), text


def pull(a, sa, b, sb):
    return (a - b) / math.hypot(sa, sb)


def main():
    port = json.loads((ROOT / "validation" / "out" / "v_result.json").read_text())["points"]
    rows = {}
    for m in MASSES:
        stem = f"he_pi0_cuboid_m{m}"
        lw, n_long, _ = read(HERE / "out" / "ref_wl" / stem / "O2" / "log.txt")
        short = [read(p) for p in sorted((HERE / "out" / "ref_w").glob(f"{stem}_r*/O2/log.txt"))]
        ses, n_rep = [s for s, _, _ in short], short[0][1]
        sw, sd = statistics.fmean(ses), statistics.stdev(ses)
        sw_sig, lw_sig = sd / math.sqrt(len(ses)), sd * math.sqrt(n_rep / n_long)
        su, _, text = read(HERE / "out" / "ref" / stem / "O2" / "log.txt")
        su_sig = su / math.sqrt(sum(float(x) for x in ACC.findall(text)))
        p = port[stem]["port"]
        pw, pw_sig = p["se_weighted"], p["sigma_weighted"]
        rows[stem] = {
            "long_weighted": [lw, lw_sig, n_long], "short_weighted": [sw, sw_sig, len(ses), n_rep],
            "unweighted": [su, su_sig], "port_weighted": [pw, pw_sig],
            "long_minus_short_pct": 100 * (lw / sw - 1), "pull_long_minus_short": pull(lw, lw_sig, sw, sw_sig),
            "port_minus_long_pct": 100 * (pw / lw - 1), "pull_port_minus_long": pull(pw, pw_sig, lw, lw_sig),
            "port_minus_short_pct": 100 * (pw / sw - 1), "pull_port_minus_short": pull(pw, pw_sig, sw, sw_sig),
            "unweighted_minus_long_pct": 100 * (su / lw - 1), "pull_unweighted_minus_long": pull(su, su_sig, lw, lw_sig),
        }
        r = rows[stem]
        print(f"{stem}: long {lw:.6g} ± {lw_sig:.2g} | short {sw:.6g} ± {sw_sig:.2g} | port {pw:.6g} ± {pw_sig:.2g} "
              f"| unweighted {su:.6g} ± {su_sig:.2g}")
        print(f"    long-short {r['long_minus_short_pct']:+.3f} % ({r['pull_long_minus_short']:+.2f}), "
              f"port-long {r['port_minus_long_pct']:+.3f} % ({r['pull_port_minus_long']:+.2f}), "
              f"port-short {r['port_minus_short_pct']:+.3f} % ({r['pull_port_minus_short']:+.2f}), "
              f"unweighted-long {r['unweighted_minus_long_pct']:+.3f} % ({r['pull_unweighted_minus_long']:+.2f})")
    summary = {}
    for key in ("pull_long_minus_short", "pull_port_minus_long", "pull_port_minus_short", "pull_unweighted_minus_long"):
        ps = [r[key] for r in rows.values()]
        summary[key] = {"mean": statistics.fmean(ps), "chi2": sum(x * x for x in ps), "ndf": len(ps)}
        print(f"{key}: mean {summary[key]['mean']:+.2f}, chi2 {summary[key]['chi2']:.1f} / {len(ps)}")
    (HERE / "out" / "long_weighted.json").write_text(json.dumps({"points": rows, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
