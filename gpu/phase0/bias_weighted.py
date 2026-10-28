"""Measure the running-pMax bias of stock unweighted `se` against stock weighted `se` on the reference points.

Unweighted side: the reference runs (out/ref/<root>_m<MeV>/O2; the 5 MeV brem points reuse out/<base>/O2), with
sigma = se / sqrt(accepted DM scatters) (Poisson on the accepted count). Weighted side: the replicas from
make_weighted_cards.py (out/ref_w/<root>_m<MeV>_r<k>/O2), with sigma = replica std / sqrt(R). Fails if a replica is
missing or ran a different number of trials than its card asked for.
"""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

HERE = Path(__file__).parent
BASES = ["mb_pi0_m005", "he_pi0_cuboid_m005", "mb_brem_m005", "he_brem_cuboid_m005"]
MASSES_MEV = [1, 5, 20, 45]
REUSED = {("mb_brem_m005", 5), ("he_brem_cuboid_m005", 5)}

RE = {
    "se": re.compile(r"^Predicted number of signal events = (\S+)", re.M),
    "accepted": re.compile(r" Events: (\S+) V_num: ", re.M),
    "trials": re.compile(r"^Number of trials = (\d+)", re.M),
    "pmax_burn": re.compile(r"^pMax at end of Burn-In = (\S+)", re.M),
    "pmax_final": re.compile(r"^Maximum event probability = (\S+)", re.M),
}


def grab(text, key, path):
    m = RE[key].findall(text)
    if not m:
        raise ValueError(f"{path}: no '{key}' line")
    return m


def unweighted(base, mev):
    root = base.rsplit("_m", 1)[0]
    d = HERE / "out" / (base if (base, mev) in REUSED else f"ref/{root}_m{mev:03d}") / "O2"
    text = (d / "log.txt").read_text()
    se = float(grab(text, "se", d)[0])
    acc = sum(float(x) for x in grab(text, "accepted", d))
    return {
        "dir": str(d.relative_to(HERE)),
        "se": se,
        "sigma": se / math.sqrt(acc),
        "accepted": acc,
        "trials": int(grab(text, "trials", d)[0]),
        "pmax_burn": float(grab(text, "pmax_burn", d)[0]),
        "pmax_final": float(grab(text, "pmax_final", d)[0]),
    }


def weighted(base, mev, manifest):
    root = base.rsplit("_m", 1)[0]
    reps = [m for m in manifest if m["base"] == base and m["m_chi_MeV"] == mev]
    if not reps:
        raise ValueError(f"no weighted replicas in the manifest for {base} m{mev}")
    ses, trials = [], 0
    for m in reps:
        d = HERE / "out" / m["stem"] / "O2"
        text = (d / "log.txt").read_text()
        n = int(grab(text, "trials", d)[0])
        if n != m["max_trials"]:
            raise ValueError(f"{d}: {n} trials, card asked for {m['max_trials']}")
        ses.append(float(grab(text, "se", d)[0]))
        trials += n
    mean = statistics.fmean(ses)
    return {"stem": f"ref_w/{root}_m{mev:03d}_r*", "replicas": len(ses), "trials": trials, "se": mean,
            "sigma": statistics.stdev(ses) / math.sqrt(len(ses)), "replica_se": ses}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=HERE / "out" / "bias_weighted.json")
    args = ap.parse_args()
    manifest = json.loads((HERE / "cards" / "ref_w" / "MANIFEST.json").read_text())
    rows = []
    for base in BASES:
        for mev in MASSES_MEV:
            u, w = unweighted(base, mev), weighted(base, mev, manifest)
            ratio = u["se"] / w["se"]
            sig = ratio * math.hypot(u["sigma"] / u["se"], w["sigma"] / w["se"])
            growth = u["pmax_final"] / u["pmax_burn"] - 1
            rows.append({"card": base.rsplit("_m", 1)[0], "m_chi_MeV": mev, "unweighted": u, "weighted": w,
                         "bias": ratio - 1, "bias_sigma": sig, "pull": (ratio - 1) / sig, "pmax_growth": growth})
    print("| card | m_chi (MeV) | unweighted se | weighted se (R) | bias | pull | pMax growth after burn-in |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        u, w = r["unweighted"], r["weighted"]
        print(f"| {r['card']} | {r['m_chi_MeV']} | {u['se']:.5g} ± {u['sigma']:.2g} | "
              f"{w['se']:.5g} ± {w['sigma']:.2g} ({w['replicas']}) | {100 * r['bias']:+.2f} ± "
              f"{100 * r['bias_sigma']:.2f} % | {r['pull']:+.1f} | {100 * r['pmax_growth']:.1f} % |")
    print("\n| card | weighted mean bias | chi2 / ndf against zero bias |\n|---|---|---|")
    for card in dict.fromkeys(r["card"] for r in rows):
        sub = [r for r in rows if r["card"] == card]
        w = [1 / r["bias_sigma"] ** 2 for r in sub]
        mean = sum(wi * r["bias"] for wi, r in zip(w, sub)) / sum(w)
        chi2 = sum(r["pull"] ** 2 for r in sub)
        print(f"| {card} | {100 * mean:+.3f} ± {100 / math.sqrt(sum(w)):.3f} % | {chi2:.1f} / {len(sub)} |")
    args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
