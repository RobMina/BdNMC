"""Summarise Phase 0 stock runs: BdNMC trial accounting plus wall/CPU/RSS from `time -l`."""
import argparse
import json
import re
from pathlib import Path

LOG_KEYS = {
    "seed": r"Seed=(\d+)",
    "pmax_burnin": r"pMax at end of Burn-In = (\S+)",
    "trials": r"Number of trials = (\d+)",
    "hit": r"Number of candidates intersecting detector = (\d+)",
    "events": r"Number of sample events generated = (\d+)",
    "se": r"Predicted number of signal events = (\S+)",
    "prob_max": r"Maximum event probability = (\S+)",
    "prob_mean": r"Average event probability = (\S+)",
}
TIME_KEYS = {
    "real_s": r"([\d.]+) real",
    "user_s": r"([\d.]+) user",
    "sys_s": r"([\d.]+) sys",
    "rss_bytes": r"(\d+)\s+maximum resident set size",
}


def grab(text, patterns):
    out = {}
    for key, pat in patterns.items():
        m = re.search(pat, text)
        out[key] = float(m.group(1)) if m else None
    return out


def summarise_run(run_dir):
    log = (run_dir / "log.txt").read_text(errors="replace")
    tim = (run_dir / "time.txt").read_text(errors="replace") if (run_dir / "time.txt").exists() else ""
    row = {"card": run_dir.parent.name, "build": run_dir.name}
    row.update(grab(log, LOG_KEYS))
    row.update(grab(tim, TIME_KEYS))
    t, cpu = row["trials"], (row["user_s"] or 0) + (row["sys_s"] or 0)
    if t and cpu:
        row["trials_per_cpu_s"] = t / cpu
        row["events_per_cpu_s"] = (row["events"] or 0) / cpu
    if t:
        row["hit_frac"] = (row["hit"] or 0) / t
        row["accept_per_trial"] = (row["events"] or 0) / t
    if row["prob_max"] and row["prob_mean"]:
        row["unweight_eff"] = row["prob_mean"] / row["prob_max"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    ap.add_argument("--json", type=Path, help="write rows as JSON here")
    args = ap.parse_args()
    rows = [summarise_run(d) for d in sorted(args.out.glob("*/*")) if (d / "log.txt").exists()]
    cols = ["card", "build", "trials", "hit_frac", "accept_per_trial", "unweight_eff", "events",
            "se", "real_s", "user_s", "trials_per_cpu_s", "events_per_cpu_s", "rss_bytes"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
        print("| " + " | ".join(cells) + " |")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
