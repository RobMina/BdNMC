"""Attribute macOS `sample` call-graph counts to BdNMC stages.

Stages are the direct callees of BdNMCSession::run() (the trial loop); anything outside run() is setup or burn-in,
which run_stock.sh only partly samples. Cross-cutting totals (RNG, detector, form factor) are inclusive counts summed
over disjoint subtrees.
"""
import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r"^(?P<lead>[\s+!:|]*)(?P<n>\d+) (?P<fn>.+?)  \(in ")

STAGES = [  # (stage, substring of a run() callee)
    ("parent sampling", "sample_particle"),
    ("parent sampling", "Sample_Particle"),
    ("decay + detector test", "GenDM"),
    ("scatter + recoil", "probscatter"),
    ("output", "Record_Particles"),
    ("output", "record_event"),
    ("output", "ostream"),
    ("output", "operator<<"),
    ("channel choice", "Random::Flat"),
]
CROSS = {
    "RNG (Random::Flat, incl. mutex)": "Random::Flat(",
    "  of which mutex lock/unlock": "mutex",
    "detector Ldet": "::Ldet(",
    "proton form factor F_1_proton": "F_1_proton(",
    "  of which complex division": "__divdc3",
    "list allocation (malloc/free)": "malloc",
}


def stage_of(fn):
    for stage, key in STAGES:
        if key in fn:
            return stage
    return "other in run()"


def parse(path):
    text = Path(path).read_text(errors="replace")
    graph = text.split("Call graph:", 1)[1].split("Total number in stack", 1)[0]
    stack, stages, cross, total, run_total = [], {}, {k: 0 for k in CROSS}, 0, 0
    for raw in graph.splitlines():
        m = LINE.match(raw)
        if not m:
            continue
        depth, n, fn = len(m.group("lead")), int(m.group("n")), m.group("fn")
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if not stack:
            total += n
        parent = stack[-1][1] if stack else ""
        is_run = fn.startswith("BdNMCSession::run(")
        # Inlined code shows as run() nested in run(); attribute through it, count only the outermost.
        if parent.startswith("BdNMCSession::run(") and not is_run:
            s = stage_of(fn)
            stages[s] = stages.get(s, 0) + n
        if is_run and not any(f.startswith("BdNMCSession::run(") for _, f in stack):
            run_total += n
        for key, pat in CROSS.items():
            if pat in fn and not any(pat in f for _, f in stack):
                cross[key] += n
        stack.append((depth, fn))
    stages["outside run() (sampling starts 5 s in)"] = total - run_total
    return {"samples": total, "stages": stages, "cross": cross}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("profiles", nargs="+", type=Path, help="sample output files (<card>/<build>/profile.txt)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}
    for p in args.profiles:
        card = p.parent.parent.name
        res = parse(p)
        out[card] = res
        tot = res["samples"]
        print(f"== {card}: {tot} samples")
        for k, v in sorted(res["stages"].items(), key=lambda kv: -kv[1]):
            print(f"   {k:34s} {100 * v / tot:5.1f} %")
        for k, v in res["cross"].items():
            print(f"   [{k}] {100 * v / tot:5.1f} %")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
