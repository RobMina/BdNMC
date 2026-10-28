"""Compare two BdNMC comprehensive event files from the same seed, value by value.

Structure (event markers, particle labels, token counts) must match exactly; numeric tokens are compared and the
largest absolute and relative difference is reported per (particle label, column). `Run` header lines are skipped.
"""
import argparse
import itertools
import json
import math
import sys
from pathlib import Path


def lines(path):
    with open(path) as f:
        for line in f:
            if not line.startswith("Run "):
                yield line.split()


def compare(a_path, b_path, rel_tol):
    stats = {}
    n_lines = n_diff_lines = n_events = 0
    worst = None
    for i, (a, b) in enumerate(itertools.zip_longest(lines(a_path), lines(b_path))):
        if a is None or b is None:
            return {"structure": f"length mismatch at line {i}"}
        if a == b:
            n_lines += 1
            if a and a[0] == "event":
                n_events += 1
            continue
        if len(a) != len(b) or a[0] != b[0]:
            return {"structure": f"token/label mismatch at line {i}: {a[:3]} vs {b[:3]}"}
        n_lines += 1
        n_diff_lines += 1
        label = a[0]
        for col, (x, y) in enumerate(zip(a[1:], b[1:]), start=1):
            if x == y:
                continue
            fx, fy = float(x), float(y)
            d = abs(fx - fy)
            r = d / max(abs(fx), abs(fy)) if max(abs(fx), abs(fy)) > 0 else 0.0
            key = f"{label}[{col}]"
            s = stats.setdefault(key, {"n": 0, "max_abs": 0.0, "max_rel": 0.0})
            s["n"] += 1
            s["max_abs"] = max(s["max_abs"], d)
            s["max_rel"] = max(s["max_rel"], r)
            if worst is None or r > worst["rel"]:
                worst = {"line": i, "key": key, "a": x, "b": y, "rel": r}
    max_rel = max((s["max_rel"] for s in stats.values()), default=0.0)
    return {
        "structure": "identical",
        "lines": n_lines,
        "events": n_events,
        "lines_with_numeric_diff": n_diff_lines,
        "max_rel": max_rel,
        "pass": max_rel <= rel_tol and math.isfinite(max_rel),
        "worst": worst,
        "per_column": stats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--rel-tol", type=float, default=1e-8, help="largest allowed relative difference")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    res = compare(args.a, args.b, args.rel_tol)
    res.update({"a": str(args.a), "b": str(args.b), "rel_tol": args.rel_tol})
    if args.json:
        args.json.write_text(json.dumps(res, indent=1))
    brief = {k: res.get(k) for k in ("structure", "lines", "events", "lines_with_numeric_diff", "max_rel", "pass", "worst")}
    print(json.dumps(brief))
    sys.exit(0 if res.get("pass") else 1)


if __name__ == "__main__":
    main()
