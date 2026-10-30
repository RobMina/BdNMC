"""Join the electrons of split stock reference runs (phase0/make_split_ref_cards.py) into one V2 file per point.

Per point of cards/ref_s/MANIFEST.json, every part must have finished (phase0/out/<part stem>/<tag>/log.txt with the
signal-event line and an events.dat); the parts' extract_stock.py arrays are concatenated in part order, with a "part"
array giving each electron's part, into <out-dir>/<point>.npz. V1 combines the parts' logs in compare_v.py.
Usage: merge_stock_parts.py [--tag O2_uva] [--out-dir validation/out/stock] [--points point ...]
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import numpy as np

from extract_stock import extract

ROOT = Path(__file__).resolve().parents[1]
TRIALS = re.compile(r"^Number of trials = (\d+)", re.M)
ACC = re.compile(r" Events: (\S+) V_num: ", re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="O2_uva", help="run directory name under phase0/out/<stem>/")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "validation" / "out" / "stock")
    ap.add_argument("--points", nargs="+")
    args = ap.parse_args()
    parts = defaultdict(list)
    for m in json.loads((ROOT / "phase0" / "cards" / "ref_s" / "MANIFEST.json").read_text()):
        parts[m["point"]].append(m)
    points = args.points or list(parts)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for point in points:
        ms = sorted(parts[point], key=lambda m: m["part"])
        if [m["part"] for m in ms] != list(range(ms[0]["n_parts"])):
            raise ValueError(f"{point}: manifest parts {[m['part'] for m in ms]}")
        arrays, trials, acc = defaultdict(list), 0, 0.0
        for m in ms:
            run = ROOT / "phase0" / "out" / m["stem"] / args.tag
            text = (run / "log.txt").read_text()
            if "Predicted number of signal events =" not in text:
                raise ValueError(f"{run}: unfinished run")
            trials += int(TRIALS.search(text).group(1))
            acc += sum(float(x) for x in ACC.findall(text))
            for k, v in extract(run / "events.dat").items():
                arrays[k].append(v)
            arrays["part"].append(np.full(len(arrays["electron"][-1]), m["part"], dtype=np.int32))
        d = {k: np.concatenate(v) for k, v in arrays.items()}
        np.savez(args.out_dir / f"{point}.npz", **d)
        print(f"{point}: {len(ms)} parts, {trials} trials, {acc:.0f} accepted, {len(d['electron'])} electrons "
              f"-> {args.out_dir / f'{point}.npz'}")


if __name__ == "__main__":
    main()
