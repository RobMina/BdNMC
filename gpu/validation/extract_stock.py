"""Extract the V2 variables from a stock comprehensive event file into a .npz.

Stock writes, per accepted DM, the lines `DM`, `Recoil_DM`, `electron` (probscatter inserts the electron and then
the recoil right after the DM, so the electron's DM is the last `DM` line before it); the parent is the event's
`pion` line. Particle line: name E px py pz m x0 y0 z0 t0 x1 y1 z1 t1 KE p*c0 p*c1 (Particle::report).
Usage: extract_stock.py <events.dat> <out.npz>
"""
import argparse
from pathlib import Path

import numpy as np


def extract(path):
    e, dm, par, vtx = [], [], [], []
    last_dm = last_pion = None
    with open(path) as f:
        for line in f:
            if line.startswith(("Run ", "event ", "endevent", "\n")) or not line.strip():
                if line.startswith("event "):
                    last_dm = last_pion = None
                continue
            tok = line.split()
            name = tok[0]
            if name == "pion":
                last_pion = [float(x) for x in tok[1:5]]
            elif name == "DM":
                last_dm = [float(x) for x in tok[1:5]]
            elif name == "electron":
                if last_dm is None or last_pion is None:
                    raise ValueError(f"electron without DM/pion: {line!r}")
                e.append([float(x) for x in tok[1:5]])
                vtx.append([float(x) for x in tok[6:10]])
                dm.append(last_dm)
                par.append(last_pion)
    return {"electron": np.array(e), "dm": np.array(dm), "parent": np.array(par), "vertex": np.array(vtx)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("events", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    d = extract(args.events)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **d)
    print(f"{args.out}: {len(d['electron'])} electrons")


if __name__ == "__main__":
    main()
