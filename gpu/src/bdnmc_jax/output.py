"""Event output: stock BdNMC's comprehensive text format and summary line, and HDF5.

Input is a pass-2 result (`pipeline.finish(candidates, weighted)`). Accepted DM are put in stock's event order:
trials in order, and within a trial the DM in slot order (stock's particle list holds the two DM in decay order and
inserts each accepted DM's Recoil_DM and electron after it).

Text (`write_text`) follows main.cpp and Particle::report:
- `Run <name>`;
- `event N` (weighted: `event N <weight>`);
- one line per particle: name E px py pz m x0 y0 z0 t0 x1 y1 z1 t1 KE p*crossing0 p*crossing1, at 10 significant
  digits;
- `endevent N` and a blank line;
- at the end, ` Summary <se> <m_V> <m_chi> <epsilon> <alpha_D> <signal> <POT> <efficiency> <events> `.

Particle fields as stock fills them:
- parent `pion` and `V`: origin and end zero;
- `DM`: origin zero, end = interaction point; the last two fields are the distances to the detector entry and exit;
- `Recoil_DM` and `electron`: origin = interaction point, end zero.

Two deliberate differences, both documented in docs/install-and-quickstart.md:
- `<events>` in the Summary line is the number of events written, not the card's samplesize (the port runs a
  fixed number of trials);
- weighted output writes one event per DM, because stock's single event weight (the last DM's) cannot represent an
  event with two DM of different weights.

HDF5 (`write_hdf5`) holds the same accepted rows as arrays, with a `weight` per row whose sum is the predicted
number of signal events in both modes, and the run's card and settings as attributes.
"""
from pathlib import Path
import time

import numpy as np

from . import card as cardmod, rng
from .constants import MASS_ELECTRON, MASS_PION_NEUTRAL

PARENT = {"pi0_decay": ("pion", MASS_PION_NEUTRAL)}  # parent name and mass as stock writes them, per channel
FORMAT = "bdnmc-jax-events/1"


def accepted_events(res):
    """(row indices into res["candidates"] in output order, event number per row, starting at 1)."""
    c = res["candidates"]
    idx = np.nonzero(res["accepted_mask"])[0]
    order = idx[np.lexsort((c["slot"][idx], c["trial"][idx], c["block"][idx]))]
    if res.get("weighted"):
        return order, np.arange(1, len(order) + 1)
    trial_id = c["block"][order].astype(np.int64) * rng.BLOCK + c["trial"][order]
    return order, np.cumsum(np.r_[True, trial_id[1:] != trial_id[:-1]]) if len(order) else np.zeros(0, int)


def row_weights(res, rows):
    """Weight per accepted DM; the weights sum to the run's predicted signal events (res["se"])."""
    if res.get("weighted"):
        return res["candidates"]["prob"][rows] / res["trials"] * res["vnum"] * res["efficiency"]
    return np.full(len(rows), res["se"] / max(len(rows), 1))


def _g10(x):
    return "%.10g" % x


def _particle(name, p4, m, origin, end, cross=(0.0, 0.0)):
    mom = float(np.sqrt(p4[1] ** 2 + p4[2] ** 2 + p4[3] ** 2))
    vals = (p4[0], p4[1], p4[2], p4[3], m, *origin, *end, p4[0] - m, mom * cross[0], mom * cross[1])
    return name + " " + " ".join(_g10(v) for v in vals) + "\n"


def _model_report(card, fmt):
    return " ".join(fmt % v for v in (card.mass_v, card.mass_dm, card.epsilon, card.alpha_d))


def write_text(res, path, summary_path=None, run=None):
    """Write the accepted events in stock's comprehensive format to `path`; append stock's summary lines to
    `summary_path` if given. Returns the number of events written."""
    card = cardmod.parse_card(res["card"])
    run = run or card.keymap.get("run") or str(int(time.time()))
    chan = card.channels[0]
    parent_name, parent_mass = PARENT[chan.production_channel]
    c = res["candidates"]
    rows, event = accepted_events(res)
    zero4 = (0.0, 0.0, 0.0, 0.0)
    n_events = int(event[-1]) if len(event) else 0
    with open(path, "w") as f:
        f.write(f"Run {run}\n")
        i = 0
        while i < len(rows):
            j = i
            while j < len(rows) and event[j] == event[i]:
                j += 1
            r0 = rows[i]
            # weighted: stock prints the event's scatter probability (pMax, reset per DM) after the event number
            head = f"event {event[i]}" + (f" {_g10(c['prob'][r0])}" if res.get("weighted") else "")
            lines = [head + "\n", _particle(parent_name, c["parent"][r0], parent_mass, zero4, zero4),
                     _particle("V", c["mediator"][r0], card.mass_v, zero4, zero4)]
            for r in rows[i:j]:
                v = tuple(c["vertex"][r])
                lines.append(_particle("DM", c["dm"][r], card.mass_dm, zero4, v, tuple(c["crossing"][r])))
                lines.append(_particle("Recoil_DM", c["recoil_dm"][r], card.mass_dm, v, zero4))
                lines.append(_particle("electron", c["electron"][r], MASS_ELECTRON, v, zero4))
            lines.append(f"endevent {event[i]}\n\n")
            f.write("".join(lines))
            i = j
        f.write(f"\n Summary {_g10(res['se'])} {_model_report(card, '%.10g')} {card.signal_channel} "
                f"{_g10(card.pot)} {_g10(card.efficiency)} {n_events} \n")
    if summary_path is not None:
        rep = _model_report(card, "%g")
        with open(summary_path, "a") as f:
            f.write(f"Run {run}\n")
            f.write(f"{chan.production_channel} {rep} {res['se']:g} {card.signal_channel} {card.pot:g} "
                    f"{card.efficiency:g} {n_events} {res['vnum']:g} {res['vnum']:g}\n")
            f.write(f"Total {rep} {res['se']:g} {card.signal_channel} {card.pot:g} {card.efficiency:g} {n_events} \n")
    return n_events


def write_hdf5(res, path, extra_attrs=None):
    """Write the accepted DM rows to an HDF5 file (group `accepted`, one row per DM) with the run's settings as
    attributes. Needs h5py (`pip install h5py`). Returns the number of rows written."""
    import h5py
    c = res["candidates"]
    rows, event = accepted_events(res)
    card_text = Path(res["card"]).read_text()
    with h5py.File(path, "w") as f:
        attrs = {"format": FORMAT, "card": card_text, "card_path": str(res["card"]), "seed": res["seed"],
                 "n_trials": res["n_trials"], "trials": res["trials"], "vnum": res["vnum"],
                 "efficiency": res["efficiency"], "se": res["se"], "se_weighted": res["se_weighted"],
                 "pmax": res["pmax"], "weighted": bool(res.get("weighted", False)),
                 "parent_sampler": res.get("parent_sampler", "stock"), "n_hits": res["n_hits"]}
        for k, v in (attrs | (extra_attrs or {})).items():
            f.attrs[k] = v
        g = f.create_group("accepted")
        g["event"] = event
        g["weight"] = row_weights(res, rows)
        for k in ("block", "trial", "slot", "prob", "L", "dm", "electron", "recoil_dm", "vertex", "parent", "mediator",
                  "crossing"):
            if k in c:
                g[k] = np.asarray(c[k])[rows]
    return len(rows)
