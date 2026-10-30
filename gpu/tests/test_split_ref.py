"""Split stock references (phase0/make_split_ref_cards.py): the part cards against their reference cards, unique
seeds, and compare_v's combination of the parts' unweighted logs."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "phase0" / "cards"
sys.path.insert(0, str(ROOT / "validation"))
import compare_v  # noqa: E402
import merge_stock_parts  # noqa: E402


def keys(path):
    return {t[0]: t[1:] for t in (line.split() for line in path.read_text().splitlines()) if t and t[0] != "#"}


def test_parts_match_reference_cards():
    parts = defaultdict(list)
    for m in json.loads((CARDS / "ref_s" / "MANIFEST.json").read_text()):
        parts[m["point"]].append(m)
    assert parts
    for point, ms in parts.items():
        ref = keys(CARDS / "ref" / f"{point}.dat")
        assert sorted(m["part"] for m in ms) == list(range(ms[0]["n_parts"]))
        assert sum(m["samplesize"] for m in ms) == int(ref["samplesize"][0])
        assert sum(m["max_trials"] for m in ms) == int(ref["max_trials"][0])
        for m in ms:
            card = keys(CARDS / f"{m['stem']}.dat")
            assert card.keys() == ref.keys()
            changed = {k for k in ref if card[k] != ref[k]}
            assert changed == {"seed", "samplesize", "max_trials"}


def test_part_seeds_unique_over_all_cards():
    # (four ref_wl cards share seeds with four ref_w mb_list replicas; they never enter the same comparison)
    seeds = defaultdict(list)
    for p in CARDS.rglob("*.dat"):
        for line in p.read_text().splitlines():
            if line.split()[:1] == ["seed"]:
                seeds[line.split()[1]].append(p)
    for s, ps in seeds.items():
        if any(p.parent.name == "ref_s" for p in ps):
            assert len(ps) == 1, (s, ps)


def write_log(path, trials, events, se):
    path.mkdir(parents=True)
    (path / "log.txt").write_text(f"pi0_decay: {se} Events: {events} V_num: 1e10\nNumber of trials = {trials}\n"
                                  f"Predicted number of signal events = {se}\n")
    return path / "log.txt"


def test_unweighted_one_run_and_parts(tmp_path):
    one = write_log(tmp_path / "a", 1000, 400, 2.0)
    assert compare_v.stock_unweighted([one]) == pytest.approx((2.0, 2.0 / math.sqrt(400)))
    two = write_log(tmp_path / "b", 3000, 1300, 2.4)
    se, sig = compare_v.stock_unweighted([one, two])
    assert se == pytest.approx((1000 * 2.0 + 3000 * 2.4) / 4000)
    assert sig == pytest.approx(se / math.sqrt(1700))


def test_run_logs_counts_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_v, "ROOT", tmp_path)
    for k in range(3):
        write_log(tmp_path / f"out/x_p{k:02d}/O2_uva", 10, 1, 1.0)
    assert len(compare_v.run_logs("x", "out/{stem}_p*/O2_uva", 3)) == 3
    with pytest.raises(ValueError):
        compare_v.run_logs("x", "out/{stem}_p*/O2_uva", 4)


def test_merge_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(merge_stock_parts, "ROOT", tmp_path)
    (tmp_path / "phase0/cards/ref_s").mkdir(parents=True)
    ms = [{"stem": f"ref_s/x_m005_p{k:02d}", "point": "x_m005", "part": k, "n_parts": 2} for k in range(2)]
    (tmp_path / "phase0/cards/ref_s/MANIFEST.json").write_text(json.dumps(ms))
    line = lambda name, v: f"{name} " + " ".join(str(v + i) for i in range(16)) + "\n"  # noqa: E731
    for k, n in enumerate([2, 3]):
        run = tmp_path / "phase0/out" / ms[k]["stem"] / "O2_uva"
        write_log(run, 100 * n, n, 1.0)
        (run / "events.dat").write_text("Run x\n" + "".join(
            f"event {i}\n" + line("pion", 10 * k + i) + line("DM", 20) + line("electron", 30) + f"endevent {i}\n\n"
            for i in range(n)))
    monkeypatch.setattr(sys, "argv", ["merge_stock_parts.py", "--out-dir", str(tmp_path / "npz")])
    merge_stock_parts.main()
    d = np.load(tmp_path / "npz" / "x_m005.npz")
    assert d["part"].tolist() == [0, 0, 1, 1, 1]
    assert d["parent"][:, 0].tolist() == [0, 1, 10, 11, 12]
    assert d["electron"].shape == (5, 4) and d["vertex"].shape == (5, 4)
