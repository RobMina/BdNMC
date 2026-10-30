"""Event output: stock's text format (structure, number formatting, readable by the stock-file parser) and HDF5."""
from pathlib import Path
import sys

import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import output, pipeline, rng

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "phase0" / "cards" / "ref" / "he_pi0_cuboid_m005.dat"
sys.path.insert(0, str(ROOT / "validation"))
from extract_stock import extract  # noqa: E402

# Particle lines from a stock -O2 event file (mb_pi0, 5 MeV), as printed by Particle::report.
STOCK_LINES = [
    "pion 1.072545783 0.07070567166 -0.06282935842 1.059806092 0.1349766 0 0 0 0 0 0 0 0 0.9375691829 0 0",
    "DM 0.517538835 -0.004143410176 -0.0003786638654 0.5174979561 0.005 0 0 0 0 -3.910130342 -0.357344556 "
    "488.3620917 1.629129273e-06 0.512538835 488.3073969 493.6636462",
    "electron 0.1845521507 -0.01030207635 -0.006465669085 0.1841502038 0.0005109989461 -3.910130342 -0.357344556 "
    "488.3620917 1.629129273e-06 0 0 0 0 0.1840411517 0 0",
]


@pytest.fixture(scope="module")
def run():
    return pipeline.candidates(CARD, rng.BLOCK, seed=5)


def test_number_format_matches_stock():
    for line in STOCK_LINES:
        for tok in line.split()[1:]:
            assert output._g10(float(tok)) == tok


def test_text_output_structure_and_round_trip(run, tmp_path):
    res = pipeline.finish(run)
    path, summ = tmp_path / "events.dat", tmp_path / "summary.dat"
    n = output.write_text(res, path, summary_path=summ, run="test")
    lines = path.read_text().splitlines()
    assert lines[0] == "Run test" and lines[-1].startswith(" Summary ")
    assert lines[-1].split()[-1] == str(n)
    sizes, two_dm = [], 0
    for i, l in enumerate(lines):
        if l.startswith("event "):
            k = lines.index(f"endevent {l.split()[1]}", i) - i - 1
            sizes.append(k)
            two_dm += k == 8
        elif l and not l.startswith(("Run", "endevent", " Summary")):
            assert len(l.split()) == 17
    assert len(sizes) == n and set(sizes) <= {5, 8} and two_dm > 0
    assert sum((k - 2) // 3 for k in sizes) == res["accepted"]
    # The stock-file parser reads it back: same electrons, DM, parents and vertices as the accepted rows.
    back = extract(path)
    rows, _ = output.accepted_events(res)
    c = res["candidates"]
    for key, ref in (("electron", c["electron"]), ("dm", c["dm"]), ("parent", c["parent"]), ("vertex", c["vertex"])):
        np.testing.assert_allclose(back[key], ref[rows], rtol=1e-9, atol=1e-12, err_msg=key)
    summary = summ.read_text().splitlines()
    assert summary[0] == "Run test" and summary[2].startswith("Total ")
    assert float(summary[2].split()[5]) == pytest.approx(res["se"], rel=1e-5)


def test_dm_vertex_lies_between_detector_entry_and_exit(run, tmp_path):
    res = pipeline.finish(run)
    path = tmp_path / "events.dat"
    output.write_text(res, path, run="t")
    for l in path.read_text().splitlines():
        if l.startswith("DM "):
            v = [float(x) for x in l.split()[1:]]
            dist = np.linalg.norm(v[9:12])
            assert v[14] - 1e-6 <= dist <= v[15] + 1e-6
            assert v[13] == pytest.approx(v[0] - v[4], rel=1e-9)


def test_weighted_text_has_one_dm_per_event(run, tmp_path):
    res = pipeline.finish(run, weighted=True)
    path = tmp_path / "events_w.dat"
    n = output.write_text(res, path, run="w")
    assert n == int(res["candidates"]["ang_ok"].sum())
    heads = [l.split() for l in path.read_text().splitlines() if l.startswith("event ")]
    assert all(len(h) == 3 for h in heads)
    rows, _ = output.accepted_events(res)
    np.testing.assert_allclose([float(h[2]) for h in heads], res["candidates"]["prob"][rows], rtol=1e-9)


@pytest.mark.parametrize("weighted", [False, True])
def test_hdf5_weights_sum_to_the_prediction(run, tmp_path, weighted):
    h5py = pytest.importorskip("h5py")
    res = pipeline.finish(run, weighted=weighted)
    path = tmp_path / "events.h5"
    n = output.write_hdf5(res, path)
    with h5py.File(path) as f:
        g = f["accepted"]
        assert len(g["weight"]) == n == len(output.accepted_events(res)[0])
        assert g["weight"][:].sum() == pytest.approx(res["se"], rel=1e-12)
        assert f.attrs["format"] == output.FORMAT and f.attrs["weighted"] == weighted
        assert g["electron"].shape == (n, 4) and g["crossing"].shape == (n, 2)


def test_event_order_is_block_trial_slot(run):
    # two blocks with DM in both slots: rows come in block order, slot 0 then slot 1 within a block
    res = pipeline.finish(pipeline.candidates(CARD, 2 * rng.BLOCK, seed=5))
    c = res["candidates"]
    assert (c["slot"] >= rng.BLOCK).any()
    rows, _ = output.accepted_events(res)
    idx = np.nonzero(res["accepted_mask"])[0]
    assert (rows == idx[np.lexsort((c["slot"][idx], c["trial"][idx], c["block"][idx]))]).all()


@pytest.mark.parametrize("card", [CARD, ROOT / "phase0" / "cards" / "ref" / "mb_brem_m020.dat"])
@pytest.mark.parametrize("weighted", [False, True])
def test_device_event_path_matches_the_host_path(card, weighted):
    # several batches (batch 1), and the forced fallback through candidates()
    kw = {"seed": 7, "batch": 1, "cap": 1024}
    host = pipeline.finish(pipeline.candidates(card, 3 * rng.BLOCK, **kw), weighted)
    rows, event = output.accepted_events(host)
    for ev in (pipeline.events(card, 3 * rng.BLOCK, weighted=weighted, **kw),
               pipeline.events(card, 3 * rng.BLOCK, weighted=weighted, max_device_bytes=1, **kw)):
        assert ev["events_path"] in ("device", "host")
        for k in ("se", "se_weighted", "pmax", "accepted", "n_hits", "prob_sum"):
            assert ev[k] == host[k], k
        r2, e2 = output.accepted_events(ev)
        assert (e2 == event).all()
        c = ev["candidates"]
        for k in host["candidates"]:
            assert np.array_equal(np.asarray(c[k])[r2], host["candidates"][k][rows]), k
