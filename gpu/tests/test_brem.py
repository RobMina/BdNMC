"""Proton bremsstrahlung (V_decay + proton_brem): rate against stock logs, sampler bar, kinematics, stock's
one-DM-per-event rule, no recompile per mass, and text output."""
from pathlib import Path
import re

import jax
import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import brem, decay, output, pipeline, rng
from bdnmc_jax.card import parse_card
from bdnmc_jax.constants import MASS_PROTON as mp

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "phase0" / "cards"
BREM_POINTS = {"mb_brem_m001": "ref", "mb_brem_m005": None, "mb_brem_m020": "ref", "mb_brem_m045": "ref",
               "he_brem_cuboid_m001": "ref", "he_brem_cuboid_m005": None, "he_brem_cuboid_m020": "ref",
               "he_brem_cuboid_m045": "ref"}  # the 5 MeV references are the base cards' runs


def _card_and_log(stem):
    sub = BREM_POINTS[stem]
    card = CARDS / sub / f"{stem}.dat" if sub else CARDS / f"{stem}.dat"
    log = ROOT / "phase0" / "out" / (f"{sub}/{stem}" if sub else stem) / "O2" / "log.txt"
    return card, log


@pytest.mark.parametrize("stem", list(BREM_POINTS))
def test_vnum_matches_stock_log(stem):
    """Stock prints Vnum = V_prod_rate x POT to 6 significant figures."""
    card_path, log = _card_and_log(stem)
    if not log.exists():
        pytest.skip("stock reference run not present")
    vnum_stock = float(re.search(r"Production events in channel 1 = (\S+)", log.read_text()).group(1))
    (ch,) = pipeline.setup(parse_card(card_path))[2]
    assert ch.kind == "brem" and ch.name == "Dark_Bremsstrahlung"
    assert ch.vnum == pytest.approx(vnum_stock, rel=5e-6)


def test_simpson_cubature_is_exact_for_cubics():
    f = lambda x, y: x ** 3 * y + 2 * y ** 3 - x * y  # noqa: E731
    exact = (2 ** 4 - 1) / 4 * (3 ** 2 - 1) / 2 + 2 * (2 - 1) * (3 ** 4 - 1) / 4 - (2 ** 2 - 1) / 2 * (3 ** 2 - 1) / 2
    assert brem.simpson_cubature(f, 1, 2, 7, 1, 3, 5) == pytest.approx(exact, rel=1e-13)


def test_fixed_bar_covers_the_density():
    for stem in ("mb_brem_m005", "he_brem_cuboid_m005"):
        (ch,) = pipeline.setup(parse_card(_card_and_log(stem)[0]))[2]
        d = ch.dist
        u = np.random.default_rng(0).random((200_000, 2))
        z, pt = d.zmin + (d.zmax - d.zmin) * u[:, 0], d.ptmin + (d.ptmax - d.ptmin) * u[:, 1]
        assert np.asarray(d.shape(z, pt, ch.m_v)).max() < ch.fmax


@pytest.mark.parametrize("stem", ["mb_brem_m005", "he_brem_cuboid_m005"])
def test_stock_starting_bar_is_the_window_maximum(stem):
    """Stock starts its running bar at d2N(zmin, ptmin). If that is the maximum of the window, its running bar
    never rises and stock's brem parents are not clipped (unlike BMPT's)."""
    (ch,) = pipeline.setup(parse_card(_card_and_log(stem)[0]))[2]
    d = ch.dist
    corner = float(d.shape(d.zmin, d.ptmin, ch.m_v))
    assert corner == pytest.approx(ch.fmax / d.margin, rel=1e-12)


def test_brem_block_kinematics_and_one_dm_per_event():
    card_path = _card_and_log("he_brem_cuboid_m005")[0]
    card = parse_card(card_path)
    (ch,) = pipeline.setup(card)[2]
    v, info = brem.sample_block(ch.dist, rng.block_key(3, 0, 0), rng.BLOCK, ch.fmax, ch.m_v)
    v = np.asarray(v)
    assert int(info["filled"]) == rng.BLOCK and int(info["over_bar"]) == 0
    np.testing.assert_allclose(v[:, 0] ** 2 - (v[:, 1:] ** 2).sum(1), ch.m_v ** 2, rtol=1e-8)
    z = v[:, 3] / np.sqrt(card.beam_energy ** 2 - mp * mp)
    pt = np.hypot(v[:, 1], v[:, 2])
    d = ch.dist
    assert (z >= d.zmin - 1e-12).all() and (z <= d.zmax + 1e-12).all() and (pt <= d.ptmax + 1e-12).all()
    u = np.random.default_rng(1).random((rng.BLOCK, 2))
    d1, d2 = (np.asarray(x) for x in decay.two_body(v, ch.m_v, ch.m_dm, ch.m_dm, u[:, 0], u[:, 1]))
    np.testing.assert_allclose(d1 + d2, v, rtol=1e-9, atol=1e-12)
    res = pipeline.run(card_path, 4 * rng.BLOCK, seed=9, batch=4)
    c = res["candidates"]
    trial = c["block"].astype(np.int64) * rng.BLOCK + c["trial"]
    assert len(np.unique(trial)) == len(trial)  # stock's short-circuit: at most one DM per trial
    assert (c["slot"] >= rng.BLOCK).any()  # second DM used when the first misses
    assert res["accepted"] > 0 and np.isfinite(c["electron"][c["prob"] > 0]).all()


def test_brem_masses_share_one_compile():
    pipeline.blocks_batch.clear_cache()
    for stem in ("mb_brem_m001", "mb_brem_m020", "mb_brem_m045"):
        pipeline.candidates(_card_and_log(stem)[0], 1000, seed=2)
    assert pipeline.blocks_batch._cache_size() == 1


def test_brem_text_output_has_no_meson_line(tmp_path):
    res = pipeline.finish(pipeline.candidates(_card_and_log("mb_brem_m005")[0], rng.BLOCK, seed=4))
    path, summ = tmp_path / "events.dat", tmp_path / "summary.dat"
    n = output.write_text(res, path, summary_path=summ, run="b")
    lines = path.read_text().splitlines()
    first = lines.index("event 1")
    assert [l.split()[0] for l in lines[first + 1:first + 5]] == ["V", "DM", "Recoil_DM", "electron"]
    assert lines[first + 5] == "endevent 1" and n == res["accepted"]
    assert summ.read_text().splitlines()[1].startswith("Dark_Bremsstrahlung ")
    import sys
    sys.path.insert(0, str(ROOT / "validation"))
    from extract_stock import extract
    back = extract(path)
    rows, _ = output.accepted_events(res)
    np.testing.assert_allclose(back["parent"], res["candidates"]["mediator"][rows], rtol=1e-9)
