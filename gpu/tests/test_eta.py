"""eta decay (eta_decay with k0_sanfordwang or bpmt_k0): branching ratio and Vnum against stock logs, the K0 parent
laws, kinematics, the refused combinations, and text output."""
from pathlib import Path
import re

import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import branching, output, parents, pipeline, rng
from bdnmc_jax.card import parse_card
from bdnmc_jax.constants import MASS_ETA

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "phase0" / "cards"
BASES = ["mb_eta_m005", "he_eta_cuboid_m005"]


@pytest.mark.parametrize("stem", BASES)
def test_branching_ratio_and_vnum_match_stock_log(stem):
    log = ROOT / "phase0" / "out" / stem / "O2" / "log.txt"
    if not log.exists():
        pytest.skip("stock run not present")
    text = log.read_text()
    assert "eta_decay in On-Shell mode" in text
    br_stock = float(re.search(r"Branching Ratio 1 = (\S+)", text).group(1))
    vnum_stock = float(re.search(r"Production events in channel 1 = (\S+)", text).group(1))
    card = parse_card(CARDS / f"{stem}.dat")
    br, off = branching.eta_channel(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
    assert not off
    (ch,) = pipeline.setup(card)[2]
    assert ch.m_meson == MASS_ETA and ch.vnum == pytest.approx(br * card.pot * card.channels[0].meson_per_pi0
                                                                * card.pi0_per_pot, rel=1e-15)
    assert br == pytest.approx(br_stock, rel=5e-6)
    assert ch.vnum == pytest.approx(vnum_stock, rel=5e-6)


def test_k0_laws_differ_from_pi0_and_keep_stock_choices():
    sw, sw0 = parents.SanfordWang.k0(), parents.SanfordWang()
    assert sw.fmax == 0.6506 and sw.ppimax == sw0.ppimax and sw.kind == "k0"
    p, th = np.array([1.0, 3.0]), np.array([0.05, 0.2])
    ref = np.sin(th) * sw._sw(sw.e, p, th)
    np.testing.assert_allclose(np.asarray(sw.density(p, th)), np.asarray(ref), rtol=0)
    b, b0 = parents.BMPT(120.0, 12, choice="k0"), parents.BMPT(120.0, 12)
    assert b.meson_mass == 0.493677 and b.p_max == pytest.approx(b0.p_max - (0.493677 - 0.13957018))
    assert not np.allclose(np.asarray(b.density(p, th)), np.asarray(b0.density(p, th)))


@pytest.mark.parametrize("stem", BASES)
def test_eta_block_kinematics(stem):
    card_path = CARDS / f"{stem}.dat"
    (ch,) = pipeline.setup(parse_card(card_path))[2]
    par, info = parents.sample_block(ch.dist, rng.block_key(5, 0, 0), rng.BLOCK, ch.fmax, ch.m_meson)
    par = np.asarray(par)
    assert int(info["filled"]) == rng.BLOCK and int(info["over_bar"]) == 0
    res = pipeline.run(card_path, 8 * rng.BLOCK, seed=11, batch=4)
    c = res["candidates"]
    np.testing.assert_allclose(c["parent"][:, 0] ** 2 - (c["parent"][:, 1:] ** 2).sum(1), MASS_ETA ** 2, rtol=1e-7)
    np.testing.assert_allclose(c["mediator"][:, 0] ** 2 - (c["mediator"][:, 1:] ** 2).sum(1), ch.m_v ** 2,
                               rtol=1e-6)
    assert res["accepted"] > 0 and np.isfinite(c["electron"][c["prob"] > 0]).all()


@pytest.mark.parametrize("dist", ["bmpt", "pi0_sanfordwang", "proton_brem"])
def test_unvalidated_eta_distributions_are_refused(dist, tmp_path):
    text = (CARDS / "he_eta_cuboid_m005.dat").read_text().replace("production_distribution bpmt_k0",
                                                                 f"production_distribution {dist}")
    (tmp_path / "c.dat").write_text(text)
    with pytest.raises(NotImplementedError):
        pipeline.setup(parse_card(tmp_path / "c.dat"))


def test_eta_text_output_parent_line(tmp_path):
    res = pipeline.finish(pipeline.candidates(CARDS / "he_eta_cuboid_m005.dat", 2 * rng.BLOCK, seed=4))
    path, summ = tmp_path / "events.dat", tmp_path / "summary.dat"
    n = output.write_text(res, path, summary_path=summ, run="e")
    lines = path.read_text().splitlines()
    first = lines.index("event 1")
    assert lines[first + 1].split()[0] == "eta" and float(lines[first + 1].split()[5]) == MASS_ETA
    rows, ev = output.accepted_events(res)
    assert len(rows) == res["accepted"] and n == ev.max()  # events, not DM: an event may hold both DM
    assert summ.read_text().splitlines()[1].startswith("eta_decay ")
    import sys
    sys.path.insert(0, str(ROOT / "validation"))
    from extract_stock import extract
    back = extract(path)
    np.testing.assert_allclose(back["parent"], res["candidates"]["parent"][rows], rtol=1e-9)
