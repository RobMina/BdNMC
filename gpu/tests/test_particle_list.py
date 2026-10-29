"""production_distribution particle_list: the parser against stock's own reading of a list (v0/plist_echo.sh), the
refusals, row cycling, a list of a sampler's parents reproducing that sampler's run, and parent positions."""
from pathlib import Path

import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import output, parents, particle_list, pipeline, rng
from bdnmc_jax.card import parse_card
from bdnmc_jax.constants import MASS_ETA, MASS_PION_NEUTRAL, speed_of_light

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
BASE = ROOT / "phase0" / "cards" / "mb_pi0_m005.dat"


@pytest.mark.parametrize("pos", ["pos", "nopos"])
def test_parse_matches_stock_echo(pos):
    # stock wrote 20 samples of the 7-row fixture (px py pz E x y z t, 6 significant digits)
    _, rows = particle_list.load(DATA / "plist_formats.dat", pos == "pos")
    assert rows.shape == (7, 7)
    par, origin = particle_list.block(rows, 0, 20, MASS_PION_NEUTRAL)
    got = np.column_stack([par[:, 1:], par[:, :1], origin])
    want = np.loadtxt(DATA / f"plist_formats_echo_{pos}.txt")
    np.testing.assert_allclose(got, want, rtol=6e-6, atol=0)


def test_rows_cycle_across_blocks():
    _, rows = particle_list.load(DATA / "plist_formats.dat", True)
    par, _ = particle_list.block(rows, 3, rng.BLOCK, MASS_PION_NEUTRAL)
    first = (3 * rng.BLOCK) % 7
    np.testing.assert_array_equal(np.asarray(par[:7, 1:]), np.roll(rows[:, :3], -first, axis=0))


@pytest.mark.parametrize("text, why", [
    ("0.1  0.2 3.0 1\n", "skip"),  # two spaces: an empty field
    ("0.1 0.2 x 1\n", "skip"),
    ("0.1 0.2 3.0\n", "fewer than 4"),
    ("0 0 0 1 1 2 3 0\n", "zero momentum"),
    ("# only a comment\n\n", "no particles"),
])
def test_refusals(text, why):
    with pytest.raises(ValueError, match=why):
        particle_list.parse(text, True)


def test_zero_momentum_without_positions_is_allowed():
    assert particle_list.parse("0 0 0 1 1 2 3 0\n", False).shape == (1, 7)


def _list_card(tmp_path, list_path, position, channel="pi0_decay"):
    out = []
    for line in BASE.read_text().splitlines():
        key = line.split()[:1]
        if key == ["production_distribution"]:
            out += ["production_distribution particle_list", f"particle_list_file {list_path}",
                    f"particle_list_position {'true' if position else 'false'}"]
        elif key == ["production_channel"]:
            out.append(f"production_channel {channel}")
        else:
            out.append(line)
    card = tmp_path / "list_card.dat"
    card.write_text("\n".join(out) + "\n")
    return card


def _write_list(path, mom, pos=None):
    cols = [mom[:, 0], mom[:, 1], mom[:, 2], np.ones(len(mom))]
    if pos is not None:
        cols += [pos[:, i] for i in range(4)]
    body = "\n".join(" ".join("%.17g" % v for v in r) for r in np.column_stack(cols))
    path.write_text("# px py pz E [x y z t]\n" + body + "\n")


@pytest.fixture(scope="module")
def sampler_parents():
    """The parents mb_pi0_m005's Sanford-Wang sampler draws in blocks 0 and 1 (seed = the card's)."""
    card = parse_card(BASE)
    (ch,) = pipeline.setup(card)[2]
    p = [np.asarray(parents.sample_block(ch.dist, rng.block_key(card.seed, 0, b), rng.BLOCK, ch.fmax, ch.m_meson)[0])
         for b in range(2)]
    return card.seed, np.concatenate(p)


def test_list_of_sampler_parents_reproduces_the_sampler_run(tmp_path, sampler_parents):
    seed, p4 = sampler_parents
    _write_list(tmp_path / "list.dat", p4[:, 1:])
    n = 2 * rng.BLOCK
    ref = pipeline.candidates(BASE, n, seed)
    res = pipeline.candidates(_list_card(tmp_path, tmp_path / "list.dat", False), n, seed)
    assert res["vnum"] == ref["vnum"] and res["n_hits"] == ref["n_hits"] > 0
    for k, v in ref["candidates"].items():
        np.testing.assert_array_equal(res["candidates"][k], v, err_msg=k)
    assert not res["candidates"]["origin"].any()


def test_positions_move_the_decay_vertex(tmp_path, sampler_parents):
    seed, p4 = sampler_parents
    rs = np.random.default_rng(1)
    z = rs.uniform(0, 50, len(p4))
    pos = np.column_stack([rs.normal(0, 0.2, len(p4)), rs.normal(0, 0.2, len(p4)), z, z / speed_of_light])
    _write_list(tmp_path / "list.dat", p4[:, 1:], pos)
    card = _list_card(tmp_path, tmp_path / "list.dat", True)
    res = pipeline.finish(pipeline.candidates(card, 2 * rng.BLOCK, seed))
    c = res["candidates"]
    assert res["n_hits"] > 0
    i = c["block"] * rng.BLOCK + c["trial"]
    np.testing.assert_array_equal(c["origin"], pos[i])
    np.testing.assert_array_equal(c["parent"][:, 1:], p4[i, 1:])
    # the interaction point is on the DM's line from the parent's position, inside the detector crossing
    d = c["vertex"][:, :3] - c["origin"][:, :3]
    mom = np.linalg.norm(c["dm"][:, 1:], axis=1)
    s = np.linalg.norm(d, axis=1)
    np.testing.assert_allclose(d, s[:, None] * c["dm"][:, 1:] / mom[:, None], rtol=0, atol=1e-9)
    assert ((s >= c["crossing"][:, 0] * mom * (1 - 1e-12)) & (s <= c["crossing"][:, 1] * mom * (1 + 1e-12))).all()
    speed = mom / c["dm"][:, 0]
    np.testing.assert_allclose(c["vertex"][:, 3], c["origin"][:, 3] + s / (speed * speed_of_light), rtol=1e-12)
    # text output: parent, V and DM start at the list position (Particle::report fields after the name:
    # E px py pz m, then origin x y z t, then end x y z t)
    path = tmp_path / "events.dat"
    output.write_text(res, path)
    rows, _ = output.accepted_events(res)
    o = c["origin"][rows[0]]
    lines = path.read_text().splitlines()
    for name, start, end in (("pion", 6, 10), ("pion", 10, 14), ("V", 6, 10), ("DM", 6, 10)):
        f = next(line for line in lines if line.startswith(name + " ")).split()
        np.testing.assert_allclose([float(v) for v in f[start:end]], o, rtol=1e-9)


def test_eta_list_uses_the_eta_mass(tmp_path, sampler_parents):
    _, p4 = sampler_parents
    _write_list(tmp_path / "list.dat", p4[:1000, 1:])
    (ch,) = pipeline.setup(parse_card(_list_card(tmp_path, tmp_path / "list.dat", False, "eta_decay")))[2]
    assert ch.kind == "list" and ch.m_meson == MASS_ETA and ch.dist.n == 1000
    par, _ = particle_list.block(ch.rows, 0, 4, ch.m_meson)
    np.testing.assert_allclose(np.asarray(par[:, 0]) ** 2 - (np.asarray(par[:, 1:]) ** 2).sum(1), MASS_ETA ** 2,
                               rtol=1e-9)


def test_envelope_and_missing_file_are_refused(tmp_path, sampler_parents):
    _, p4 = sampler_parents
    _write_list(tmp_path / "list.dat", p4[:10, 1:])
    card = parse_card(_list_card(tmp_path, tmp_path / "list.dat", False))
    with pytest.raises(NotImplementedError, match="parent_sampler"):
        pipeline.setup(card, "envelope")
    del card.channels[0].raw["particle_list_file"]
    with pytest.raises(ValueError, match="particle_list_file undefined"):
        pipeline.setup(card)
