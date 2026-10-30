"""Off-shell meson decays (offshell.py): branching ratio and mode against the stock log, the (s, theta) sampler's law
and bar, DecayDM_Off_Shell kinematics (including stock's x-z-plane V* direction), the pipeline and text output."""
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import bdnmc_jax  # noqa: F401
from bdnmc_jax import branching, offshell, output, pipeline, rng
from bdnmc_jax.card import parse_card
from bdnmc_jax.constants import MASS_ETA, MASS_PION_NEUTRAL, pi
from bdnmc_jax.kinematics import lorentz

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / "phase0" / "cards"
KAPPA, ALPHA = 1e-3, 0.1
# (meson, m_V, m_chi): V heavier than the meson; V lighter than 2 m_chi; eta with V above m_eta
REGIMES = [("pi0", 0.2, 0.005), ("pi0", 0.05, 0.03), ("eta", 0.6, 0.02)]


@pytest.mark.parametrize("stem", ["mb_offshell_m005", "he_offshell_cuboid_m005"])
def test_mode_and_vnum_match_stock_log(stem):
    log = ROOT / "phase0" / "out" / stem / "O2" / "log.txt"
    if not log.exists():
        pytest.skip("stock run not present")
    text = log.read_text()
    assert "pi0_decay in Off-Shell mode" in text
    card = parse_card(CARDS / f"{stem}.dat")
    (ch,) = pipeline.setup(card)[2]
    assert isinstance(ch.offshell, offshell.OffShell) and ch.offshell.meson == "pi0"
    vnum_stock = float(re.search(r"Production events in channel 1 = (\S+)", text).group(1))
    assert ch.vnum == pytest.approx(vnum_stock, rel=5e-6)


@pytest.mark.parametrize("meson, mv, mx", REGIMES)
def test_regimes_are_off_shell(meson, mv, mx):
    chan = branching.pi0_channel if meson == "pi0" else branching.eta_channel
    m = MASS_PION_NEUTRAL if meson == "pi0" else MASS_ETA
    br, off = chan(mv, mx, KAPPA, ALPHA)
    assert br > 0 and off and not branching.uses_on_shell_decay(off, mv, mx, m)


def _draw(osh, mv, mx, n=1 << 17, seed=3):
    mvg = offshell.mv_gamma(mv, mx, KAPPA, ALPHA)
    fmax = osh.find_fmax(mv, mx, mvg)
    x, filled, _, nover = rng.rejection(jax.random.key(seed), osh.propose, fmax, n, n,
                                        params=(osh.m_meson, mv, mx, mvg))
    assert int(filled) == n and int(nover) == 0
    return np.asarray(x), mvg


@pytest.mark.parametrize("meson, mv, mx", REGIMES)
def test_sampler_follows_the_density(meson, mv, mx):
    osh = offshell.OffShell(meson)
    x, mvg = _draw(osh, mv, mx)
    m = osh.m_meson
    grid = np.linspace(4 * mx * mx, m * m, 20001)
    pdf = np.asarray(osh.shape(jnp.asarray(grid), pi / 2, m, mv, mx, mvg))
    cdf = np.concatenate([[0], np.cumsum((pdf[1:] + pdf[:-1]) / 2)])
    cdf /= cdf[-1]
    assert stats.kstest(x[:, 0], lambda s: np.interp(s, grid, cdf)).pvalue > 1e-3
    # theta marginal: sin^3, CDF (2 - 3 cos + cos^3) / 4
    assert stats.kstest(x[:, 1], lambda t: (2 - 3 * np.cos(t) + np.cos(t) ** 3) / 4).pvalue > 1e-3


def test_bar_covers_a_resonance_inside_the_window():
    # off-shell sampling with the V* pole inside [4 m_chi^2, m_pi0^2] (stock's pmax starts at the pole value there)
    _draw(offshell.OffShell("pi0"), 0.1, 0.01)


def _at_rest(m, n):
    return jnp.stack([jnp.full(n, m), jnp.zeros(n), jnp.zeros(n), jnp.zeros(n)], axis=-1)


def test_decay_kinematics_in_the_meson_frame():
    n, m, mx = 4096, MASS_PION_NEUTRAL, 0.005
    rs = np.random.default_rng(0)
    s = jnp.asarray(rs.uniform(4 * mx * mx, m * m, n))
    th = jnp.asarray(rs.uniform(0, pi, n))
    u = [jnp.asarray(rs.random(n)) for _ in range(3)]
    med, d1, d2, mm = map(np.asarray, offshell.decay(_at_rest(m, n), m, mx, s, th, *u))
    np.testing.assert_allclose(mm, np.sqrt(np.asarray(s)), rtol=1e-15)
    np.testing.assert_allclose(d1 + d2, med, rtol=0, atol=1e-13)  # chi chi = V*
    pv = np.linalg.norm(med[:, 1:], axis=1)
    np.testing.assert_allclose(med[:, 0] + pv, m, rtol=1e-12)  # V* + photon = meson at rest (TriangleFunc
    # cancels near s = m^2)
    np.testing.assert_allclose(med[:, 0] ** 2 - pv ** 2, np.asarray(s), rtol=1e-9)
    assert np.abs(med[:, 2]).max() < 1e-15  # stock: V* direction in the x-z plane (Rotate_z of a z vector)
    # the DM angle to the V* axis in the V* frame is theta (or pi - theta for the second DM)
    b = med[:, 1:] / med[:, :1]
    beta = np.linalg.norm(b, axis=1)
    d1s = np.asarray(lorentz(jnp.asarray(d1), jnp.asarray(beta), *(jnp.asarray(-b[:, i]) for i in range(3))))
    cos = (d1s[:, 1:] * med[:, 1:]).sum(1) / np.linalg.norm(d1s[:, 1:], axis=1) / pv
    np.testing.assert_allclose(cos, np.cos(np.asarray(th)), atol=1e-9)


def test_decay_conserves_momentum_in_the_lab():
    n, m, mx = 1024, MASS_ETA, 0.02
    rs = np.random.default_rng(1)
    p = rs.normal(0, 1, (n, 3)) * [0.3, 0.3, 5.0]
    par = jnp.asarray(np.column_stack([np.sqrt(m * m + (p ** 2).sum(1)), p]))
    s = jnp.asarray(rs.uniform(4 * mx * mx, m * m, n))
    med, d1, d2, _ = map(np.asarray, offshell.decay(par, m, mx, s, jnp.asarray(rs.uniform(0, pi, n)),
                                                     *(jnp.asarray(rs.random(n)) for _ in range(3))))
    np.testing.assert_allclose(d1 + d2, med, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(med[:, 0] ** 2 - (med[:, 1:] ** 2).sum(1), np.asarray(s), rtol=1e-7)


@pytest.fixture(scope="module")
def mb_off():
    return pipeline.finish(pipeline.candidates(CARDS / "mb_offshell_m005.dat", 2 * rng.BLOCK, seed=5))


def test_pipeline_runs_off_shell(mb_off, tmp_path):
    c = mb_off["candidates"]
    assert mb_off["n_hits"] > 0 and mb_off["accepted"] > 0
    assert ((c["med_mass"] > 2 * 0.005) & (c["med_mass"] < MASS_PION_NEUTRAL)).all()
    med = c["mediator"]
    np.testing.assert_allclose(med[:, 0] ** 2 - (med[:, 1:] ** 2).sum(1), c["med_mass"] ** 2, rtol=1e-6)
    path = tmp_path / "events.dat"
    output.write_text(mb_off, path)
    rows, _ = output.accepted_events(mb_off)
    v = next(line for line in path.read_text().splitlines() if line.startswith("V ")).split()
    assert float(v[5]) == pytest.approx(c["med_mass"][rows[0]], rel=1e-9)


def test_on_shell_cards_are_unchanged():
    (ch,) = pipeline.setup(parse_card(CARDS / "mb_pi0_m005.dat"))[2]
    assert ch.offshell is None
