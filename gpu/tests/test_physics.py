"""Self-consistency of the ported physics (V0 against the C++ itself needs the v0 harness; see v0/README.md)."""
import math
from pathlib import Path
import re

import numpy as np
import pytest
from scipy import integrate

import bdnmc_jax  # noqa: F401
from bdnmc_jax import branching, electron
from bdnmc_jax.card import parse_card
from bdnmc_jax.constants import MASS_ELECTRON as Me

ROOT = Path(__file__).resolve().parents[1]
SC = electron.ElectronScatter(0.005, 0.1, 1e-3, 0.1, 0.035, 0.9)


@pytest.mark.parametrize("E", [0.05, 0.3, 1.0, 5.0])
def test_sigma2_is_integral_of_dsigma(E):
    lo, hi = float(electron.scat_min(SC, E)), float(electron.scat_max(SC, E))
    f = lambda x: float(electron.dsigma_dEe(x, E, SC.m_dm, SC.m_v, SC.kappa, SC.alpha_d))
    ref, _ = integrate.quad(f, lo, hi, epsrel=1e-12, limit=200)
    got = float(electron.sigma2(E, SC.m_dm, SC.m_v, SC.kappa, SC.alpha_d, hi, lo))
    assert got == pytest.approx(ref, rel=1e-8)


def test_recoil_energy_and_angle_are_inverse():
    E, m = 2.0, 0.005
    for c in [1.0, 0.9, 0.5, 0.1]:
        Ee = float(electron.ee_theta(E, c, m))
        assert math.cos(float(electron.theta_ee(Ee, E, m))) == pytest.approx(c, rel=1e-10)


def test_sigma2_zero_when_window_empty():
    assert float(electron.sigma2(0.04, 0.005, 0.1, 1e-3, 0.1, 0.03, 0.035)) == 0.0


@pytest.mark.parametrize("stem", ["mb_pi0_m001", "mb_pi0_m005", "mb_pi0_m020", "mb_pi0_m045",
                                  "he_pi0_cuboid_m001", "he_pi0_cuboid_m045"])
def test_branching_ratio_matches_stock_log(stem):
    """Stock prints the branching ratio and Vnum to 6 significant figures; the port must agree to that."""
    log = ROOT / "phase0" / "out" / "ref" / stem / "O2" / "log.txt"
    if not log.exists():
        pytest.skip("stock reference run not present")
    text = log.read_text()
    br_stock = float(re.search(r"Branching Ratio 1 = (\S+)", text).group(1))
    vnum_stock = float(re.search(r"Production events in channel 1 = (\S+)", text).group(1))
    card = parse_card(ROOT / "phase0" / "cards" / "ref" / f"{stem}.dat")
    br, off = branching.pi0_channel(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
    assert not off
    vnum = br * card.pot * card.channels[0].meson_per_pi0 * card.pi0_per_pot
    assert br == pytest.approx(br_stock, rel=5e-6)
    assert vnum == pytest.approx(vnum_stock, rel=5e-6)


def test_double_exponential_stock_behaviour():
    """Stock's rule counts the k = 0 node twice (func_check adds f(-x0) + f(x0) with x0 = 0), so it overestimates by
    w0 f(mid) / scale, an error that only halves with each halving of h. At stock's precision (1e-4) it stops a few
    1e-4 high; tighter precision does not converge within DDMAX."""
    exact = math.e - 1
    got = branching.double_exponential_adapt(math.exp, 0.0, 1.0, 100, 0.1, 1e-4)
    assert exact < got < exact * (1 + 5e-4)
    with pytest.raises(RuntimeError):
        branching.double_exponential_adapt(math.exp, 0.0, 1.0, 100, 0.1, 1e-10)
