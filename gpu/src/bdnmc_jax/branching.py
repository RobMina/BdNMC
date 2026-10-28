"""Meson -> gamma V -> gamma chi chi branching ratios and the off-shell decision (stock src/branchingratios.cpp,
src/decay_gen_kinetic_mixing.cpp, src/Integrator.cpp). Setup-time scalars, so plain Python floats.

`brpi0_to_gamma_dm_dm` integrates with stock's adaptive double-exponential rule at stock's settings (N=100,
h=0.1, precision 1e-4), ported step for step: its result is the `Vnum` normalisation, so the port must reproduce
stock's number, not the exact integral.
"""
import math

from .constants import (MASS_ELECTRON, MASS_ETA, MASS_MUON, MASS_PION_NEUTRAL as mpi0, alphaEM, bretato2gamma,
                        brpi0to2gamma, pi)

meta = MASS_ETA
DDMAX = 20


def brpi0toVgamma(mv, mx, kappa, alphaD):
    if mv > mpi0:
        return 0.0
    return 2 * kappa ** 2 * (1 - mv * mv / mpi0 ** 2) ** 3 * brpi0to2gamma


def bretatoVgamma(mv, mx, kappa, alphaD):
    if mv > meta:
        return 0.0
    return 2 * kappa ** 2 * (1 - mv * mv / meta ** 2) ** 3 * bretato2gamma


def GammaV(mv, mx, kappa, alphaD):
    term = 0.0
    if mv > 2 * mx:
        term = alphaD * (mv * mv - 4 * mx * mx) * math.sqrt(mv * mv / 4 - mx * mx)
    if mv > 2 * MASS_ELECTRON:
        term += 4 * kappa ** 2 * alphaEM * (2 * MASS_ELECTRON ** 2 + mv * mv) * math.sqrt(mv * mv / 4 - MASS_ELECTRON ** 2)
    if mv > 2 * MASS_MUON:
        term += 4 * kappa ** 2 * alphaEM * (2 * MASS_MUON ** 2 + mv * mv) * math.sqrt(mv * mv / 4 - MASS_MUON ** 2)
    return 1.0 / (6.0 * mv * mv) * term


def GammaV_to_dm_dm(mv, mx, kappa, alphaD):
    if mv <= 2 * mx:
        return 0.0
    return alphaD * (mv * mv - 4 * mx * mx) * math.sqrt(mv * mv / 4.0 - mx * mx) / (6.0 * mv * mv)


def brV_to_dm_dm(mv, mx, kappa, alphaD):
    if mv <= 2 * mx:
        return 0.0
    return GammaV_to_dm_dm(mv, mx, kappa, alphaD) / GammaV(mv, mx, kappa, alphaD)


def d2brpi0_to_gamma_dm_dm(mv, mx, kappa, alphaD, s, theta):
    return (math.sqrt(1 - 4 * mx * mx / s) * (mpi0 ** 2 - s) ** 3 * (s - 4 * mx * mx) * alphaD * kappa ** 2
            * math.sin(theta) ** 3 / (8 * mpi0 ** 6 * pi * ((s - mv * mv) ** 2 + (mv * GammaV(mv, mx, kappa, alphaD)) ** 2)))


def dbrpi0_to_gamma_dm_dm(mv, mx, kappa, alphaD, s):
    return (math.sqrt(1.0 - 4.0 * mx * mx / s) * (mpi0 ** 2 - s) ** 3 * (s - 4.0 * mx * mx) * alphaD * kappa ** 2
            / (6.0 * mpi0 ** 6 * pi * ((s - mv * mv) ** 2 + (mv * GammaV(mv, mx, kappa, alphaD)) ** 2)))


def dbreta_to_gamma_dm_dm(mv, mx, kappa, alphaD, s):
    return (bretato2gamma / brpi0to2gamma * math.sqrt(1.0 - 4.0 * mx * mx / s) * (meta ** 2 - s) ** 3
            * (s - 4.0 * mx * mx) * alphaD * kappa ** 2
            / (6.0 * meta ** 6 * pi * ((s - mv * mv) ** 2 + (mv * GammaV(mv, mx, kappa, alphaD)) ** 2)))


# --- Integrator.cpp: double-exponential (tanh-sinh) rule -------------------------------------------------------

def _ddweight(k, h):
    return 1.0 / 2.0 * h * pi * math.cosh(k * h) / math.cosh(1.0 / 2.0 * pi * math.sinh(k * h)) ** 2


def _ddabscissa(k, h):
    # Stock returns long double; on arm64 macOS long double is double, which is what this reproduces.
    return math.tanh(1.0 / 2.0 * pi * math.sinh(h * k))


def _res_int(x, lo, hi):
    return 1.0 / 2.0 * (lo + hi - lo * x + hi * x)


def _func_check(f, a, lo, hi):
    def safe(x):
        try:
            v = f(x)
        except (ValueError, ZeroDivisionError):  # C++ returns NaN, which func_check zeroes
            return 0.0
        return 0.0 if math.isnan(v) else v
    return safe(_res_int(-a, lo, hi)) + safe(_res_int(a, lo, hi))


def _de(f, lo, hi, N, h, k0=0, step=1):
    if lo >= hi:
        return 0.0
    scale = 2.0 / (hi - lo)
    total = 0.0
    for k in range(k0, int(N) + 1 if step == 1 else int(N), step):
        a = _ddabscissa(k, h)
        if a >= 1:
            break
        total += _ddweight(k, h) * _func_check(f, a, lo, hi)
    return total / scale


def double_exponential_adapt(f, lo, hi, N, h, precision):
    dd0 = _de(f, lo, hi, N, h)
    if lo >= hi:
        return 0.0
    for _ in range(DDMAX):
        dd1 = dd0 + _de(f, lo, hi, N * 2.0, h, k0=int(N) + 1)  # DoubleExponential_Nout
        if abs(dd0 - dd1) > abs(dd0 * precision):
            dd0 = dd1
            N *= 2.0
            continue
        dd2 = 0.5 * dd0 + _de(f, lo, hi, 2 * N, h / 2.0, k0=1, step=2)  # DoubleExponential_hdub
        if abs(dd0 - dd2) > abs(dd0 * precision):
            dd0 = dd2
            h *= 0.5
            N *= 2.0
            continue
        break
    else:
        raise RuntimeError("double_exponential_adapt reached DDMAX without converging (stock only warns)")
    return dd0


def brpi0_to_gamma_dm_dm(mv, mx, kappa, alphaD):
    return double_exponential_adapt(lambda s: dbrpi0_to_gamma_dm_dm(mv, mx, kappa, alphaD, s),
                                    4 * mx * mx, mpi0 ** 2, 100, 0.1, 1e-4)


def breta_to_gamma_dm_dm(mv, mx, kappa, alphaD):
    return double_exponential_adapt(lambda s: dbreta_to_gamma_dm_dm(mv, mx, kappa, alphaD, s),
                                    4 * mx * mx, meta ** 2, 100, 0.1, 1e-4)


def pi0_channel(mv, mx, kappa, alphaD):
    """pion_decay_gen::Evaluate_Branching_Ratio: (branching ratio, off_shell flag).

    Raises for 2 mx > mpi0, as stock does. The MVP ports only the on-shell decay, so callers must refuse
    off_shell=True; GenDM also takes the on-shell path when 2mx <= mv < m_meson and 2mx >= 0.97 mv.
    """
    if 2 * mx > mpi0:
        raise ValueError("pi0_decay forbidden: 2 m_chi > m_pi0")
    br = brpi0_to_gamma_dm_dm(mv, mx, kappa, alphaD)
    onshell = brpi0toVgamma(mv, mx, kappa, alphaD) * brV_to_dm_dm(mv, mx, kappa, alphaD)
    off_shell = br > onshell * 1.30
    return br, off_shell


def eta_channel(mv, mx, kappa, alphaD):
    """eta_decay_gen::Evaluate_Branching_Ratio (stock returns BR 0 instead of raising when 2 mx > m_eta)."""
    if 2 * mx > meta:
        return 0.0, False
    br = breta_to_gamma_dm_dm(mv, mx, kappa, alphaD)
    onshell = bretatoVgamma(mv, mx, kappa, alphaD) * brV_to_dm_dm(mv, mx, kappa, alphaD)
    return br, br > onshell * 1.30


def uses_on_shell_decay(off_shell, mv, mx, m_meson):
    """pion_decay_gen::GenDM's branch: True when the on-shell DecayDM is used."""
    return (not off_shell) or ((mx * 2 <= mv) and (mv < m_meson) and (2 * mx >= 0.97 * mv))
