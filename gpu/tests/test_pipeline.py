"""Smoke and reproducibility tests of the pi0 pipeline on one block of trials."""
from pathlib import Path

import numpy as np
import pytest

import bdnmc_jax  # noqa: F401
from bdnmc_jax import pipeline, rng

CARDS = Path(__file__).resolve().parents[1] / "phase0" / "cards"


@pytest.fixture(scope="module")
def mb_pi0():
    return pipeline.run(CARDS / "mb_pi0_m005.dat", rng.BLOCK, seed=7)


def test_runs_and_is_finite(mb_pi0):
    c = mb_pi0["candidates"]
    assert mb_pi0["n_hits"] > 0
    live = c["prob"] > 0
    assert np.isfinite(c["prob"]).all() and (c["prob"] >= 0).all()
    for k in ("dm", "electron", "recoil_dm", "vertex"):
        assert np.isfinite(c[k][live]).all(), k
    assert mb_pi0["accepted"] > 0 and mb_pi0["se"] > 0


def test_zero_probability_hits_are_never_accepted():
    # The ref card at seed 1 has hits too soft for the 35 MeV recoil window in its first block.
    res = pipeline.run(CARDS / "ref" / "mb_pi0_m005.dat", rng.BLOCK, seed=1)
    c = res["candidates"]
    dead = c["prob"] == 0
    assert dead.any()
    assert not res["accepted_mask"][dead].any() and not c["ang_ok"][dead].any()
    assert np.isfinite(c["electron"][~dead]).all()


def test_same_seed_same_result(mb_pi0):
    again = pipeline.run(CARDS / "mb_pi0_m005.dat", rng.BLOCK, seed=7)
    np.testing.assert_array_equal(again["candidates"]["prob"], mb_pi0["candidates"]["prob"])
    np.testing.assert_array_equal(again["candidates"]["electron"], mb_pi0["candidates"]["electron"])


def test_truncated_run_is_a_prefix(mb_pi0):
    # A trial's random numbers depend only on its index, so fewer trials give a subset of the same candidates.
    half = pipeline.run(CARDS / "mb_pi0_m005.dat", rng.BLOCK // 2, seed=7)
    full = mb_pi0["candidates"]
    keep = full["trial"] < rng.BLOCK // 2
    np.testing.assert_array_equal(half["candidates"]["prob"], full["prob"][keep])


def test_new_mass_does_not_recompile():
    # Masses and couplings are traced (goal P2): the four ref masses of one card share one compiled block.
    pipeline.blocks_batch.clear_cache()
    for mev in (1, 5, 20, 45):
        pipeline.candidates(CARDS / "ref" / f"mb_pi0_m{mev:03d}.dat", 1000, seed=3)
    assert pipeline.blocks_batch._cache_size() == 1


def test_batch_and_cap_do_not_change_results():
    # P3: blocks per device call, the starting cap and its doubling change only scheduling and padding.
    card, n = CARDS / "ref" / "he_pi0_cuboid_m005.dat", 5 * rng.BLOCK - 123
    a = pipeline.candidates(card, n, seed=11, batch=1, cap=64)["candidates"]
    b = pipeline.candidates(card, n, seed=11, batch=3, cap=8192)["candidates"]
    assert len(a["prob"]) > 64
    for k in pipeline.CANDIDATE_KEYS:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_electron_inside_kinematic_window(mb_pi0):
    c = mb_pi0["candidates"]
    Ee = c["electron"][:, 0]
    assert (Ee >= 0.035 - 1e-12).all() and (Ee <= 0.9 + 1e-12).all()
    # Momentum is conserved at the scatter: DM = electron + recoil DM (three-momenta).
    np.testing.assert_allclose(c["dm"][:, 1:], c["electron"][:, 1:] + c["recoil_dm"][:, 1:], rtol=0, atol=1e-12)
