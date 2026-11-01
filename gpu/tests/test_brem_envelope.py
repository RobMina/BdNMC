"""The proton-brem envelope sampler (parent_sampler="envelope"): its bound holds, it samples the stock sampler's
(z, pT) law, its results do not depend on the batch size, and a new m_V does not recompile."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

import bdnmc_jax  # noqa: F401
from bdnmc_jax import brem, card as cardmod, pipeline, rng
from bdnmc_jax.constants import MASS_PROTON as mp

CARDS = Path(__file__).resolve().parents[1] / "phase0" / "cards"
BASES = ("mb_brem_m005", "he_brem_cuboid_m005")

COMPILES = []
jax.monitoring.register_event_duration_secs_listener(
    lambda event, secs, **kw: COMPILES.append(secs) if event == "/jax/core/compile/backend_compile_duration" else None)


def _chan(stem, sampler):
    return pipeline.setup(cardmod.parse_card(CARDS / f"{stem}.dat"), sampler)[2][0]


def _card_with(tmp_path, stem, **keys):
    lines = (CARDS / f"{stem}.dat").read_text().splitlines()
    for k, v in keys.items():
        lines = [f"{k} {v}" if ln.split()[:1] == [k] else ln for ln in lines]
    path = tmp_path / f"{stem}_{'_'.join(f'{k}{v}' for k, v in keys.items())}.dat"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_setup_builds_the_envelope_and_keeps_stock_the_default():
    stock, env = _chan("he_brem_cuboid_m005", "stock"), _chan("he_brem_cuboid_m005", "envelope")
    assert isinstance(stock.dist, brem.ProtonBrem) and isinstance(stock.fmax, float)
    assert isinstance(env.dist, brem.BremEnvelope) and env.dist.base == stock.dist
    assert isinstance(env.fmax, brem.EnvelopeTables) and float(env.fmax.efficiency) > 0.7
    assert env.vnum == stock.vnum and env.kind == "brem"
    # the default is the stock sampler
    assert isinstance(pipeline.setup(cardmod.parse_card(CARDS / "he_brem_cuboid_m005.dat"))[2][0].dist, brem.ProtonBrem)


@pytest.mark.parametrize("stem", BASES)
def test_bound_holds_on_random_points(stem):
    """Points uniform over the whole window (the builder checks cell by cell), at several m_V."""
    base = _chan(stem, "stock").dist
    env = brem.BremEnvelope(base)
    rs = np.random.default_rng(5)
    for mv in (0.003, 0.1, 0.6):
        t = env.tables(mv, n_check=200_000)
        u = rs.random((400_000, 2))
        z = base.zmin + (base.zmax - base.zmin) * u[:, 0]
        pt = base.ptmin + (base.ptmax - base.ptmin) * u[:, 1]
        # the window corners too (the density peaks at (zmin, ptmin))
        z = np.concatenate([z, [base.zmin, base.zmin, base.zmax, base.zmax]])
        pt = np.concatenate([pt, [base.ptmin, base.ptmax, base.ptmin, base.ptmax]])
        iz = np.clip(np.searchsorted(np.asarray(t.z_lo), z, side="right") - 1, 0, env.n_z - 1)
        ip = np.clip(np.searchsorted(np.asarray(t.pt_lo), pt, side="right") - 1, 0, env.n_pt - 1)
        f = np.asarray(base.shape(jnp.asarray(z), jnp.asarray(pt), mv))
        assert (f <= np.asarray(t.bound)[iz * env.n_pt + ip]).all(), mv
        # the cells tile the window exactly
        np.testing.assert_allclose(np.asarray(t.z_lo)[0], base.zmin)
        np.testing.assert_allclose(np.asarray(t.z_lo)[-1] + np.asarray(t.dz)[-1], base.zmax, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(t.pt_lo)[-1] + np.asarray(t.dpt)[-1], base.ptmax, rtol=1e-12)
        assert np.asarray(t.cdf)[-1] == 1.0 and (np.diff(np.asarray(t.cdf)) > 0).all()


def test_bound_violation_is_reported():
    ch = _chan("mb_brem_m005", "envelope")
    t = ch.fmax
    tight = brem.EnvelopeTables(t.z_lo, t.dz, t.pt_lo, t.dpt, t.bound / 2, t.cdf, t.efficiency)
    _, info = brem.sample_block(ch.dist, jax.random.key(0), 4096, tight, ch.m_v)
    assert int(info["over_bar"]) > 0


def test_degenerate_window_is_refused():
    with pytest.raises(NotImplementedError):
        brem.BremEnvelope(brem.ProtonBrem(8.9, 0.3, 0.7, 0.2, 0.2)).tables(0.1)


def _z_pt(v, beam_energy):
    v = np.asarray(v)
    return v[:, 3] / np.sqrt(beam_energy ** 2 - mp * mp), np.hypot(v[:, 1], v[:, 2])


@pytest.mark.parametrize("stem", BASES)
def test_envelope_samples_the_stock_law(stem):
    """2 blocks from each sampler (independent seeds): KS in z and pT, and a two-sample chi2 of the joint (z, pT)
    law in 8 x 8 cells of equal stock population."""
    draws = {}
    for name, seed in (("stock", 1), ("envelope", 2)):
        ch = _chan(stem, name)
        zs, pts = [], []
        for b in range(2):
            v, info = brem.sample_block(ch.dist, rng.block_key(seed, 0, b), rng.BLOCK, ch.fmax, ch.m_v)
            assert int(info["filled"]) == rng.BLOCK and int(info["over_bar"]) == 0
            if name == "envelope":
                assert int(info["proposed"]) < 2 * rng.BLOCK  # stock needs 8 (mb) and 96 (he) per V
            z, pt = _z_pt(v, ch.dist.beam_energy)
            zs.append(z)
            pts.append(pt)
        draws[name] = np.concatenate(zs), np.concatenate(pts)
    (zs, ps), (ze, pe) = draws["stock"], draws["envelope"]
    assert stats.ks_2samp(zs, ze).pvalue > 1e-3
    assert stats.ks_2samp(ps, pe).pvalue > 1e-3
    ez = np.quantile(zs, np.linspace(0, 1, 9)[1:-1])
    ep = np.quantile(ps, np.linspace(0, 1, 9)[1:-1])
    a = np.bincount(np.digitize(zs, ez) * 8 + np.digitize(ps, ep), minlength=64)
    b = np.bincount(np.digitize(ze, ez) * 8 + np.digitize(pe, ep), minlength=64)
    chi2 = ((a - b) ** 2 / (a + b)).sum()  # equal sample sizes
    assert stats.chi2.sf(chi2, len(a) - 1) > 1e-3


# The recoil electron (and Recoil_DM = DM - electron) can differ in the last bit between batch/cap shapes on CPU,
# with either sampler (XLA's codegen of the recoil kinematics depends on the array length). On x86-64 (the CI
# runner) prob differs too, by up to 8.4e-12 relative in about 2 % of the candidates. Everything else, the sampled V
# and DM and the candidate set included, is bit-identical.
RTOL = {"electron": 1e-15, "recoil_dm": 1e-15, "prob": 1e-10}


@pytest.mark.parametrize("sampler", ["stock", "envelope"])
def test_batch_and_cap_do_not_change_results(sampler):
    card, n = CARDS / "he_brem_cuboid_m005.dat", 3 * rng.BLOCK - 77
    a = pipeline.candidates(card, n, seed=11, batch=1, cap=64, parent_sampler=sampler)
    b = pipeline.candidates(card, n, seed=11, batch=3, cap=8192, parent_sampler=sampler)
    assert a["parent_over_bar"] == b["parent_over_bar"] == 0 and a["parent_sampler"] == sampler
    assert len(a["candidates"]["prob"]) > 0
    for k in a["candidates"]:
        if k in RTOL:
            np.testing.assert_allclose(a["candidates"][k], b["candidates"][k], rtol=RTOL[k], atol=1e-17, err_msg=k)
        else:
            np.testing.assert_array_equal(a["candidates"][k], b["candidates"][k], err_msg=k)
    c = pipeline.candidates(card, n, seed=12, batch=3, cap=8192, parent_sampler=sampler)["candidates"]
    assert not np.array_equal(c["dm"][:10], b["candidates"]["dm"][:10])


def test_new_dark_photon_mass_does_not_recompile(tmp_path):
    """Goal P2: the envelope's per-mass tables are traced, so a second m_V (and m_chi) compiles nothing."""
    pipeline.blocks_batch.clear_cache()
    first = _card_with(tmp_path, "mb_brem_m005", dark_photon_mass=0.1)
    second = _card_with(tmp_path, "mb_brem_m005", dark_photon_mass=0.25, dark_matter_mass=0.02)
    pipeline.candidates(first, 1000, seed=2, parent_sampler="envelope")
    n0 = len(COMPILES)
    res = pipeline.candidates(second, 1000, seed=2, parent_sampler="envelope")
    assert len(COMPILES) == n0, f"{len(COMPILES) - n0} compiles for the second mass"
    assert pipeline.blocks_batch._cache_size() == 1 and res["parent_over_bar"] == 0
