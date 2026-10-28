"""Meson-decay pipeline for one channel: parent -> decay -> detector -> scatter probability -> recoil electron.

Option A of the plan (D3): stock's physics and samplers, batched. Two deviations from stock's event loop, both
needed for batching and both removing an order dependence:
- unweighting is done in two passes, against the maximum scatter probability of the whole run, not a running
  maximum (so the unweighted `se` here is unbiased; stock's is biased high, see gpu-bdnmc-plan.md);
- the trial count is fixed (stock stops at `samplesize` events or `max_trials`).

Pass 1 (`block`, jitted) handles BLOCK trials and keeps only the DM that reach the detector ("candidates"), with
their recoil electron, interaction point and acceptance uniform already drawn. Pass 2 (`run`) takes the global
maximum and accepts. Weighted mode is the same pass 1 with every candidate kept, weight = its probability.
"""
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from . import branching, card as cardmod, decay, detector, electron, parents, rng
from .constants import MASS_ELECTRON as Me, MASS_PION_NEUTRAL, MASS_ETA, pi, speed_of_light
from .kinematics import momentum, phi, rotate_y, rotate_z, theta


@dataclass(frozen=True)
class Channel:
    name: str
    m_meson: float
    dist: object
    fmax: float
    vnum: float
    m_v: float
    m_dm: float


def setup(card):
    """Build the detector, signal model and channels from a card; refuse what is not ported."""
    if card.model not in ("Dark_Photon", "Dark_Photon_DM"):
        raise NotImplementedError(f"model {card.model}")
    if card.signal_channel != "NCE_electron":
        raise NotImplementedError(f"signal channel {card.signal_channel}")
    if card.timing_cut > 0:
        raise NotImplementedError("timing_cut")
    det = detector.from_card(card)
    sc = electron.ElectronScatter(card.mass_dm, card.mass_v, card.epsilon, card.alpha_d,
                                  card.min_scatter_energy, card.max_scatter_energy,
                                  card.min_scatter_angle, card.max_scatter_angle)
    chans = []
    for ch in card.channels:
        if ch.production_channel != "pi0_decay" or ch.dist_mods:
            raise NotImplementedError(f"channel {ch.production_channel} / dist_mods {ch.dist_mods}")
        br, off = branching.pi0_channel(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
        if not branching.uses_on_shell_decay(off, card.mass_v, card.mass_dm, MASS_PION_NEUTRAL):
            raise NotImplementedError("off-shell pi0 decay")
        if ch.production_distribution == "pi0_sanfordwang":
            if "sanfordwang_file" in ch.raw or "distribution_parameter_file" in ch.raw:
                raise NotImplementedError("Sanford-Wang parameter file")
            dist = parents.SanfordWang()
            fmax = dist.fmax
        elif ch.production_distribution == "bmpt":
            if ch.max_meson_momentum > 0:
                raise NotImplementedError("max_meson_momentum")
            dist = parents.BMPT(card.beam_energy, card.target_mass_number)
            fmax = dist.find_fmax()
        else:
            raise NotImplementedError(f"distribution {ch.production_distribution}")
        vnum = br * card.pot * ch.meson_per_pi0 * card.pi0_per_pot  # Kinetic_Mixing.cpp:141
        chans.append(Channel(ch.production_channel, MASS_PION_NEUTRAL, dist, float(fmax), vnum,
                             card.mass_v, card.mass_dm))
    if len(chans) != 1:
        raise NotImplementedError("exactly one production channel for now")
    return det, sc, chans


def _recoil(sc, key, key_phi, dm, live, u_pos, t_in, t_out, max_rounds=10_000):
    """Electron_Scatter::scatterevent + the bookkeeping of probscatter(partlist): recoil electron, Recoil_DM and
    the interaction point, for each candidate DM (rows of dm (k, 4)). Only `live` rows (prob > 0) are sampled:
    stock never calls scatterevent for a zero-probability DM, whose energy window may be empty (then the recoil
    would be outside the kinematic range and ThetaEe NaN)."""
    E = dm[:, 0]
    lo, hi = electron.scat_min(sc, E), electron.scat_max(sc, E)
    dmax = electron.dsigma_dEe(lo, E, sc.m_dm, sc.m_v, sc.kappa, sc.alpha_d)
    k = dm.shape[0]

    def cond(st):
        xe, done, r = st
        return jnp.any(~done) & (r < max_rounds)

    def body(st):
        xe, done, r = st
        u = jax.random.uniform(jax.random.fold_in(key, r), (k, 2), dtype=jnp.float64)
        x = u[:, 0] * (hi - lo) + lo
        acc = ~done & (electron.dsigma_dEe(x, E, sc.m_dm, sc.m_v, sc.kappa, sc.alpha_d) > dmax * u[:, 1])
        return jnp.where(acc, x, xe), done | acc, r + 1

    xe, done, _ = jax.lax.while_loop(cond, body, (jnp.where(live, 0.0, lo), ~live, 0))
    th_e = electron.theta_ee(xe, E, sc.m_dm)
    ph_e = jax.random.uniform(key_phi, (k,), dtype=jnp.float64) * 2 * pi
    pe = jnp.sqrt(xe * xe - Me ** 2)
    px, py, pz = pe * jnp.sin(th_e) * jnp.cos(ph_e), pe * jnp.sin(th_e) * jnp.sin(ph_e), pe * jnp.cos(th_e)
    e4 = jnp.stack([jnp.sqrt(Me * Me + px * px + py * py + pz * pz), px, py, pz], axis=-1)
    e4 = rotate_z(rotate_y(e4, theta(dm)), phi(dm))
    rp = dm[:, 1:] - e4[:, 1:]
    r4 = jnp.concatenate([jnp.sqrt(sc.m_dm ** 2 + rp[:, 0] ** 2 + rp[:, 1] ** 2 + rp[:, 2] ** 2)[:, None], rp], axis=-1)
    # Particle::Generate_Position: u in [crossing0, crossing1]; DM origin is (0,0,0,0) for these parents.
    t = t_in + (t_out - t_in) * u_pos
    mom = momentum(dm)
    speed = mom / dm[:, 0]
    time = mom * t / speed / speed_of_light
    pos = (time * speed * speed_of_light)[:, None] * dm[:, 1:] / mom[:, None]
    ang_ok = ((sc.min_angle <= 0) | (theta(e4) > sc.min_angle)) & ((sc.max_angle > 2 * pi) | (theta(e4) < sc.max_angle))
    return e4, r4, jnp.concatenate([pos, time[:, None]], axis=-1), ang_ok & live, done


@partial(jax.jit, static_argnames=("det", "sc", "ch", "cap"))
def block(det, sc, ch, e_density, seed, chan_index, block_index, n_valid, cap):
    """Pass 1 for one block. Returns per-block sums and up to `cap` candidates (fixed shape, `count` valid)."""
    n = rng.BLOCK
    key = rng.block_key(seed, chan_index, block_index)
    par, pinfo = parents.sample_block(ch.dist, key, n, ch.fmax, ch.m_meson)
    u = rng.uniform(key, rng.DECAY, (n, 4))
    med, d1, d2 = decay.decay_dm(par, ch.m_meson, ch.m_v, ch.m_dm, u[:, 0], u[:, 1], u[:, 2], u[:, 3])
    valid = jnp.arange(n) < n_valid
    dm = jnp.concatenate([d1, d2])  # candidate slot = dm index * n + trial index
    L, t_in, t_out = detector.ldet(det, jnp.zeros((2 * n, 3)), dm[:, 1:])
    hit = (L > 0) & jnp.concatenate([valid, valid])
    prob = jnp.where(hit, electron.scatter_probability(sc, dm[:, 0], L, e_density), 0.0)
    idx = jnp.nonzero(hit, size=cap, fill_value=2 * n)[0]
    count = jnp.sum(hit)
    safe = jnp.minimum(idx, 2 * n - 1)
    cdm, ct_in, ct_out = dm[safe], t_in[safe], t_out[safe]
    u_acc = rng.uniform(key, rng.ACCEPT, (2 * n,))[safe]
    u_pos = rng.uniform(key, rng.POSITION, (2 * n,))[safe]
    live = prob[safe] > 0
    e4, r4, vtx, ang_ok, recoil_done = _recoil(sc, jax.random.fold_in(key, rng.RECOIL),
                                               jax.random.fold_in(key, rng.RECOIL_PHI), cdm, live, u_pos, ct_in, ct_out)
    trial = safe % n
    return {
        "count": count, "prob_sum": jnp.sum(prob), "parent_info": pinfo,
        "slot": idx, "trial": trial, "prob": prob[safe], "u_acc": u_acc, "ang_ok": ang_ok,
        "recoil_done": recoil_done, "dm": cdm, "electron": e4, "recoil_dm": r4, "vertex": vtx,
        "parent": par[trial], "mediator": med[trial], "L": L[safe],
    }


CANDIDATE_KEYS = ("block", "trial", "slot", "prob", "u_acc", "ang_ok", "dm", "electron", "recoil_dm", "vertex",
                  "parent", "mediator", "L")


def candidates(card_path, n_trials, seed=None, blocks=None, cap_fraction=0.25):
    """Pass 1 over blocks `blocks` (a range; default all blocks of n_trials). Shards over disjoint block ranges
    combine exactly into the full run, because each block's random numbers depend only on (seed, block)."""
    card = cardmod.parse_card(card_path)
    seed = card.seed if seed is None else seed
    det, sc, (ch,) = setup(card)
    cap = int(2 * rng.BLOCK * cap_fraction)
    nblocks = -(-n_trials // rng.BLOCK)
    blocks = range(nblocks) if blocks is None else blocks
    if blocks.start < 0 or blocks.stop > nblocks:
        raise ValueError(f"blocks {blocks} outside 0..{nblocks}")
    parts, prob_sum, n_hits, over_bar = [], 0.0, 0, 0
    for b in blocks:
        nv = min(rng.BLOCK, n_trials - b * rng.BLOCK)
        out = jax.device_get(block(det, sc, ch, card.e_density, seed, 0, b, nv, cap))
        if out["count"] > cap:
            raise RuntimeError(f"block {b}: {out['count']} candidates exceed cap {cap}; raise cap_fraction")
        pi_ = out["parent_info"]
        if pi_["filled"] != rng.BLOCK:
            raise RuntimeError(f"block {b}: parent sampler ran out of rounds")
        if pi_["over_bar"] and not isinstance(ch.dist, parents.SanfordWang):
            raise RuntimeError(f"block {b}: {pi_['over_bar']} parent proposals above the bar")
        if not out["recoil_done"][: out["count"]].all():
            raise RuntimeError(f"block {b}: recoil sampler ran out of rounds")
        # ThetaEe's acos argument can round above 1 at the recoil endpoint (stock shares this and writes NaN).
        live = out["prob"][: out["count"]] > 0
        if not np.isfinite(out["electron"][: out["count"]][live]).all():
            raise RuntimeError(f"block {b}: non-finite recoil electron")
        k = int(out["count"])
        parts.append({key: np.asarray(out[key])[:k] for key in CANDIDATE_KEYS if key != "block"}
                     | {"block": np.full(k, b)})
        prob_sum += float(out["prob_sum"])
        n_hits += k
        over_bar += int(pi_["over_bar"])
    cat = {key: np.concatenate([p[key] for p in parts]) for key in CANDIDATE_KEYS}
    trials = sum(min(rng.BLOCK, n_trials - b * rng.BLOCK) for b in blocks)
    return {"card": str(card_path), "seed": seed, "n_trials": n_trials, "blocks": [blocks.start, blocks.stop],
            "trials": trials, "n_hits": n_hits, "prob_sum": prob_sum, "parent_over_bar": over_bar,
            "vnum": ch.vnum, "efficiency": card.efficiency, "candidates": cat}


def merge(shards):
    """Combine pass-1 shards of one run (disjoint block ranges, same card, seed and n_trials)."""
    first = shards[0]
    for s in shards:
        if (s["card"], s["seed"], s["n_trials"]) != (first["card"], first["seed"], first["n_trials"]):
            raise ValueError("shards from different runs")
    blocks = sorted(b for s in shards for b in range(*s["blocks"]))
    if len(blocks) != len(set(blocks)):
        raise ValueError("overlapping shards")
    order = sorted(range(len(shards)), key=lambda i: shards[i]["blocks"][0])
    cat = {k: np.concatenate([shards[i]["candidates"][k] for i in order]) for k in CANDIDATE_KEYS}
    return first | {"blocks": blocks, "trials": sum(s["trials"] for s in shards),
                    "n_hits": sum(s["n_hits"] for s in shards), "prob_sum": sum(s["prob_sum"] for s in shards),
                    "parent_over_bar": sum(s["parent_over_bar"] for s in shards), "candidates": cat}


def finish(res, weighted=False):
    """Pass 2: accept against the run's maximum probability and compute se (unweighted and weighted)."""
    cat, n = res["candidates"], res["trials"]
    pmax = float(cat["prob"].max()) if len(cat["prob"]) else 0.0
    weight_sum = float(cat["prob"][cat["ang_ok"]].sum())
    if weighted:
        accepted = cat["ang_ok"]
        se = weight_sum / n * res["vnum"] * res["efficiency"]
    else:
        accepted = (cat["prob"] > cat["u_acc"] * pmax) & cat["ang_ok"]
        se = accepted.sum() / n * res["vnum"] * pmax * res["efficiency"]
    return res | {"pmax": pmax, "se": se, "se_weighted": weight_sum / n * res["vnum"] * res["efficiency"],
                  "accepted": int(accepted.sum()), "accepted_mask": accepted}


def run(card_path, n_trials, seed=None, cap_fraction=0.25, weighted=False):
    """Run n_trials trials of the card's single channel (pass 1 and pass 2 in one process)."""
    return finish(candidates(card_path, n_trials, seed, cap_fraction=cap_fraction), weighted)
