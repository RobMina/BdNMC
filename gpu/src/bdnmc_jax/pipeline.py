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
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from . import branching, brem, card as cardmod, decay, detector, electron, parents, rng
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
    kind: str = "meson"  # "meson": meson -> gamma V(-> chi chi); "brem": proton bremsstrahlung V -> chi chi


# Masses, the sampler bar and Vnum are traced leaves under jit (no recompile per mass, goal P2); the name, the
# parent distribution (beam, target and the brem window only) and the kind are static.
jax.tree_util.register_dataclass(Channel, data_fields=["m_meson", "fmax", "vnum", "m_v", "m_dm"],
                                 meta_fields=["name", "dist", "kind"])


@lru_cache(maxsize=None)
def _envelope(dist):
    return parents.make_envelope(dist)


PARENT_SAMPLERS = ("stock", "envelope")
# Meson channels (Kinetic_Mixing.cpp:133-155): meson mass, Evaluate_Branching_Ratio, the ported distributions.
# Stock also accepts bmpt for eta and the K0 distributions for pi0; those pairs are not validated, so refused.
MESONS = {"pi0_decay": (MASS_PION_NEUTRAL, branching.pi0_channel, ("pi0_sanfordwang", "bmpt")),
          "eta_decay": (MASS_ETA, branching.eta_channel, ("k0_sanfordwang", "bpmt_k0"))}


def setup(card, parent_sampler="stock"):
    """Build the detector, signal model and channels from a card; refuse what is not ported. `parent_sampler`:
    "stock" (stock's proposal against one bar) or "envelope" (the same law, piecewise bound; parents.Envelope)."""
    if parent_sampler not in PARENT_SAMPLERS:
        raise ValueError(f"parent_sampler {parent_sampler!r} not in {PARENT_SAMPLERS}")
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
        if ch.dist_mods:
            raise NotImplementedError(f"dist_mods {ch.dist_mods}")
        if ch.production_channel in ("V_decay", "Brem_V") and ch.production_distribution == "proton_brem":
            chans.append(_brem_channel(card, ch, parent_sampler))
            continue
        if ch.production_channel not in MESONS:
            raise NotImplementedError(f"channel {ch.production_channel} with {ch.production_distribution}")
        m_meson, channel_br, dists = MESONS[ch.production_channel]
        if ch.production_distribution not in dists:
            raise NotImplementedError(f"{ch.production_channel} with {ch.production_distribution}")
        br, off = channel_br(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
        if not branching.uses_on_shell_decay(off, card.mass_v, card.mass_dm, m_meson):
            raise NotImplementedError(f"off-shell {ch.production_channel}")
        if br <= 0:
            raise NotImplementedError(f"{ch.production_channel} closed (2 m_chi > meson mass)")
        if ch.production_distribution in ("pi0_sanfordwang", "k0_sanfordwang"):
            if "sanfordwang_file" in ch.raw or "distribution_parameter_file" in ch.raw:
                raise NotImplementedError("Sanford-Wang parameter file")
            dist = parents.SanfordWang() if ch.production_distribution == "pi0_sanfordwang" else parents.SanfordWang.k0()
            fmax = dist.fmax
        elif ch.production_distribution in ("bmpt", "bpmt_k0"):
            if ch.max_meson_momentum > 0:
                raise NotImplementedError("max_meson_momentum")
            dist = parents.BMPT(card.beam_energy, card.target_mass_number,
                                choice="pi0" if ch.production_distribution == "bmpt" else "k0")
            fmax = dist.find_fmax()
        else:
            raise NotImplementedError(f"distribution {ch.production_distribution}")
        if parent_sampler == "envelope":
            dist, fmax = _envelope(dist), 1.0
        vnum = br * card.pot * ch.meson_per_pi0 * card.pi0_per_pot  # Kinetic_Mixing.cpp:141
        chans.append(Channel(ch.production_channel, m_meson, dist, float(fmax), vnum,
                             card.mass_v, card.mass_dm))
    if len(chans) != 1:
        raise NotImplementedError("exactly one production channel for now")
    return det, sc, chans


def _brem_channel(card, ch, parent_sampler):
    """V_decay with production_distribution proton_brem (Kinetic_Mixing.cpp:186-198): Vnum = V_prod_rate x POT."""
    if parent_sampler != "stock":
        raise NotImplementedError("the envelope sampler is not ported for proton_brem")
    if 2 * card.mass_dm >= card.mass_v:
        raise NotImplementedError("V -> chi chi is closed (2 m_chi >= m_V)")
    dist = brem.ProtonBrem(card.beam_energy, ch.zmin, ch.zmax, ch.ptmin, ch.ptmax)
    dist.check()
    vnum = dist.v_prod_rate(card.epsilon, card.mass_v) * card.pot
    return Channel("Dark_Bremsstrahlung", card.mass_v, dist, float(dist.find_fmax(card.mass_v)), vnum,
                   card.mass_v, card.mass_dm, kind="brem")


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


def _block(det, sc, ch, e_density, seed, chan_index, block_index, n_valid, cap):
    """Pass 1 for one block. Returns per-block sums and up to `cap` candidates (fixed shape, `count` valid)."""
    n = rng.BLOCK
    key = rng.block_key(seed, chan_index, block_index)
    if ch.kind == "brem":
        med, pinfo = brem.sample_block(ch.dist, key, n, ch.fmax, ch.m_v)
        par = med  # no meson: the V is the parent of the event
        u = rng.uniform(key, rng.DECAY, (n, 2))
        d1, d2 = decay.two_body(med, ch.m_v, ch.m_dm, ch.m_dm, u[:, 0], u[:, 1])
    else:
        par, pinfo = parents.sample_block(ch.dist, key, n, ch.fmax, ch.m_meson)
        u = rng.uniform(key, rng.DECAY, (n, 4))
        med, d1, d2 = decay.decay_dm(par, ch.m_meson, ch.m_v, ch.m_dm, u[:, 0], u[:, 1], u[:, 2], u[:, 3])
    valid = jnp.arange(n) < n_valid
    dm = jnp.concatenate([d1, d2])  # candidate slot = dm index * n + trial index
    L, t_in, t_out = detector.ldet(det, jnp.zeros((2 * n, 3)), dm[:, 1:])
    hit = (L > 0) & jnp.concatenate([valid, valid])
    if ch.kind == "brem":
        # V_decay_gen::GenDM tests `det_int(dm1) > 0 || det_int(dm2) > 0`: the second DM is kept only when the
        # first misses, so a brem event never has two DM.
        hit = hit & jnp.concatenate([jnp.ones(n, bool), ~(L[:n] > 0)])
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
        "crossing": jnp.stack([ct_in, ct_out], axis=-1),  # stock's Particle::crossing (entry, exit); p x t is metres
    }


@partial(jax.jit, static_argnames=("det", "cap"))
def blocks_batch(det, sc, ch, e_density, seed, chan_index, block_indices, n_valid, cap):
    """Pass 1 for a batch of blocks in one device call (leading axis = block). Each block's random numbers depend
    only on (seed, channel, block), so the batch size does not change the results (goal P3). Compiled once per
    detector, parent distribution, cap and batch size; `sc` and `ch` values (masses, couplings) are traced."""
    return jax.vmap(lambda b, nv: _block(det, sc, ch, e_density, seed, chan_index, b, nv, cap))(block_indices, n_valid)


CANDIDATE_KEYS = ("block", "trial", "slot", "prob", "u_acc", "ang_ok", "dm", "electron", "recoil_dm", "vertex",
                  "parent", "mediator", "L", "crossing")


def candidates(card_path, n_trials, seed=None, blocks=None, batch=8, cap=1024, parent_sampler="stock"):
    """Pass 1 over blocks `blocks` (a range; default all blocks of n_trials), `batch` blocks per device call.
    Shards over disjoint block ranges combine exactly into the full run, because each block's random numbers
    depend only on (seed, block). `cap` is the starting number of candidate rows kept per block: a batch with a
    block above it is rerun with the cap doubled (one recompile), which changes only the padding."""
    card = cardmod.parse_card(card_path)
    seed = card.seed if seed is None else seed
    det, sc, (ch,) = setup(card, parent_sampler)
    nblocks = -(-n_trials // rng.BLOCK)
    blocks = range(nblocks) if blocks is None else blocks
    if blocks.start < 0 or blocks.stop > nblocks:
        raise ValueError(f"blocks {blocks} outside 0..{nblocks}")
    chunks = [list(blocks[i:i + batch]) for i in range(0, len(blocks), batch)]

    def launch(chunk, cap):
        # Pad a short last batch with n_valid = 0 blocks (no hits) so the batch shape, and the compile, stay fixed.
        bi = chunk + [chunk[-1]] * (batch - len(chunk))
        nv = [min(rng.BLOCK, n_trials - b * rng.BLOCK) for b in chunk] + [0] * (batch - len(chunk))
        return blocks_batch(det, sc, ch, card.e_density, seed, 0, jnp.asarray(bi), jnp.asarray(nv), cap)

    parts, prob_sum, n_hits, over_bar = [], 0.0, 0, 0
    pending = launch(chunks[0], cap) if chunks else None
    for i, chunk in enumerate(chunks):
        out, used_cap = jax.device_get(pending), cap
        if i + 1 < len(chunks):
            pending = launch(chunks[i + 1], cap)  # dispatched before this batch is checked (async)
        while out["count"][: len(chunk)].max() > used_cap:
            used_cap *= 2
            out = jax.device_get(launch(chunk, used_cap))
        if used_cap > cap:
            cap = used_cap
            if i + 1 < len(chunks):
                pending = launch(chunks[i + 1], cap)
        for j, b in enumerate(chunk):
            o = jax.tree.map(lambda x: x[j], out)
            pi_ = o["parent_info"]
            if pi_["filled"] != rng.BLOCK:
                raise RuntimeError(f"block {b}: parent sampler ran out of rounds")
            if pi_["over_bar"] and not isinstance(ch.dist, parents.SanfordWang):
                raise RuntimeError(f"block {b}: {pi_['over_bar']} parent proposals above the bar")
            k = int(o["count"])
            if not o["recoil_done"][:k].all():
                raise RuntimeError(f"block {b}: recoil sampler ran out of rounds")
            # ThetaEe's acos argument can round above 1 at the recoil endpoint (stock shares this and writes NaN).
            live = o["prob"][:k] > 0
            if not np.isfinite(o["electron"][:k][live]).all():
                raise RuntimeError(f"block {b}: non-finite recoil electron")
            parts.append({key: np.asarray(o[key])[:k] for key in CANDIDATE_KEYS if key != "block"}
                         | {"block": np.full(k, b)})
            prob_sum += float(o["prob_sum"])
            n_hits += k
            over_bar += int(pi_["over_bar"])
    cat = {key: np.concatenate([p[key] for p in parts]) for key in CANDIDATE_KEYS}
    trials = sum(min(rng.BLOCK, n_trials - b * rng.BLOCK) for b in blocks)
    return {"card": str(card_path), "seed": seed, "n_trials": n_trials, "blocks": [blocks.start, blocks.stop],
            "trials": trials, "n_hits": n_hits, "prob_sum": prob_sum, "parent_over_bar": over_bar,
            "vnum": ch.vnum, "efficiency": card.efficiency, "batch": batch, "cap": cap,
            "parent_sampler": parent_sampler, "candidates": cat}


def merge(shards):
    """Combine pass-1 shards of one run (disjoint block ranges, same card, seed and n_trials)."""
    first = shards[0]
    run_id = lambda s: (s["card"], s["seed"], s["n_trials"], s.get("parent_sampler", "stock"))  # noqa: E731
    for s in shards:
        if run_id(s) != run_id(first):
            raise ValueError("shards from different runs")
    blocks = sorted(b for s in shards for b in range(*s["blocks"]))
    if len(blocks) != len(set(blocks)):
        raise ValueError("overlapping shards")
    order = sorted(range(len(shards)), key=lambda i: shards[i]["blocks"][0])
    keys = [k for k in CANDIDATE_KEYS if all(k in s["candidates"] for s in shards)]  # older shards lack some
    cat = {k: np.concatenate([shards[i]["candidates"][k] for i in order]) for k in keys}
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
                  "accepted": int(accepted.sum()), "accepted_mask": accepted, "weighted": weighted}


def run(card_path, n_trials, seed=None, weighted=False, **kw):
    """Run n_trials trials of the card's single channel (pass 1 and pass 2 in one process); `kw` go to
    `candidates` (batch, cap)."""
    return finish(candidates(card_path, n_trials, seed, **kw), weighted)
