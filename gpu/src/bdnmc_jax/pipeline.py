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

from . import (branching, brem, card as cardmod, decay, detector, electron, offshell, output, parents, particle_list,
               rng)
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
    # "meson": meson -> gamma V(-> chi chi); "list": the same with parents from a particle_list file;
    # "brem": proton bremsstrahlung V -> chi chi
    kind: str = "meson"
    rows: object = None  # particle_list rows (N, 7), kind "list" only
    offshell: object = None  # offshell.OffShell when the meson decays through V* (meson and list kinds)
    decay_fmax: float = 0.0  # its bar
    mv_gamma: float = 0.0  # m_V Gamma_V in its propagator


# Masses, the sampler bars, Vnum, m_V Gamma_V and the list rows are traced leaves under jit (no recompile per mass,
# goal P2); the name, the parent distribution (beam, target and the brem window; a list's path and length), the kind
# and whether the decay is off-shell are static.
jax.tree_util.register_dataclass(
    Channel, data_fields=["m_meson", "fmax", "vnum", "m_v", "m_dm", "rows", "decay_fmax", "mv_gamma"],
    meta_fields=["name", "dist", "kind", "offshell"])


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
        if ch.production_distribution not in dists + ("particle_list",):
            raise NotImplementedError(f"{ch.production_channel} with {ch.production_distribution}")
        br, off = channel_br(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
        if br <= 0:
            raise NotImplementedError(f"{ch.production_channel} closed (2 m_chi > meson mass)")
        dec = {}
        if not branching.uses_on_shell_decay(off, card.mass_v, card.mass_dm, m_meson):
            osh = offshell.OffShell("pi0" if ch.production_channel == "pi0_decay" else "eta")
            mvg = offshell.mv_gamma(card.mass_v, card.mass_dm, card.epsilon, card.alpha_d)
            dec = {"offshell": osh, "decay_fmax": osh.find_fmax(card.mass_v, card.mass_dm, mvg), "mv_gamma": mvg}
        vnum = br * card.pot * ch.meson_per_pi0 * card.pi0_per_pot  # Kinetic_Mixing.cpp:141
        if ch.production_distribution == "particle_list":
            chans.append(_list_channel(ch, parent_sampler, m_meson, vnum, card, dec))
            continue
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
        chans.append(Channel(ch.production_channel, m_meson, dist, float(fmax), vnum,
                             card.mass_v, card.mass_dm, **dec))
    if len(chans) != 1:
        raise NotImplementedError("exactly one production channel for now")
    return det, sc, chans


def _list_channel(ch, parent_sampler, m_meson, vnum, card, dec):
    """Meson decay with parents from `particle_list_file` (Model.cpp:131-146; particle_list.py)."""
    if parent_sampler != "stock":
        raise NotImplementedError("parent_sampler applies to the analytic distributions, not to particle_list")
    if "particle_list_file" not in ch.raw:
        raise ValueError("particle_list_file undefined for production_distribution particle_list")
    position = ch.raw.get("particle_list_position", "").lower() == "true"  # Parameter.cpp:239
    dist, rows = particle_list.load(ch.raw["particle_list_file"], position)
    return Channel(ch.production_channel, m_meson, dist, 0.0, vnum, card.mass_v, card.mass_dm, kind="list",
                   rows=jnp.asarray(rows), **dec)


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


def _recoil(sc, key, key_phi, dm, live, u_pos, t_in, t_out, origin=None, max_rounds=10_000):
    """Electron_Scatter::scatterevent + the bookkeeping of probscatter(partlist): recoil electron, Recoil_DM and
    the interaction point, for each candidate DM (rows of dm (k, 4); `origin` (k, 4) x, y, z, t, default zero).
    Only `live` rows (prob > 0) are sampled: stock never calls scatterevent for a zero-probability DM, whose energy
    window may be empty (then the recoil would be outside the kinematic range and ThetaEe NaN)."""
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
    # Particle::Generate_Position: u in [crossing0, crossing1], then Set_Time.
    t = t_in + (t_out - t_in) * u_pos
    mom = momentum(dm)
    speed = mom / dm[:, 0]
    time = mom * t / speed / speed_of_light
    if origin is None:  # DM origin (0,0,0,0)
        pos = (time * speed * speed_of_light)[:, None] * dm[:, 1:] / mom[:, None]
    else:
        time = time + origin[:, 3]
        pos = ((time - origin[:, 3]) * speed * speed_of_light)[:, None] * dm[:, 1:] / mom[:, None] + origin[:, :3]
    ang_ok = ((sc.min_angle <= 0) | (theta(e4) > sc.min_angle)) & ((sc.max_angle > 2 * pi) | (theta(e4) < sc.max_angle))
    return e4, r4, jnp.concatenate([pos, time[:, None]], axis=-1), ang_ok & live, done


def _meson_decay(ch, key, par, n, pinfo):
    """meson -> gamma V(*) -> gamma chi chi for n parents: on-shell DecayDM, or the (s, theta) sampler and
    DecayDM_Off_Shell (offshell.py). Returns (V, dm1, dm2, V mass or None); the off-shell sampler's diagnostics go
    into `pinfo`."""
    if ch.offshell is None:
        u = rng.uniform(key, rng.DECAY, (n, 4))
        return (*decay.decay_dm(par, ch.m_meson, ch.m_v, ch.m_dm, u[:, 0], u[:, 1], u[:, 2], u[:, 3]), None)
    x, filled, nprop, nover = rng.rejection(jax.random.fold_in(key, rng.OFFSHELL), ch.offshell.propose,
                                            ch.decay_fmax, n, n, params=(ch.m_meson, ch.m_v, ch.m_dm, ch.mv_gamma))
    pinfo |= {"decay_filled": filled, "decay_proposed": nprop, "decay_over_bar": nover}
    u = rng.uniform(key, rng.DECAY, (n, 3))
    return offshell.decay(par, ch.m_meson, ch.m_dm, x[:, 0], x[:, 1], u[:, 0], u[:, 1], u[:, 2])


def _block(det, sc, ch, e_density, seed, chan_index, block_index, n_valid, cap):
    """Pass 1 for one block. Returns per-block sums and up to `cap` candidates (fixed shape, `count` valid)."""
    n = rng.BLOCK
    key = rng.block_key(seed, chan_index, block_index)
    origin, m_med = None, None
    if ch.kind == "list":
        par, origin = particle_list.block(ch.rows, block_index, n, ch.m_meson)
        pinfo = {"filled": jnp.asarray(n), "proposed": jnp.asarray(n), "over_bar": jnp.asarray(0)}
        med, d1, d2, m_med = _meson_decay(ch, key, par, n, pinfo)
    elif ch.kind == "brem":
        med, pinfo = brem.sample_block(ch.dist, key, n, ch.fmax, ch.m_v)
        par = med  # no meson: the V is the parent of the event
        u = rng.uniform(key, rng.DECAY, (n, 2))
        d1, d2 = decay.two_body(med, ch.m_v, ch.m_dm, ch.m_dm, u[:, 0], u[:, 1])
    else:
        par, pinfo = parents.sample_block(ch.dist, key, n, ch.fmax, ch.m_meson)
        med, d1, d2, m_med = _meson_decay(ch, key, par, n, pinfo)
    valid = jnp.arange(n) < n_valid
    dm = jnp.concatenate([d1, d2])  # candidate slot = dm index * n + trial index
    origin2 = None if origin is None else jnp.concatenate([origin, origin])
    L, t_in, t_out = detector.ldet(det, jnp.zeros((2 * n, 3)) if origin is None else origin2[:, :3], dm[:, 1:])
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
                                               jax.random.fold_in(key, rng.RECOIL_PHI), cdm, live, u_pos, ct_in, ct_out,
                                               None if origin is None else origin2[safe])
    trial = safe % n
    out = {
        "count": count, "prob_sum": jnp.sum(prob), "parent_info": pinfo,
        "slot": idx, "trial": trial, "prob": prob[safe], "u_acc": u_acc, "ang_ok": ang_ok,
        "recoil_done": recoil_done, "e_finite": jnp.isfinite(e4).all(axis=-1),
        "dm": cdm, "electron": e4, "recoil_dm": r4, "vertex": vtx,
        "parent": par[trial], "mediator": med[trial], "L": L[safe],
        "crossing": jnp.stack([ct_in, ct_out], axis=-1),  # stock's Particle::crossing (entry, exit); p x t is metres
    }
    if origin is not None:
        out["origin"] = origin[trial]  # parent origin = V origin = DM origin (x, y, z, t)
    if m_med is not None:
        out["med_mass"] = m_med[trial]  # off-shell: the V* mass sqrt(s)
    return out


@partial(jax.jit, static_argnames=("det", "cap"))
def blocks_batch(det, sc, ch, e_density, seed, chan_index, block_indices, n_valid, cap):
    """Pass 1 for a batch of blocks in one device call (leading axis = block). Each block's random numbers depend
    only on (seed, channel, block), so the batch size does not change the results (goal P3). Compiled once per
    detector, parent distribution, cap and batch size; `sc` and `ch` values (masses, couplings) are traced."""
    return jax.vmap(lambda b, nv: _block(det, sc, ch, e_density, seed, chan_index, b, nv, cap))(block_indices, n_valid)


CANDIDATE_KEYS = ("block", "trial", "slot", "prob", "u_acc", "ang_ok", "dm", "electron", "recoil_dm", "vertex",
                  "parent", "mediator", "L", "crossing", "origin", "med_mass")  # origin: particle_list channels;
# med_mass: off-shell decays only


PASS2_KEYS = ("slot", "trial", "prob", "u_acc", "ang_ok")  # what pass 2 and the output order need on the host
CHECK_KEYS = ("count", "parent_info", "prob_sum", "recoil_done", "e_finite")


def _pass1(card_path, n_trials, seed, blocks, batch, cap, parent_sampler, host_keys, stats):
    """Pass 1 batch by batch, the next batch dispatched before the current one is copied and checked. Returns
    (card, seed, channel, blocks, generator); the generator yields (chunk, cap used, host dict, valid-row mask,
    device dict) per batch and adds prob_sum, n_hits, over_bar and cap to `stats`. The host dict holds the check
    fields and `host_keys` (None: every key) for the batch's real blocks."""
    card = cardmod.parse_card(card_path)
    seed = card.seed if seed is None else seed
    det, sc, (ch,) = setup(card, parent_sampler)
    nblocks = -(-n_trials // rng.BLOCK)
    blocks = range(nblocks) if blocks is None else blocks
    if blocks.start < 0 or blocks.stop > nblocks:
        raise ValueError(f"blocks {blocks} outside 0..{nblocks}")
    chunks = [list(blocks[i:i + batch]) for i in range(0, len(blocks), batch)]
    stats |= {"prob_sum": 0.0, "n_hits": 0, "over_bar": 0, "cap": cap}

    def launch(chunk, cap):
        # Pad a short last batch with n_valid = 0 blocks (no hits) so the batch shape, and the compile, stay fixed.
        bi = chunk + [chunk[-1]] * (batch - len(chunk))
        nv = [min(rng.BLOCK, n_trials - b * rng.BLOCK) for b in chunk] + [0] * (batch - len(chunk))
        return blocks_batch(det, sc, ch, card.e_density, seed, 0, jnp.asarray(bi), jnp.asarray(nv), cap)

    def gen():
        cap = stats["cap"]
        pending = launch(chunks[0], cap) if chunks else None
        for i, chunk in enumerate(chunks):
            dev, used_cap = pending, cap
            if i + 1 < len(chunks):
                pending = launch(chunks[i + 1], cap)  # dispatched before this batch is copied and checked (async)
            while np.asarray(dev["count"])[: len(chunk)].max() > used_cap:
                used_cap *= 2
                dev = launch(chunk, used_cap)
            if used_cap > cap:
                cap = stats["cap"] = used_cap
                if i + 1 < len(chunks):
                    pending = launch(chunks[i + 1], cap)
            keys = CHECK_KEYS + (CANDIDATE_KEYS if host_keys is None else host_keys)
            nb = len(chunk)
            out = jax.tree.map(lambda x: x[:nb], jax.device_get({k: dev[k] for k in keys if k in dev}))
            rows = _check_batch(ch, chunk, out, used_cap)
            for s in out["prob_sum"]:
                stats["prob_sum"] += float(s)  # block by block in block order, so the sum does not depend on the batch
            stats["n_hits"] += int(out["count"].sum())
            stats["over_bar"] += int(out["parent_info"]["over_bar"].sum())
            yield chunk, used_cap, out, rows, dev
    return card, seed, ch, blocks, gen()


def _check_batch(ch, chunk, out, cap):
    """Fail on a block whose parent or decay sampler did not fill it or exceeded its bar, whose recoil sampler ran
    out of rounds, or with a non-finite recoil electron. Returns the valid-row mask (block, row)."""
    pi_, counts = out["parent_info"], out["count"]
    rows = np.arange(cap) < counts[:, None]

    def first(bad):
        return chunk[int(np.argmax(bad))]
    if (pi_["filled"] != rng.BLOCK).any():
        raise RuntimeError(f"block {first(pi_['filled'] != rng.BLOCK)}: parent sampler ran out of rounds")
    if pi_["over_bar"].any() and not isinstance(ch.dist, parents.SanfordWang):
        raise RuntimeError(f"block {first(pi_['over_bar'] > 0)}: parent proposals above the bar")
    if "decay_filled" in pi_ and ((pi_["decay_filled"] != rng.BLOCK) | (pi_["decay_over_bar"] > 0)).any():
        raise RuntimeError(f"block {first((pi_['decay_filled'] != rng.BLOCK) | (pi_['decay_over_bar'] > 0))}: "
                           "off-shell (s, theta) sampler did not fill the block or exceeded its bar")
    if not out["recoil_done"][rows].all():
        raise RuntimeError(f"block {first(~(out['recoil_done'] | ~rows).all(axis=1))}: recoil sampler ran out of "
                           "rounds")
    # ThetaEe's acos argument can round above 1 at the recoil endpoint (stock shares this and writes NaN).
    bad_e = rows & (out["prob"] > 0) & ~out["e_finite"]
    if bad_e.any():
        raise RuntimeError(f"block {first(bad_e.any(axis=1))}: non-finite recoil electron")
    return rows


def _result(card_path, card, seed, ch, blocks, n_trials, batch, parent_sampler, stats, cat):
    trials = sum(min(rng.BLOCK, n_trials - b * rng.BLOCK) for b in blocks)
    return {"card": str(card_path), "seed": seed, "n_trials": n_trials, "blocks": [blocks.start, blocks.stop],
            "trials": trials, "n_hits": stats["n_hits"], "prob_sum": stats["prob_sum"],
            "parent_over_bar": stats["over_bar"], "vnum": ch.vnum, "efficiency": card.efficiency, "batch": batch,
            "cap": stats["cap"], "parent_sampler": parent_sampler, "candidates": cat}


def candidates(card_path, n_trials, seed=None, blocks=None, batch=8, cap=1024, parent_sampler="stock"):
    """Pass 1 over blocks `blocks` (a range; default all blocks of n_trials), `batch` blocks per device call.
    Shards over disjoint block ranges combine exactly into the full run, because each block's random numbers
    depend only on (seed, block). `cap` is the starting number of candidate rows kept per block: a batch with a
    block above it is rerun with the cap doubled (one recompile), which changes only the padding."""
    stats = {}
    card, seed, ch, blocks, batches = _pass1(card_path, n_trials, seed, blocks, batch, cap, parent_sampler, None,
                                             stats)
    parts = []
    for chunk, _, out, rows, _ in batches:
        parts.append({key: out[key][rows] for key in CANDIDATE_KEYS if key != "block" and key in out}
                     | {"block": np.repeat(np.asarray(chunk), out["count"])})
    cat = {key: np.concatenate([p[key] for p in parts]) for key in CANDIDATE_KEYS if parts and key in parts[0]}
    return _result(card_path, card, seed, ch, blocks, n_trials, batch, parent_sampler, stats, cat)


@jax.jit
def _gather_rows(out, idx):
    return {k: v.reshape((-1,) + v.shape[2:])[idx] for k, v in out.items()}


def events(card_path, n_trials, seed=None, weighted=False, batch=8, cap=1024, parent_sampler="stock",
           max_device_bytes=None):
    """Pass 1 and pass 2 for event output: the same result and accepted events as `finish(candidates(...))`, but
    the candidates' large fields stay on the device until pass 2 has chosen the accepted rows, which are then
    gathered there in output order; only they and the pass-2 fields are copied to the host. `candidates` in the
    result holds the accepted rows in output order (`output_order` True, event numbers in `event`). If the
    device-resident batches would exceed `max_device_bytes` (default: half the device memory, or 8 GB when the
    backend does not report it), the run is redone through `candidates` instead, with the same result."""
    if max_device_bytes is None:
        limit = (jax.devices()[0].memory_stats() or {}).get("bytes_limit")
        max_device_bytes = limit // 2 if limit else 8 << 30
    stats = {}
    card, seed, ch, blocks, batches = _pass1(card_path, n_trials, seed, None, batch, cap, parent_sampler,
                                             PASS2_KEYS, stats)
    small, kept, resident = [], [], 0
    for chunk, used_cap, out, rows, dev in batches:
        big = {k: v for k, v in dev.items() if k not in CHECK_KEYS + PASS2_KEYS}
        resident += sum(v.nbytes for v in big.values())
        if resident > max_device_bytes:
            del kept
            res = finish(candidates(card_path, n_trials, seed, batch=batch, cap=cap, parent_sampler=parent_sampler),
                         weighted)
            return res | {"events_path": "host"}
        small.append({k: out[k][rows] for k in PASS2_KEYS} | {"block": np.repeat(np.asarray(chunk), out["count"])})
        kept.append((big, used_cap, out["count"]))
    cat = {k: np.concatenate([s[k] for s in small]) for k in (*PASS2_KEYS, "block")} if small else {}
    res = finish(_result(card_path, card, seed, ch, blocks, n_trials, batch, parent_sampler, stats, cat), weighted)
    order, event = output.accepted_events(res)
    # output order is (block, trial, slot), so each batch's rows form one contiguous run of `order`
    ends = np.cumsum([int(c.sum()) for _, _, c in kept])
    bounds = np.searchsorted(np.searchsorted(ends, order, side="right"), np.arange(len(kept) + 1))
    gathered = []
    for (big, used_cap, counts), start, a, b in zip(kept, np.r_[0, ends[:-1]], bounds[:-1], bounds[1:]):
        sel = order[a:b] - start  # index among the batch's valid rows
        first = np.cumsum(counts) - counts  # each block's first valid row
        j = np.searchsorted(first, sel, side="right") - 1
        flat = j * used_cap + (sel - first[j])
        idx = np.concatenate([flat, np.zeros(-len(flat) % rng.BLOCK, flat.dtype)])  # few distinct lengths to compile
        g = jax.device_get(_gather_rows(big, jnp.asarray(idx)))
        gathered.append({k: v[:len(flat)] for k, v in g.items()})
    rows_out = {k: np.concatenate([g[k] for g in gathered]) for k in gathered[0]} if gathered else {}
    rows_out |= {k: cat[k][order] for k in (*PASS2_KEYS, "block")} if small else {}
    return res | {"candidates": rows_out, "accepted_mask": np.ones(len(order), bool), "output_order": True,
                  "event": event, "events_path": "device"}


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
