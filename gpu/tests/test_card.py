from pathlib import Path

import pytest

from bdnmc_jax.card import parse_card, split_string, stod

CARDS = Path(__file__).resolve().parents[1] / "phase0" / "cards"


def test_split_string_keeps_later_tokens():
    # Stock passes find(" ", sp+1) as a *length*, so the value keeps the trailing comment; stod reads the number.
    status, key, val = split_string("epsilon 1e-3 # kinetic mixing")
    assert (status, key) == (1, "epsilon")
    assert val.startswith("1e-3")
    assert stod(val) == 1e-3


def test_split_string_skip_and_malformed():
    assert split_string("")[0] == 0
    assert split_string("# comment")[0] == 0
    assert split_string("novalue")[0] == -1


@pytest.mark.parametrize("stem", ["mb_pi0_m005", "he_pi0_cuboid_m005", "mb_brem_m005", "he_brem_cuboid_m005"])
def test_phase0_cards_parse(stem):
    card = parse_card(CARDS / f"{stem}.dat")
    assert card.mass_dm == 0.005 and card.mass_v == 0.1 and card.signal_channel == "NCE_electron"
    assert [m.name for m in card.materials] == ["Carbon", "Hydrogen"]
    assert card.e_density == pytest.approx(6 * 3.63471e22 + 7.26942e22, rel=1e-15)
    assert len(card.channels) == 1
    # Material and channel blocks must not leak their keys into the top-level map.
    assert "mass" not in card.keymap and "production_distribution" not in card.keymap


def test_channel_block_fields():
    ch = parse_card(CARDS / "mb_brem_m005.dat").channels[0]
    assert ch.production_channel == "V_decay" and ch.production_distribution == "proton_brem"
    assert (ch.zmin, ch.zmax, ch.ptmax) == (0.3, 0.7, 0.2)
