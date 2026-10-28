"""Parameter-card parser with stock BdNMC's semantics (src/Parameter.cpp).

Reproduced on purpose, quirks included:
- `split_string`: the value is `substr(space+1, find(" ", space+1))`, whose second argument is a *length*, so the
  value usually keeps later tokens; numeric reads then take the leading number (`stod`/`stoi`).
- `material` and `production_channel` open blocks that consume the following lines while their keys are recognised
  (blank and comment lines are skipped); the first other line goes back to the top level.
- Top-level keys go into one map, so a repeated key keeps its last value.
"""
from dataclasses import dataclass, field
import re
from pathlib import Path

_NUM = re.compile(r"\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_INT = re.compile(r"\s*[+-]?\d+")


def stod(s):
    """C++ std::stod: parse the leading floating-point number, raise if there is none."""
    m = _NUM.match(s)
    if not m:
        raise ValueError(f"stod: no number in {s!r}")
    return float(m.group())


def stoi(s):
    m = _INT.match(s)
    if not m:
        raise ValueError(f"stoi: no integer in {s!r}")
    return int(m.group())


def split_string(line):
    """Return (status, key, val): 0 skip, -1 malformed, 1 key/value (Parameter.cpp split_string)."""
    if not line or line[0] in "#\n\r":
        return 0, None, None
    sp = line.find(" ")
    if sp < 0:
        return -1, None, None
    key = line[:sp]
    nxt = line.find(" ", sp + 1)
    val = line[sp + 1:] if nxt < 0 else line[sp + 1:sp + 1 + nxt]
    return 1, key, val.rstrip("\r\n ")


MATERIAL_KEYS = {"number_density", "proton_number", "neutron_number", "electron_number", "mass"}
CHANNEL_KEYS = {  # parse_production_channel; dist_mod opens a nested block
    "production_distribution", "particle_list_file", "meson_per_pi0", "parton_V_neutron_file",
    "parton_V_proton_file", "proton_pdf_file_path", "neutron_pdf_file_path", "tolerance", "energy_bins", "zmax",
    "zmin", "ptmax", "ptmin", "qmin", "num_per_pot", "particle_list_position", "sanfordwang_file",
    "distribution_parameter_file", "dist_mod", "max_meson_momentum", "secondary_spectrum_file",
    "z_production_start", "z_production_end", "spectrum_prune_fraction", "brem_table_x_bins", "brem_table_pt_bins",
    "brem_table_pt_power", "target_Z", "target_A", "target_n_density",
}
DIST_MOD_KEYS = {"x-offset", "y-offset", "z-offset", "t-offset"}


@dataclass
class Material:
    name: str
    number_density: float = 0.0
    proton_number: float = 0.0
    neutron_number: float = 0.0
    electron_number: float = 0.0
    mass: float = 0.0


@dataclass
class Channel:
    production_channel: str
    production_distribution: str = "default"
    meson_per_pi0: float = 1.0
    max_meson_momentum: float = -1.0
    zmin: float = -1.0
    zmax: float = -1.0
    ptmin: float = 0.0
    ptmax: float = -1.0
    raw: dict = field(default_factory=dict)  # every key the block consumed, as text
    dist_mods: list = field(default_factory=list)  # [(name, {"x-offset": int, ...})]


@dataclass
class Card:
    keymap: dict
    materials: list
    channels: list

    # Typed reads with the stock defaults (Parameter::Parameter, Set_Model_Parameters, Set_Target_Parameters).
    def get(self, key, default=None, conv=stod):
        if key in self.keymap:
            return conv(self.keymap[key])
        if default is None:
            raise KeyError(f"card has no '{key}' and stock has no default")
        return default

    @property
    def mass_dm(self): return self.get("dark_matter_mass", 0.01)
    @property
    def mass_v(self): return self.get("dark_photon_mass", 0.1)
    @property
    def epsilon(self): return self.get("epsilon", 1e-3)
    @property
    def alpha_d(self): return self.get("alpha_D", 0.1)
    @property
    def pot(self): return self.get("POT")
    @property
    def pi0_per_pot(self): return self.get("pi0_per_POT", 0.9)
    @property
    def beam_energy(self): return self.get("beam_energy", 8.0)
    @property
    def max_scatter_energy(self): return self.get("max_scatter_energy", 1e9)
    @property
    def min_scatter_energy(self): return self.get("min_scatter_energy", 0.0)
    @property
    def min_scatter_angle(self): return self.get("min_scatter_angle", 0.0)
    @property
    def max_scatter_angle(self): return self.get("max_scatter_angle", 2.1 * 3.141592653589793)
    @property
    def target_mass_number(self):
        return self.get("p_num_target", 0.0) + self.get("n_num_target", 0.0)
    @property
    def signal_channel(self): return self.keymap["signal_channel"]
    @property
    def model(self): return self.keymap.get("model", "Dark_Photon")
    @property
    def weighted(self): return self.keymap.get("weighted") == "true"
    @property
    def timing_cut(self): return self.get("timing_cut", 0.0)
    @property
    def efficiency(self): return self.get("efficiency", 1.0)
    @property
    def seed(self): return self.get("seed", -1, stoi)
    @property
    def max_trials(self): return self.get("max_trials", -1, stoi)
    @property
    def samplesize(self): return self.get("samplesize", conv=stoi)

    @property
    def e_density(self):
        """detector::ENtot(): sum of electron_number x number_density over materials (per cm^3)."""
        return sum(m.electron_number * m.number_density for m in self.materials)


def parse_card(path):
    lines = Path(path).read_text().split("\n")
    keymap, materials, channels = {}, [], []
    i = 0
    while i < len(lines):
        status, key, val = split_string(lines[i])
        i += 1
        if status != 1:
            continue
        if key == "material":
            mat = Material(val)
            i = _block(lines, i, MATERIAL_KEYS, lambda k, v: setattr(mat, k, stod(v)))
            materials.append(mat)
        elif key == "production_channel":
            channels.append(_channel(lines, i, val))
            i = channels[-1].raw.pop("__next__")
        else:
            keymap[key] = val
    return Card(keymap, materials, channels)


def _block(lines, i, keys, store):
    """Consume lines while they are blank/comment or carry a key in `keys`; return the first other line's index."""
    while i < len(lines):
        status, key, val = split_string(lines[i])
        if status == 0:
            i += 1
            continue
        if status != 1 or key not in keys:
            return i
        store(key, val)
        i += 1
    return i


def _channel(lines, i, name):
    ch = Channel(name)
    numeric = {"meson_per_pi0", "max_meson_momentum", "zmin", "zmax", "ptmin", "ptmax"}
    while i < len(lines):
        status, key, val = split_string(lines[i])
        if status == 0:
            i += 1
            continue
        if status != 1 or key not in CHANNEL_KEYS:
            break
        i += 1
        if key == "dist_mod":
            offsets = {}
            i = _block(lines, i, DIST_MOD_KEYS, lambda k, v: offsets.__setitem__(k, stoi(v)))
            ch.dist_mods.append((val, offsets))
            continue
        ch.raw[key] = val
        if key == "production_distribution":
            ch.production_distribution = val
        elif key in numeric:
            setattr(ch, key, stod(val))
    ch.raw["__next__"] = i
    return ch
