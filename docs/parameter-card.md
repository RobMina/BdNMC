# Parameter card reference

Every option that stock BdNMC reads from a parameter card, compiled from the source (`src/`, upstream master
`6cba825`). Source locations are given as `file:line` in `src/`.
- **Inferred items:** anything not confirmed by reading the code to the end is marked *(inferred)*.
- **Not inspected:** the `BdNMC_LANL` submodule (cross-section functions) was not checked for card reads.
- **GPU port:** it reads the same cards but accepts only a subset; see
  [install-and-quickstart.md](install-and-quickstart.md#what-the-port-accepts).

## Format

A card is a text file of `key value` lines:
- **Comments:** lines starting with `#` (in the first column) and blank lines are ignored.
- **Separator:** key and value are separated by the **first space**; use one space, not a tab.
- **Units:** GeV for masses, energies and momenta, metres for lengths and positions, cm^-3 for number densities,
  radians for angles, seconds for times.
- **Blocks:** most keys are top-level. Three kinds of block group the keys that follow them:
  - `material <name>`: the [material block](#material-blocks);
  - `production_channel <name>`: the [production-channel block](#production-channel-blocks), which may contain
    `dist_mod <name>` sub-blocks;
  - a block ends at the first line whose key it does not recognise, and that line is then read as top-level.
- **Pitfalls:** read [Parser behaviour and pitfalls](#parser-behaviour-and-pitfalls) before writing a card. A
  comment after a text value, an indented line or a non-integer sample size silently changes the run.

Minimal working card (the `parameter.dat` setup):

```
model Dark_Photon
dark_matter_mass 0.005
dark_photon_mass 0.1
epsilon 1e-3
alpha_D 0.1
POT 2e20
pi0_per_POT 0.9
samplesize 1000
beam_energy 8.9

production_channel pi0_decay
production_distribution pi0_sanfordwang

signal_channel NCE_electron
min_scatter_energy 0.035
max_scatter_energy 0.9
output_file Events/events.dat
summary_file Events/summary.dat

detector sphere
x-position 0.0
y-position -1.9
z-position 491.0
radius 5.0

material Carbon
number_density 3.63471e22
proton_number 6
neutron_number 6
electron_number 6
mass 11.2593
```

## Run and normalisation

"Required" means the run stops with "Missing parameters" when the key is absent (`main.cpp:115-119`).

| Key | Type | Default | Meaning | Read at |
|---|---|---|---|---|
| `POT` | real | required | protons on target; scales every channel's DM count (`Vnum`) | `Parameter.cpp:376` |
| `pi0_per_POT` | real | 0.9 | pi0 per proton on target; scales the meson channels | `Parameter.cpp:378` |
| `samplesize` | integer | required | signal events to generate (in `output_mode particle_list`, lines to write). Parsed with `stoi`: `1e5` means **1** | `Parameter.cpp:379` |
| `max_trials` | integer | -1 (no limit) | stop the main loop after this many trials; narrowed to `int` at `main.cpp:361` | `Parameter.cpp:438` |
| `min_event` | real | -1 | skip burn-in if the expected DM count is below this; skip the run if pMax x `Vnum` <= this (`main.cpp:300,367`) | `Parameter.cpp:377` |
| `burn_max` | integer | 1000 | burn-in scatters used to estimate the maximum scatter probability (pMax); negative means 1000; **0 skips burn-in, so an unweighted run is skipped** (`main.cpp:280-284,367`) | `Parameter.cpp:433` |
| `burn_timeout` | integer | 20000 | burn-in gives up after `burn_max` x `burn_timeout` attempts (`main.cpp:316`) | `Parameter.cpp:434` |
| `weighted` | `true` | false | weighted events: no burn-in, every hit kept with weight = its scatter probability (written after `event N`); the prediction becomes the sum of weights | `Parameter.cpp:395` |
| `efficiency` | real | 1.0 | multiplies the predicted signal (`main.cpp:478,481`) | `Parameter.cpp:440` |
| `seed` | integer | -1 | >= 0 seeds the generator (mt19937); < 0 seeds from the clock (`main.cpp:123-126`) | `Parameter.cpp:436` |
| `run` | text | current Unix time | run label in the event and summary files | `Parameter.cpp:397` |
| `beam_energy` | real | 8.0 | proton beam energy (GeV). Burman-Smith and the dark-scalar proton-proton cross section take it as **kinetic** energy (`BurmanSmith.cpp:38`, `Scalar_Mediator.cpp:109`); **Sanford-Wang ignores it** (see [parameter files](#distribution-parameter-files)) | `Parameter.cpp:418` |

## Model

`model` selects the physics model (`main.cpp:196-211`; default `Dark_Photon`):

| `model` | Class | Notes |
|---|---|---|
| `Dark_Photon` | `Kinetic_Mixing` | default; kinetically mixed dark photon. Baryonic variants are chosen with `*_baryonic` channel names |
| `Dark_Photon_DM` | `Kinetic_Mixing` | same as `Dark_Photon` |
| `Axion_Dark_Photon` | `Axion_Dark_Photon` | axion coupled to photon and dark photon |
| `Inelastic_Dark_Matter` | `Inelastic_Dark_Matter` | two DM states with a mass splitting |
| `Pseudoscalar_Mediator` | `Pseudoscalar` | pseudoscalar mediator |
| `Dark_Scalar` | `Scalar_Mediator` | scalar mediator mixing with the Higgs |

Any other name crashes (null model, `main.cpp:212`). `Baryonic_Dark_Matter` is declared in `Model.h` but not
implemented.

### Model parameters

The models read these keys with `Query_Map`.
- **Missing key:** `Query_Map` returns 0, and a required key stops the run ("<model> requires X to be defined").
- **Bad value:** a non-numeric value throws and terminates.
- **Generic defaults:** the four generic keys also have defaults in `Parameter.cpp:669-672` (`dark_matter_mass`
  0.01, `dark_photon_mass` 0.1, `epsilon` 1e-3, `alpha_D` 0.1). Only `Axion_Dark_Photon` ever uses those defaults;
  every other model requires the keys.

| Key | Models | Required | Meaning | Read at |
|---|---|---|---|---|
| `dark_photon_mass` | Dark_Photon, Inelastic_Dark_Matter, Axion_Dark_Photon | yes (Axion: default 0.1) | dark photon (A') mass | `Kinetic_Mixing.cpp:36`, `Inelastic_Dark_Matter.cpp:32`, `Axion_Dark_Photon.cpp:139` |
| `dark_matter_mass` | Dark_Photon, Pseudoscalar_Mediator, Inelastic_Dark_Matter | yes (IDM: this or the two below) | DM mass; for IDM sets both states to this mass | `Kinetic_Mixing.cpp:40`, `Pseudoscalar.cpp:35`, `Inelastic_Dark_Matter.cpp:40,47` |
| `dark_matter_1_mass`, `dark_matter_2_mass` | Inelastic_Dark_Matter | both, when `dark_matter_mass` is absent | the two DM masses (swapped if the second is lighter) | `Inelastic_Dark_Matter.cpp:41,49-54` |
| `alpha_D` | Dark_Photon, Inelastic_Dark_Matter | yes | dark-sector coupling alpha_D | `Kinetic_Mixing.cpp:44`, `Inelastic_Dark_Matter.cpp:55` |
| `epsilon` | Dark_Photon, Inelastic_Dark_Matter, Axion_Dark_Photon | yes (Dark_Photon: or `kappa`) | kinetic mixing (Axion: printed only) | `Kinetic_Mixing.cpp:48`, `Inelastic_Dark_Matter.cpp:36`, `Axion_Dark_Photon.cpp:140` |
| `kappa` | Dark_Photon | alternative to `epsilon` | same as `epsilon`, read only when `epsilon` is absent | `Kinetic_Mixing.cpp:48` |
| `axion_mass` | Axion_Dark_Photon | yes | axion mass | `Axion_Dark_Photon.cpp:138` |
| `gagpg` | Axion_Dark_Photon | no (0: every rate is 0) | axion-photon-dark photon coupling; the only coupling the physics uses | `Axion_Dark_Photon.cpp:142` |
| `gagg` | Axion_Dark_Photon | no (0) | axion-photon-photon coupling; reported only | `Axion_Dark_Photon.cpp:141` |
| `gagpgp` | Axion_Dark_Photon | no (0) | axion-dark photon-dark photon coupling; reported only | `Axion_Dark_Photon.cpp:143` |
| `gaee` | Axion_Dark_Photon | no (0) | axion-electron coupling; read, never used | `Axion_Dark_Photon.cpp:144` |
| `dark_mediator_mass` | Pseudoscalar_Mediator | yes | pseudoscalar mass | `Pseudoscalar.cpp:26,35` |
| `g_chi` | Pseudoscalar_Mediator | yes | DM coupling | `Pseudoscalar.cpp:23,35` |
| `g_quark` | Pseudoscalar_Mediator | yes | quark coupling | `Pseudoscalar.cpp:24,35` |
| `g_pseudoscalar_electron` | Pseudoscalar_Mediator | yes | electron coupling | `Pseudoscalar.cpp:25,35` |
| `g_neutrino` | Pseudoscalar_Mediator | no (**0**, although the class default is 1) | neutrino coupling, used only when the mediator is lighter than 2 m_DM | `Pseudoscalar.cpp:36,84-89` |
| `dark_scalar_mass` | Dark_Scalar | yes | scalar mass | `Scalar_Mediator.cpp:56` |
| `epsilon_l` | Dark_Scalar | yes | lepton coupling | `Scalar_Mediator.cpp:61` |
| `epsilon_q` | Dark_Scalar | yes | quark coupling | `Scalar_Mediator.cpp:65` |
| `epsilon_W` | Dark_Scalar | no (= `epsilon_q`) | W coupling (enters S -> gamma gamma) | `Scalar_Mediator.cpp:69-72` |

### Target

| Key | Default | Meaning | Read at |
|---|---|---|---|
| `p_num_target` | 0 | target Z: Burman-Smith (must be >= 1, otherwise it throws) and part of the BMPT mass number | `Parameter.cpp:662` |
| `n_num_target` | 0 | target neutrons per nucleus: BMPT mass number A = `p_num_target` + `n_num_target` (`Model.cpp:121`). With A = 0 the BMPT sampler probably never finishes *(inferred)* | `Parameter.cpp:661` |
| `e_num_target`, `atomic_mass_target` | 0 | parsed, never used | `Parameter.cpp:660,665` |
| `n_density_target`, `target_length`, `proton_target_cross_section` | 0 | read only by Drell-Yan code that is commented out: no effect | `Parameter.cpp:663,664,430` |

## Signal channel and cuts

| Key | Type | Default | Meaning | Read at |
|---|---|---|---|---|
| `signal_channel` | name | required | detection process; names per model below | `Parameter.cpp:386-392` |
| `min_scatter_energy` | real | 0 | lower energy cut on the signal (GeV). What it applies to depends on the channel (below) | `Parameter.cpp:420`, `main.cpp:237` |
| `max_scatter_energy` | real | 1e9 | upper energy cut (GeV) | `Parameter.cpp:419`, `main.cpp:237` |
| `min_scatter_angle` | real | 0 | lower cut on the recoil polar angle (rad); <= 0 disables it. For `Signal_Decay`: minimum opening angle of the daughters | `Parameter.cpp:423`, `main.cpp:236` |
| `max_scatter_angle` | real | 2.1 pi | upper angle cut (rad); above 2 pi it is disabled | `Parameter.cpp:424` |
| `coherent` | `true` | false | coherent scattering off nuclei for `NCE_nucleon`, `NCE_nucleon_baryonic` (Dark_Photon) and `NCE_nucleon`/`Nucleon_Scatter` (Inelastic_Dark_Matter); uses the material `mass` | `Parameter.cpp:393` |
| `kinetic_energy_cut` | `true` | false | Inelastic_Dark_Matter only: the energy cuts act on recoil kinetic rather than total energy | `Parameter.cpp:394` |
| `max_dm_energy` | real | `beam_energy` | top of the tabulated scattering cross sections (nucleon, pion-inelastic, IDM, axion channels) | `Parameter.cpp:421` |
| `dm_energy_resolution` | real | 0.1 | energy step of those tables (GeV); smaller is slower to set up and more accurate | `Parameter.cpp:422` |
| `timing_cut` | real | 0 | if > 0 (s), each event is weighted by the fraction arriving later than the cut, relative to a speed-of-light particle (`main.cpp:52-58,422-427`) | `Parameter.cpp:425` |
| `scatter_dist_filename` | path | "" | cross-section table for `Inelastic_Nucleon_Scattering[_Baryonic]` | `Parameter.cpp:426` |

`signal_channel` values:

| Model | Values (read at) |
|---|---|
| Dark_Photon | `NCE_electron`, `NCE_nucleon`, `NCE_nucleon_baryonic`, `Pion_Inelastic`, `Pion_Inelastic_Charged`, `Inelastic_Delta_to_Gamma`, `Inelastic_Nucleon_Scattering`, `Inelastic_Nucleon_Scattering_Baryonic`, `Signal_Decay` (`Kinetic_Mixing.cpp:229-336`) |
| Axion_Dark_Photon | `Electron_Scatter`, `DP_e_ep_a_Decay`, `DP_gamma_a_Decay` (`Axion_Dark_Photon.cpp:279-403`) |
| Inelastic_Dark_Matter | `Electron_Scatter` and `NCE_electron` (the same process), `NCE_nucleon` and `Nucleon_Scatter` (the same), `Pion_Inelastic`, `Pion_Inelastic_Charged`, `Inelastic_Delta_to_Gamma`, `DM2_Signal_Decay` (`Inelastic_Dark_Matter.cpp:467-697`) |
| Pseudoscalar_Mediator | `NCE_electron` (`Pseudoscalar.cpp:45`) |
| Dark_Scalar | `Scalar_Signal_Decay` (`Scalar_Mediator.cpp:129`) |

An unknown name terminates the run. `Inelastic_Nucleon_Scattering*` needs `scatter_dist_filename` and refuses
`output_mode comprehensive`. If either is violated, the setup reports an error but continues with no signal
generator, and the run then has undefined behaviour (`Kinetic_Mixing.cpp:257-265`).

What `min/max_scatter_energy` apply to (`main.cpp:237` overwrites the first signal generator's limits with the
card values):

| Signal | Cut applies to |
|---|---|
| `NCE_electron` (Dark_Photon) | **total** electron energy: `min_scatter_energy 0.035` keeps kinetic energies above 34.5 MeV, not 35 MeV |
| `NCE_nucleon*` | nucleon kinetic energy |
| `Pion_Inelastic*` | pion kinetic energy |
| Inelastic_Dark_Matter and Axion two-to-two scatters | recoil total energy, or kinetic with `kinetic_energy_cut true` |
| Pseudoscalar `NCE_electron` | recoil total energy |
| `Signal_Decay` (Dark_Photon) | each daughter's total energy; the angle cut is the opening angle |
| second-generation decays (`SignalDecay_2`) | parent energy |
| `Inelastic_Nucleon_Scattering*` | not applied |

## Detector

`detector` (required) chooses the shape: `sphere`, `cylinder` or `cuboid`, case-sensitive. The shape's keys are
all required (`parameter_keys.h:73-75`, `Parameter.cpp:463-528`); positions are the detector centre, with the beam
along +z.

| Key | sphere | cylinder | cuboid | Meaning |
|---|---|---|---|---|
| `x-position`, `y-position`, `z-position` | yes | yes | yes | centre (m) |
| `radius` | yes | yes | | radius (m) |
| `length` | | yes | yes | cylinder: length along its axis; cuboid: extent along its local z before rotation (m) |
| `width` | | | yes | cuboid: local x extent (m) |
| `height` | | | yes | cuboid: local y extent (m) |
| `det-theta` | | yes | yes | cylinder: polar angle of the axis; cuboid: second X-Y-X Euler angle (rad) |
| `det-phi` | | yes | yes | cylinder: azimuth of the axis; cuboid: first X-Y-X Euler angle (rad) |
| `det-psi` | | | yes | cuboid: third X-Y-X Euler angle (rad) |

`detector.h` lists the cuboid constructor arguments in a different order from `detector.cpp`. The parser follows
`detector.cpp`, so the keys above mean what they say.

## Material blocks

`material <name>` starts a block; any number may follow, and each adds one component of the detector material. The
detector's electron density (the `NCE_electron` target) is the sum of `electron_number` x `number_density` over
the blocks; the proton and neutron sums work the same way (`detector.cpp:18-23`).

| Key | Default | Meaning |
|---|---|---|
| `number_density` | 0 | density of this component (cm^-3; molecules or atoms, consistently with the numbers below) |
| `proton_number` | 0 | protons per unit |
| `neutron_number` | 0 | neutrons per unit |
| `electron_number` | 0 | electrons per unit |
| `mass` | **uninitialised** (not 0, despite the sample card's comment) | nucleus mass (GeV); used only for coherent scattering |

Read at `Parameter.cpp:300-309`. Put the `detector` keys before the first `material` block. With no valid
detector, the first material block crashes the program (null pointer).

## Production-channel blocks

`production_channel <name>` starts a block; at least one is required, and every block is used in card order.
Defaults are set at `Parameter.cpp:150-169`.

| Key | Type | Default | Meaning | Read at |
|---|---|---|---|---|
| `production_distribution` | name | `default` | parent kinematics (values below). **Effectively required**: the default name is rejected and the run terminates (`Model.cpp:64-67`) | `Parameter.cpp:202` |
| `meson_per_pi0` | real | 1 | mesons of this channel per pi0: the channel's count is BR x `POT` x `meson_per_pi0` x `pi0_per_POT`. Set it for every non-pi0 meson (eta, omega, ...): the default makes them as abundant as pi0. For Dark_Photon `Signal_Decay` with pi0 or eta, <= 0 means 1/30 | `Parameter.cpp:206` |
| `particle_list_file` | path | "" | input for `production_distribution particle_list`; output path in `output_mode particle_list` | `Parameter.cpp:204` |
| `particle_list_position` | `true` (any case) | false | read the origin (x y z t, columns 5-8) from the particle list | `Parameter.cpp:234-238` |
| `ptmin`, `ptmax` | real | 0, -1 | proton-bremsstrahlung transverse-momentum range (GeV) | `Parameter.cpp:224-227` |
| `zmin`, `zmax` | real | -1, -1 | bremsstrahlung momentum-fraction range | `Parameter.cpp:220-223` |
| `num_per_pot` | real | 0 | DM count per POT for `brem_dp`/`brem_axion` (Axion) and `V_decay` without a brem distribution (Inelastic_Dark_Matter) | `Parameter.cpp:232` |
| `max_meson_momentum` | real | -1 | if > 0, caps the BMPT meson momentum (GeV) | `Parameter.cpp:248` |
| `sanfordwang_file`, `distribution_parameter_file` | path | | read a [distribution parameter file](#distribution-parameter-files) into this channel (the two names are the same option) | `Parameter.cpp:239-240` |
| `dist_mod` | name | | starts a [`dist_mod` sub-block](#dist_mod-sub-blocks) | `Parameter.cpp:242-247` |
| `tolerance`, `energy_bins`, `qmin`, `proton_pdf_file_path`, `neutron_pdf_file_path` | | 0.05, 100, 1.3, "", "" | read only by the Drell-Yan generator, which is not built: no effect | `Parameter.cpp:212-229` |
| `parton_V_neutron_file`, `parton_V_proton_file` | path | "" | checked only for the `parton_V` distributions, which no longer exist: no effect | `Parameter.cpp:208-211` |

### Production channels per model

**Dark_Photon** (`Kinetic_Mixing.cpp:66-211`):

| `production_channel` | DM count |
|---|---|
| `pi0_decay`, `eta_decay`, `omega_decay`, `rho_decay`, `phi_decay` and the `_baryonic` variants (`pi0`, `eta`, `omega`, `phi`; no `rho_decay_baryonic`) | BR x `POT` x `meson_per_pi0` x `pi0_per_POT` |
| `V_decay`, `Brem_V`, `V_decay_baryonic` | bremsstrahlung rate x `POT`, **only** with `production_distribution proton_brem` (or `proton_brem_baryonic`); with any other distribution the channel silently gives 0. Needs `ptmax` >= `ptmin` and `zmax` >= `zmin` >= 0; a violation prints an error and the channel gives 0 |
| `parton_production`, `parton_production_baryonic` | always 0 (not implemented) |
| `piminus_capture` | BR x `pi0_per_POT` x `POT` |

With `signal_channel Signal_Decay`, only `pi0_decay` and `eta_decay` (on-shell A' gamma), or any channel with
`production_distribution proton_brem`, are accepted (`Kinetic_Mixing.cpp:87-131`).

What the Dark_Photon brem channel does with `ptmin`/`ptmax`/`zmin`/`zmax` (found while porting it):
- `d2N_proton_brem_to_V(z, pt2)` is called with pT in its pT^2 argument, by both the rate integral
  (`V_prod_rate`, Simpson cubature over [`zmin`, `zmax`] x [`ptmin`, `ptmax`]) and the sampler. The sampler draws
  pT in [`ptmin`, `ptmax`] and uses it as pT in the V's momentum, but the density it weights by is evaluated as if
  that value were pT^2.
- The DM count has no V -> chi chi branching ratio.
- Each event's V decays to two DM; the second is tested only if the first misses the detector, so an event has at
  most one DM in it.
- The channel name in the event file is `Dark_Bremsstrahlung`.

**Axion_Dark_Photon** (`Axion_Dark_Photon.cpp:164-275`):
- `pi0_decay`, `eta_decay`: meson -> gamma a A';
- `brem_dp`, `brem_axion`: `POT` x `num_per_pot`;
- `Reactor_Production`: `POT`, only with `DP_gamma_a_Decay`, `DP_e_ep_a_Decay` or `DP_Signal_Decay`.

**Inelastic_Dark_Matter** (`Inelastic_Dark_Matter.cpp:271-463`):
- `pi0_decay`, `eta_decay`: three-body when its branching ratio exceeds 1.8x the on-shell one, otherwise an
  on-shell A' chain;
- `V_decay`: bremsstrahlung rate with `proton_brem`, otherwise `POT` x `num_per_pot`; invalid brem limits stop the
  run.

**Pseudoscalar_Mediator** (`Pseudoscalar.cpp:63-107`): `Proton_Bremsstrahlung` only. Its `ptmax`/`z` limits are not
checked: set them.

**Dark_Scalar** (`Scalar_Mediator.cpp:89-124`): `proton_brem` only. Here it is a channel name, and it still needs
a valid `production_distribution` (e.g. `proton_beam`).

### Production distributions

The same list for every model (`Model.cpp:94-138`):

| `production_distribution` | Parents | Inputs |
|---|---|---|
| `proton_beam` | beam protons along z | `beam_energy` |
| `proton_brem` | beam protons (Dark_Photon and IDM add the bremsstrahlung distribution) | `beam_energy`, `ptmin`/`ptmax`/`zmin`/`zmax` |
| `proton_brem_baryonic` | as above, baryonic form factor | same (Dark_Photon only; IDM treats it as `proton_beam`) |
| `burmansmith` | Burman-Smith pi0 | `beam_energy` as kinetic energy, `p_num_target` >= 1 |
| `pi0_sanfordwang` | Sanford-Wang pi0 | its built-in 8.89 GeV fit, or a [parameter file](#distribution-parameter-files); **ignores `beam_energy`** |
| `k0_sanfordwang` | Sanford-Wang K0 (used for eta) | as above |
| `bmpt` | BMPT pi0 | `beam_energy`, A = `p_num_target` + `n_num_target`, `max_meson_momentum` |
| `bpmt_k0` | BMPT K0 | as `bmpt`; **spelled `bpmt_k0`** (`Model.cpp:107,112`) |
| `particle_list` | rows of `particle_list_file` | columns px py pz [E] [x y z [t]], separated by spaces, tabs or commas. E is read but ignored: it is recomputed from the channel's meson mass. The list is reused from the top when exhausted (`Particle_List.cpp:73-122`) |

### `dist_mod` sub-blocks

`dist_mod position_offset` inside a production-channel block shifts the parents' origin (`Model.cpp:150-156`). Any
other name is accepted and ignored.

| Key | Default | Meaning |
|---|---|---|
| `x-offset`, `y-offset`, `z-offset` | 0 | origin shift (m). **Parsed as integers** (`stoi`): `20.7` becomes 20, `0.5` becomes 0, `1e3` becomes 1 |
| `t-offset` | 0 | creation-time shift (s *(inferred)*), also an integer |

### Distribution parameter files

`sanfordwang_file` / `distribution_parameter_file` name a file in the same `key value` format.
- **Merging:** several files merge, and a later one wins for a repeated key.
- **Missing file:** prints "Could not open" and keeps the defaults.
- **Effect:** only `pi0_sanfordwang` and `k0_sanfordwang` use the file (`sanfordwang.cpp:20-55`; defaults in
  `sanfordwang.h:34-46`). `data/sanfordwang_parameters.dat` is an example.

| Key | Default | Meaning |
|---|---|---|
| `Beam_Energy` | 8.89 | proton momentum of the Sanford-Wang fit (GeV). This, not the top-level `beam_energy`, sets the Sanford-Wang beam |
| `maximum_pi_momentum` | 7.89 | upper meson-momentum limit of the sampler (GeV) |
| `c1` ... `c9` | 220.7, 1.08, 1.0, 1.978, 1.32, 5.572, 0.0868, 9.686, 1.0 | pi+ fit |
| `d1` ... `d9` | 213.7, 0.9379, 5.454, 1.21, 1.284, 4.781, 0.07338, 8.329, 1.0 | pi- fit (pi0 uses the pi+ and pi- fits) |
| `e1` ... `e9` | 15.13, 1.975, 4.084, 0.928, 0.731, 4.362, 0.048, 13.3, 1.278 | K0 fit |

Setting any `c`/`d` (or `e`) parameter re-runs the pi (or K0) sampler's burn-in. The proton-bremsstrahlung fit keys
`rD`, `H`, `M`, `eta1`, `eta2`, `R1pp`, `R2pp` and `Ppp` are read by functions that are never called: a parameter
file has no effect on `proton_brem`.

## Output

| Key | Default | Meaning | Read at |
|---|---|---|---|
| `output_file` | `../Events/events.dat` | event file, overwritten | `Parameter.cpp:442-447`, `main.cpp:133-135` |
| `summary_file` | `../Events/summary.dat` | one summary line per run, **appended** | `Parameter.cpp:448-453`, `main.cpp:137-139`, `268` |
| `output_mode` | `comprehensive` | below | `Parameter.cpp:396` |

The default paths are relative to the working directory, which `bin/BDNMC` sets to the repository root. So they
point to an `Events/` directory beside the repository, not the `Events/` that `bin/BDNMC` creates. Set both keys.

| `output_mode` | Behaviour |
|---|---|
| `comprehensive` | every particle of every event to `output_file`; summary line to `summary_file` |
| `summary` | no event file; summary only |
| `dm_detector_distribution` | no burn-in; one line per DM that reaches the detector (channel, path length, four-momentum); the summary adds the acceptance |
| `particle_list` | samples `samplesize` parents from the first channel's distribution, writes px py pz E x y z t to that channel's `particle_list_file` (not `output_file`), and exits |
| anything else | runs the whole simulation and writes nothing |

## Parser behaviour and pitfalls

These follow from `split_string` and the block parsers in `Parameter.cpp`; items marked *(traced)* were followed
through the code by hand, not executed.

1. **Only a `#` in the first column starts a comment.** Nothing after a value is stripped.
2. **Text values must stand alone.** The value is taken with a length equal to the index of the next space, so a
   value followed by more text keeps part of it.
   - Numbers survive (`POT 2e20 # comment` works: the number parser stops at the space).
   - **Names and paths do not**: `signal_channel NCE_electron # comment`, `detector sphere 1` or a path with a
     comment become garbage and no longer match.
   - Trailing spaces and Windows line ends are removed.
3. **Use a space, not a tab,** between key and value. A tab-only line is "improperly formatted": dropped at top
   level, and it ends a block.
4. **Do not indent.** An indented line has an empty key; inside a block it ends the block, and the indented keys are
   then read at top level and ignored. A line that is only spaces, or a key with nothing after its space, leads to
   undefined behaviour.
5. **Booleans** are true only for the exact string `true` (`True`, `1`, `yes` are false), except
   `particle_list_position`, which ignores case.
6. **Integers are parsed with `stoi`.** `samplesize 1e5` means 1 and `samplesize 2.5e4` means 2. The same holds for
   `burn_max`, `burn_timeout`, `seed`, `max_trials` and the `dist_mod` offsets. Write integers in full
   (`samplesize 100000`).
7. **Repeated keys:** the last occurrence wins, within a block too. Each `material`, `production_channel` and
   `dist_mod` adds a new entry; parameter files merge.
8. **A block ends at the first key it does not know.** Any top-level key placed between a block's lines ends it, and
   the rest of the block's keys are then read at top level and ignored. Production keys such as `meson_per_pi0` or
   `ptmax` written at top level do nothing (`parameter.dat` shows a commented `#meson_per_pi0` at top level, which
   would do nothing if uncommented).
9. **Never put an unrecognised or malformed line directly after a block header.** The parser then re-reads the header
   forever *(traced)*; for `dist_mod` it also keeps adding modifiers. A blank or comment line, or a recognised key,
   right after the header avoids this. Example of the loop: `production_channel pi0_decay` immediately followed by
   `signal_channel NCE_electron`.
10. **Declare the detector before any `material` block.** A missing or invalid detector followed by a material
    crashes inside the parser. A bad number in the cylinder or cuboid keys leaves no detector without reporting it.
11. **Model-parameter defaults apply only to `Axion_Dark_Photon`** ([Model parameters](#model-parameters)).
12. **Some errors do not stop the run.** Several checks in `Kinetic_Mixing.cpp` (lines 103, 113, 192, 259, 263)
    return `-1` from a `bool` function, which counts as success:
    - invalid bremsstrahlung limits give a channel with zero rate;
    - a dark photon heavier than the meson continues;
    - a missing `scatter_dist_filename`, or comprehensive output for `Inelastic_Nucleon_Scattering`, leaves no signal
      generator (undefined behaviour).
13. **Paths are relative to the working directory.** That is the repository root under `bin/BDNMC`, and `bin/BDNMC`
    prefixes card arguments with the caller's directory, so pass card paths relative to where you run it (an
    absolute card path breaks).
    - The sample card's commented `sanfordwang_file ../sanfordwang_parameters.dat` does not point to the shipped
      `data/sanfordwang_parameters.dat`.
14. **`burn_max 0`** with an unweighted run leaves the maximum at 0 and skips the run ("pMax less than tolerance
    limit").

## Keys with no effect

- **Parsed, never used:** `e_num_target`, `atomic_mass_target`, `repeat_count` (the code fixes the repeat count at
  1), `gaee`. `gagg` and `gagpgp` are only reported.
- **Read only by code that is not built** (Drell-Yan): `n_density_target`, `target_length`,
  `proton_target_cross_section`, `tolerance`, `energy_bins`, `qmin`, `proton_pdf_file_path`, `neutron_pdf_file_path`.
- **Checked only for distributions that no longer exist:** `parton_V_neutron_file`, `parton_V_proton_file`.
- **Read by functions that are never called:** the proton-bremsstrahlung fit keys in parameter files (`rD`, `H`,
  `M`, `eta1`, `eta2`, `R1pp`, `R2pp`, `Ppp`).
- **Accepted, no effect:**
  - `dist_mod` names other than `position_offset`;
  - `parton_production*` channels;
  - Dark_Photon `V_decay*` without a `proton_brem` distribution.

Every key that appears in `parameter.dat` and `Parameter_Card_Generator.py` is read somewhere.
