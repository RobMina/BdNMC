"""Physical constants, copied verbatim from stock src/constants.h so that every derived number matches."""
import math

pi = math.pi

alphaEM = 1.0 / 137.035999074
G_ELEC = math.sqrt(4 * math.pi * alphaEM)
speed_of_light = 299792458.0  # m/s

convGeV2cm2 = 3.89e-28
convmcm = 100.0  # Electron_Scatter.cpp, DMscattering.cpp

MASS_PROTON = 0.938272
MASS_NEUTRON = 0.939565
MASS_ELECTRON = 0.0005109989461
MASS_MUON = 0.10576583745

MASS_PION_NEUTRAL = 0.1349766
MASS_PION_CHARGED = 0.13957018
MASS_ETA = 0.547862
MASS_KAON_CHARGED = 0.493677

brpi0to2gamma = 0.98823
bretato2gamma = 0.3941

PDG_PI0 = 111
PDG_K0 = 311
