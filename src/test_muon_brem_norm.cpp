// test_muon_brem_norm.cpp — Unit check for per-POT yield normalization.
//
// Loads a one-row synthetic muon spectrum and compares V_prod_rate() against the
// hand-computed yield from muon_brem_normalization.md:
//
//   N_A'/POT = (dNμ/dE) ΔE · n · L_cm · σ(E) · convGeV2cm2
//
// Output CSV (single data row):
//   dN_per_POT_per_GeV,dE_gev,n_density_cm3,dz_m,sigma_GeV2,hand_yield,vprodrate,ratio_hand_over_vprod

#include "Lepton_Brem_Distribution.h"
#include "constants.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

int main() {
    const std::string spectrum_path = "/tmp/muon_brem_norm_test_spectrum.csv";
    const double E_mu = 20.0;
    const double mA = 0.003;
    const double epsilon = 1.0e-3;
    const double dN_dE = 2.5e-3;
    const double dE = 0.55;
    const double dz_m = 7.3;
    const double n_density = 8.46e22;
    const double target_Z = 26.0;
    const double target_A = 56.0;
    const double xmin = 0.05;
    const double xmax = 0.95;
    const double ptmin = 0.0;
    const double ptmax = 1.0;

    {
        std::ofstream out(spectrum_path);
        out << "z_m,dz_m,E_gev,dN_per_POT_per_GeV,dE_gev\n";
        out << "723.660,7.300," << E_mu << "," << dN_dE << "," << dE << "\n";
    }

    Lepton_Brem_Distribution dist(
        MASS_MUON, mA, epsilon, 0.5,
        spectrum_path,
        target_Z, target_A, n_density,
        ptmax, xmax, xmin,
        723.660, 730.960,
        0.0, 72, 72, ptmin, 3.0);

    const double sigma = dist.integrated_cross_section_for_energy(E_mu);
    const double hand_yield =
        dN_dE * dE * n_density * (dz_m * m_to_cm) * sigma * convGeV2cm2;
    const double vprod = dist.V_prod_rate();
    const double ratio = (vprod > 0.0) ? hand_yield / vprod : 0.0;

    std::cout << std::setprecision(8);
    std::cout << "dN_per_POT_per_GeV,dE_gev,n_density_cm3,dz_m,sigma_GeV2,hand_yield,vprodrate,ratio_hand_over_vprod\n";
    std::cout << dN_dE << "," << dE << "," << n_density << "," << dz_m << ","
              << std::scientific << sigma << ","
              << hand_yield << "," << vprod << "," << ratio
              << std::defaultfloat << "\n";

    return (ratio > 0.99 && ratio < 1.01) ? 0 : 1;
}
