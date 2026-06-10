// test_muon_brem_sigma.cpp — Integrated vector-boson cross section σ(E) for validation.
//
// Uses Lepton_Brem_Distribution::integrated_cross_section_for_energy(), which
// returns ε² ∫ dx ∫ dpT² dσ/(dx dpT²) over the configured phase-space box [GeV^-2].
//
// Output CSV:
//   E_mu_GeV,mA_GeV,target_Z,target_A,epsilon,xmin,xmax,ptmin,ptmax,sigma_GeV2

#include "Lepton_Brem_Distribution.h"
#include "constants.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <vector>

namespace {

double arg_double(int argc, char** argv, int idx, double def) {
    return (idx < argc) ? std::atof(argv[idx]) : def;
}

} // namespace

int main(int argc, char** argv) {
    const double epsilon = arg_double(argc, argv, 1, 1.0e-3);
    const double xmin = arg_double(argc, argv, 2, 0.05);
    const double xmax = arg_double(argc, argv, 3, 0.95);
    const double ptmin = arg_double(argc, argv, 4, 0.0);
    const double ptmax = arg_double(argc, argv, 5, 1.0);
    const double target_Z = arg_double(argc, argv, 6, 26.0);
    const double target_A = arg_double(argc, argv, 7, 56.0);

    const std::vector<double> E_grid = {3.0, 10.0, 20.0, 40.0, 60.0};
    const std::vector<double> mA_grid = {0.003, 0.012, 0.036, 0.150, 0.300, 0.900};

    std::cout << std::setprecision(8);
    std::cout << "E_mu_GeV,mA_GeV,target_Z,target_A,epsilon,xmin,xmax,ptmin,ptmax,sigma_GeV2\n";

    for (double mA : mA_grid) {
        Lepton_Brem_Distribution dist(
            MASS_MUON, mA, epsilon, 0.5,
            target_Z, target_A,
            xmin, xmax, ptmin, ptmax);

        for (double E : E_grid) {
            if (E * xmax <= mA) continue;
            const double sigma = dist.integrated_cross_section_for_energy(E);
            std::cout << E << "," << mA << "," << target_Z << "," << target_A << ","
                      << epsilon << "," << xmin << "," << xmax << ","
                      << ptmin << "," << ptmax << ","
                      << std::scientific << sigma << std::defaultfloat << "\n";
        }
    }
    return 0;
}
