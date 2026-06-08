// test_exact.cpp — Evaluate the exact Liu-Miller d^2sigma/(eps^2 dx d(cos theta))
// on a 4D grid for comparison with dark_boson_cross_sections.py::dsigma_dx_dcos().
//
// Grid:
//   E_mu  (GeV)   : 3.0, 15.0, 60.0
//   mA    (GeV)   : 0.03, 0.30, 0.90
//   x             : 0.1, 0.3, 0.6, 0.9
//   theta (rad)   : 0.001, 0.01, 0.1
//
// Output CSV columns:
//   E_mu_GeV, mA_GeV, x, theta_rad, pT_GeV, k_abs_GeV, dsigma_exact_GeV_m2
// where dsigma_exact = d^2sigma / (eps^2 dx d(cos theta)) in GeV^-2.
//
// Target: Iron (Z=26, A=56)

#include "Lepton_Brem_Distribution.h"
#include "constants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>

int main() {
    const std::vector<double> E_grid  = {3.0, 15.0, 60.0};
    const std::vector<double> mA_grid = {0.030, 0.300, 0.900};
    const std::vector<double> x_grid  = {0.1, 0.3, 0.6, 0.9};
    const std::vector<double> th_grid = {0.001, 0.01, 0.1};

    const double target_Z = 26.0;
    const double target_A = 56.0;
    const double epsilon  = 1.0;
    const double alpha_D  = 0.5;

    const double zmin = 0.0001, zmax = 0.9999;
    const double ptmin = 0.0,   ptmax = 10.0;

    std::cout << std::setprecision(6);
    std::cout << "E_mu_GeV,mA_GeV,x,theta_rad,pT_GeV,k_abs_GeV,dsigma_exact_GeV_m2\n";

    for (double mA : mA_grid) {
        Lepton_Brem_Distribution lbd(
            MASS_MUON, mA, epsilon, alpha_D,
            target_Z, target_A,
            zmin, zmax, ptmin, ptmax);

        for (double E : E_grid) {
            for (double x : x_grid) {
                double EA = x * E;
                if (EA <= mA) continue;
                double k_abs = std::sqrt(EA*EA - mA*mA);

                for (double theta : th_grid) {
                    double pT      = k_abs * std::sin(theta);
                    double dsigma  = lbd.eval_dsigma_exact(x, theta, E);

                    std::cout << E      << ","
                              << mA     << ","
                              << x      << ","
                              << theta  << ","
                              << pT     << ","
                              << k_abs  << ","
                              << std::scientific << dsigma
                              << std::defaultfloat << "\n";
                }
            }
        }
    }
    return 0;
}
