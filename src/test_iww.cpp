// test_iww.cpp — Evaluate the IWW d^2N/(dx dpT^2) on a 4D grid for comparison
// with the exact tree-level cross section computed by compare_iww_exact.py.
//
// Grid dimensions:
//   E_mu  (GeV)   : 3.0, 15.0, 60.0
//   mA    (GeV)   : 0.03, 0.30, 0.90
//   x             : 0.1, 0.3, 0.6, 0.9   (E_A'/E_mu, energy fraction)
//   theta (rad)   : 0.001, 0.01, 0.1     (polar angle of A' w.r.t. beam)
//
// At each point pT = k_abs * sin(theta) where k_abs = sqrt((x*E)^2 - mA^2).
// Points that are kinematically forbidden (x*E < mA) are skipped.
//
// Output CSV columns:
//   E_mu_GeV, mA_GeV, x, theta_rad, pT_GeV, k_abs_GeV, d2N_iww
// where d2N_iww = d^2N/(dx dpT^2) per parent lepton per target nucleus (GeV^-4)
// with epsilon=1.
//
// Target: Iron (Z=26, A=56)

#include "Lepton_Brem_Distribution.h"
#include "constants.h"

#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

int main() {
    // Grid values
    const std::vector<double> E_grid   = {3.0, 15.0, 60.0};
    const std::vector<double> mA_grid  = {0.030, 0.300, 0.900};
    const std::vector<double> x_grid   = {0.1, 0.3, 0.6, 0.9};
    const std::vector<double> th_grid  = {0.001, 0.01, 0.1};

    // Target: Iron
    const double target_Z = 26.0;
    const double target_A = 56.0;

    // epsilon=1 so d2N equals d^2sigma/(dx dpT^2) directly.
    // alpha_D is irrelevant (enters only branching ratio, not the production formula).
    const double epsilon  = 1.0;
    const double alpha_D  = 0.5;

    // Phase-space integration bounds (wide; only used to initialise the object,
    // not referenced inside d2N_lepton_brem itself).
    const double zmin = 0.0001, zmax = 0.9999;
    const double ptmin = 0.0,   ptmax = 10.0;

    std::cout << std::setprecision(6);
    std::cout << "E_mu_GeV,mA_GeV,x,theta_rad,pT_GeV,k_abs_GeV,d2N_iww\n";

    for (double mA : mA_grid) {
        // One object per mA value (form-factor params are independent of mA,
        // but mA sets the mass stored internally for threshold checks).
        Lepton_Brem_Distribution lbd(
            MASS_MUON, mA, epsilon, alpha_D,
            target_Z, target_A,
            zmin, zmax, ptmin, ptmax);

        for (double E : E_grid) {
            for (double x : x_grid) {
                double EA = x * E;
                if (EA <= mA) continue;            // below threshold
                double k_abs = std::sqrt(EA * EA - mA * mA);

                for (double theta : th_grid) {
                    double pT = k_abs * std::sin(theta);
                    double d2N = lbd.eval_d2N(x, pT, E);

                    std::cout << E       << ","
                              << mA      << ","
                              << x       << ","
                              << theta   << ","
                              << pT      << ","
                              << k_abs   << ","
                              << std::scientific << d2N
                              << std::defaultfloat << "\n";
                }
            }
        }
    }
    return 0;
}
