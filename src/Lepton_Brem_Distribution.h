#ifndef GUARD_Lepton_Brem_Dist_h
#define GUARD_Lepton_Brem_Dist_h

#include "Distribution.h"
#include <string>
#include <vector>

// Dark-photon (A') production via lepton bremsstrahlung: l N -> l N A'
// Implements the exact Liu-Miller differential cross section from
// PRD 96, 016004 (2017), arXiv:1705.01633, Eq. (21).
//
// The secondary lepton spectrum is read from a CSV file with columns:
//   z_m, dz_m, E_gev, dN_per_POT_per_GeV
// Only rows whose z_m matches z_production_start (within 0.01 m) are used.
// If z_production_start <= 0 all rows are used (single-region mode).
//
// V_prod_rate() returns the integrated A' yield per beam proton (POT).
// sample_particle() sets the A' 4-momentum and, if the spectrum has a
// non-trivial z column, randomly places the production vertex along z.

class Lepton_Brem_Distribution : public Distribution {
public:
    // Full constructor: loads spectrum CSV and precomputes production rate.
    Lepton_Brem_Distribution(double parent_mass, double mA, double epsilon,
                              double alphaD,
                              const std::string& spectrum_file,
                              double target_Z, double target_A,
                              double ptmax, double zmax, double zmin,
                              double z_production_start = 0.0,
                              double ptmin = 0.0);

    // Validation constructor: no spectrum file; skips rate calculation.
    // Use this to call eval_dsigma_exact / eval_d2N for cross-check comparisons.
    Lepton_Brem_Distribution(double parent_mass, double mA, double epsilon,
                              double alphaD,
                              double target_Z, double target_A,
                              double zmin, double zmax,
                              double ptmin, double ptmax);

    ~Lepton_Brem_Distribution(){};

    double V_prod_rate() const { return vprodrate; }

    // Exact differential cross section in (x, theta) space.
    // Returns d^2sigma / (eps^2 dx d(cos theta)) in GeV^-2
    // (no epsilon^2 factor — for direct comparison with Python).
    // x     = E_A' / E_parent  (energy fraction)
    // theta = polar angle of A' w.r.t. beam axis (rad)
    // E     = parent lepton energy (GeV)
    double eval_dsigma_exact(double x, double theta, double E) const {
        return dsigma_dx_dcos_impl(E, x, theta);
    }

    // Returns d^2N / (dx dpT^2) per parent lepton per target nucleus [GeV^-4]
    // Includes epsilon^2 factor.  Equivalent to eval_dsigma_exact with
    // Jacobian d(costheta)/d(pT^2) = -1/(2 k_abs^2 cos theta).
    double eval_d2N(double x, double pT, double E) const {
        return d2N_lepton_brem(x, pT, E);
    }

private:
    void sample_particle(Particle& part) override;
    void load_spectrum(const std::string& path, double z_start);
    void calc_prod_rate();

    // Exact Liu-Miller cross section without epsilon^2 [GeV^-2]
    double dsigma_dx_dcos_impl(double E, double x, double theta,
                                int n_t = 48, int n_phi = 24) const;

    // Phi-averaged A23 tensor (vector boson) for each t node.
    // Returns vector of length t_nodes.size().
    std::vector<double> phi_avg_a23(double E, double x, double theta,
                                    const std::vector<double>& t_nodes,
                                    const std::vector<double>& phi_nodes) const;

    // Kinematic recoil q limits (Eq. 22 of Liu-Miller).
    // Returns false if kinematics are forbidden.
    bool q_limits_exact(double E, double x, double theta,
                        double& q_min, double& q_max) const;

    // Wrapper: exact formula converted to (x, pT) space with epsilon^2 [GeV^-4]
    double d2N_lepton_brem(double x, double pT, double E_parent) const;

    double parent_mass, mA, epsilon, alphaD;
    double ZMIN, ZMAX, PTMIN, PTMAX;
    double target_Z, target_A, target_mass;  // target_mass = A * AMU_GEV
    double vprodrate, pmax;

    // Atomic screening length and nuclear form-factor parameter
    double a_screen;   // 111 Z^{-1/3} / m_e  (GeV^{-1})
    double d_param;    // 0.164 A^{-2/3}       (GeV^2)

    struct SpectrumRow {
        double z_m, dz_m, E_gev, weight; // weight = dN_mu/dE/dz per POT * dz_m
    };
    std::vector<SpectrumRow> spectrum;
    std::vector<double> cum_weight;    // CDF over rows for (z_vertex, E_mu) sampling
    std::vector<double> pmax_per_row;  // per-row rejection envelope in (x, pT)
};

#endif
