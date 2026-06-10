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
//   z_m, dz_m, E_gev, dN_per_POT_per_GeV [, dE_gev]
// If z_production_end > z_production_start, rows whose z interval overlaps
// [z_production_start, z_production_end) are used.  Otherwise only rows whose
// z_m matches z_production_start (within 0.01 m) are used.  If
// z_production_start <= 0 all rows are used (single-region mode).
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
                              double target_n_density,
                              double ptmax, double zmax, double zmin,
                              double z_production_start = 0.0,
                              double z_production_end = 0.0,
                              double spectrum_prune_fraction = 1.0e-8,
                              int table_x_bins = 72,
                              int table_pt_bins = 72,
                              double ptmin = 0.0,
                              double table_pt_power = 3.0);

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
    double eval_dsigma_exact(double x, double theta, double E) const {
        return dsigma_dx_dcos_impl(E, x, theta);
    }

    // Returns d^2N / (dx dpT^2) per parent lepton per target nucleus [GeV^-4]
    // Includes epsilon^2 factor.
    double eval_d2N(double x, double pT, double E) const {
        return d2N_lepton_brem(x, pT, E);
    }

    // epsilon^2 * Integral dx dpT^2 d sigma/(dx dpT^2) over the configured
    // (zmin, zmax, ptmin, ptmax) range [GeV^-2 per nucleus].
    double integrated_cross_section_for_energy(double E) const;

private:
    struct SpectrumRow {
        double z_m, dz_m, E_gev, dN_per_POT_per_GeV, dE_gev;
    };

    struct SamplingTable {
        std::vector<double> grid_values;
        std::vector<double> cell_cdf;
        double integral;
    };

    void sample_particle(Particle& part) override;
    void load_spectrum(const std::string& path, double z_start, double z_end);
    void calc_prod_rate();
    void build_sampling_table(const SpectrumRow& row, std::vector<double>& grid_values,
                              std::vector<double>& cell_cdf, double& integral) const;
    double pt_grid_value(int index) const;

    double dsigma_dx_dcos_impl(double E, double x, double theta,
                                int n_t = 48, int n_phi = 24) const;

    std::vector<double> phi_avg_a23(double E, double x, double theta,
                                    const std::vector<double>& t_nodes,
                                    const std::vector<double>& phi_nodes) const;

    bool q_limits_exact(double E, double x, double theta,
                        double& q_min, double& q_max) const;

    double d2N_lepton_brem(double x, double pT, double E_parent) const;

    double parent_mass, mA, epsilon, alphaD;
    double ZMIN, ZMAX, PTMIN, PTMAX;
    double target_Z, target_A, target_mass, target_n_density;
    double vprodrate, pmax;
    double spectrum_prune_fraction;
    double table_pt_power;
    int table_x_bins, table_pt_bins;

    double a_screen;
    double d_param;

    std::vector<SpectrumRow> spectrum;
    std::vector<double> cum_weight;
    std::vector<SamplingTable> energy_tables;
    std::vector<int> table_index_per_row;
};

#endif
