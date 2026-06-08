// Dark-photon production via lepton bremsstrahlung: l N -> l N A'
// Implements the exact Liu-Miller cross section from PRD 96, 016004 (2017).
// Reference Python: dark_brem/dark_boson_cross_sections.py -> dsigma_dx_dcos()

#include "Lepton_Brem_Distribution.h"
#include "Particle.h"
#include "Integrator.h"
#include "Random.h"
#include "constants.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <functional>

using std::cout;
using std::cerr;
using std::endl;

static const double me      = MASS_ELECTRON;
static const double AMU_GEV = 0.93149410242;   // atomic mass unit in GeV

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature helpers (anonymous namespace)
// ---------------------------------------------------------------------------
namespace {

// Nodes and weights on [-1, 1].  Uses Newton refinement starting from
// good initial guesses; converges to machine precision for n <= 200.
void gl_std(int n, std::vector<double>& nodes, std::vector<double>& weights) {
    nodes.resize(n);
    weights.resize(n);
    for (int i = 0; i < (n + 1) / 2; ++i) {
        double x = std::cos(pi * (4.0*i + 3.0) / (4.0*n + 2.0));
        double p0 = 1.0, p1 = x;
        for (int it = 0; it < 100; ++it) {
            p0 = 1.0; p1 = x;
            for (int k = 1; k < n; ++k) {
                double p2 = ((2.0*k+1.0)*x*p1 - k*p0) / (k + 1.0);
                p0 = p1; p1 = p2;
            }
            double dp = n * (p0 - x*p1) / (1.0 - x*x);
            double dx = -p1 / dp;
            x += dx;
            if (std::abs(dx) < 3e-16) break;
        }
        p0 = 1.0; p1 = x;
        for (int k = 1; k < n; ++k) {
            double p2 = ((2.0*k+1.0)*x*p1 - k*p0) / (k + 1.0);
            p0 = p1; p1 = p2;
        }
        double dp = n * (p0 - x*p1) / (1.0 - x*x);
        double w  = 2.0 / ((1.0 - x*x) * dp * dp);
        nodes[i]       = -x;
        nodes[n-1-i]   =  x;
        weights[i]     = w;
        weights[n-1-i] = w;
    }
}

// Nodes and weights on [lo, hi].
void gl_interval(int n, double lo, double hi,
                 std::vector<double>& nodes, std::vector<double>& weights) {
    gl_std(n, nodes, weights);
    double mid = 0.5*(lo+hi), half = 0.5*(hi-lo);
    for (int i = 0; i < n; ++i) {
        nodes[i]   = mid + half*nodes[i];
        weights[i] *= half;
    }
}

// Minkowski inner product with (-,+,+,+) signature: dot = a_vec·b_vec - a0*b0
// Matches Python: dot(a0, avec, b0, bvec) = sum(avec*bvec) - a0*b0
inline double dot4(double a0, const double av[3],
                   double b0, const double bv[3]) {
    return av[0]*bv[0] + av[1]*bv[1] + av[2]*bv[2] - a0*b0;
}

inline void cross3(const double a[3], const double b[3], double out[3]) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

inline double norm3(const double v[3]) {
    return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// Two unit vectors perpendicular to v_hat (Gram-Schmidt, matching Python).
void unit_perp_basis(const double vh[3], double e1[3], double e2[3]) {
    double trial[3] = {0.0, 0.0, 1.0};
    double d = vh[0]*trial[0] + vh[1]*trial[1] + vh[2]*trial[2];
    if (std::abs(d) > 0.9) { trial[0]=1.0; trial[1]=0.0; trial[2]=0.0; }
    cross3(vh, trial, e1);
    double ne = norm3(e1);
    e1[0]/=ne; e1[1]/=ne; e1[2]/=ne;
    cross3(vh, e1, e2);
    // e2 is already unit (|v_hat| = |e1| = 1, v_hat ⊥ e1)
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------
Lepton_Brem_Distribution::Lepton_Brem_Distribution(
        double parent_mass_, double mA_, double epsilon_, double alphaD_,
        const std::string& spectrum_file,
        double target_Z_, double target_A_,
        double ptmax, double zmax, double zmin,
        double z_production_start,
        double ptmin)
    : parent_mass(parent_mass_), mA(mA_), epsilon(epsilon_), alphaD(alphaD_),
      ZMIN(zmin), ZMAX(zmax), PTMIN(ptmin), PTMAX(ptmax),
      target_Z(target_Z_), target_A(target_A_),
      target_mass(target_A_ * AMU_GEV),
      vprodrate(0.0), pmax(0.0)
{
    a_screen = 111.0 * std::pow(target_Z, -1.0/3.0) / me;
    d_param  = 0.164  * std::pow(target_A, -2.0/3.0);
    set_mass(mA);
    load_spectrum(spectrum_file, z_production_start);
    calc_prod_rate();
}

// Validation constructor: no spectrum file.
Lepton_Brem_Distribution::Lepton_Brem_Distribution(
        double parent_mass_, double mA_, double epsilon_, double alphaD_,
        double target_Z_, double target_A_,
        double zmin, double zmax, double ptmin, double ptmax)
    : parent_mass(parent_mass_), mA(mA_), epsilon(epsilon_), alphaD(alphaD_),
      ZMIN(zmin), ZMAX(zmax), PTMIN(ptmin), PTMAX(ptmax),
      target_Z(target_Z_), target_A(target_A_),
      target_mass(target_A_ * AMU_GEV),
      vprodrate(0.0), pmax(0.0)
{
    a_screen = 111.0 * std::pow(target_Z, -1.0/3.0) / me;
    d_param  = 0.164  * std::pow(target_A, -2.0/3.0);
    set_mass(mA);
}

// ---------------------------------------------------------------------------
// q_limits_exact: kinematic nuclear recoil bounds, Eq. (22) of Liu-Miller.
// Returns false when kinematics are forbidden.
// ---------------------------------------------------------------------------
bool Lepton_Brem_Distribution::q_limits_exact(double E, double x, double theta,
                                               double& q_min, double& q_max) const {
    const double M   = target_mass;
    const double ml  = parent_mass;
    const double mA_ = mA;
    double e_k = x * E;
    if (e_k <= mA_) return false;

    double p_abs      = std::sqrt(E*E - ml*ml);
    double k_abs      = std::sqrt(e_k*e_k - mA_*mA_);
    double cos_th     = std::cos(theta);
    double v_abs      = std::sqrt(p_abs*p_abs + k_abs*k_abs - 2.0*p_abs*k_abs*cos_th);
    double u_tilde    = 2.0*(p_abs*k_abs*cos_th - E*e_k) + mA_*mA_;
    double energy_sum = M + E - e_k;

    double radicand   = u_tilde*u_tilde + 4.0*M*u_tilde*energy_sum + 4.0*M*M*v_abs*v_abs;
    double denominator = 2.0*(energy_sum*energy_sum - v_abs*v_abs);
    if (denominator <= 0.0 || radicand < 0.0) return false;

    double root   = std::sqrt(radicand);
    double center = v_abs * (u_tilde + 2.0*M*energy_sum);
    double qm     = (center - energy_sum*root) / denominator;
    double qp     = (center + energy_sum*root) / denominator;
    double lo = std::min(qm, qp);
    double hi = std::max(qm, qp);
    if (hi <= 0.0) return false;
    q_min = std::max(0.0, lo);
    q_max = hi;
    return true;
}

// ---------------------------------------------------------------------------
// phi_avg_a23: A23 tensor for vector boson, averaged over phi_q.
//
// For each t node, loops over phi_nodes (GL nodes on [0, 2pi]) and returns
// the arithmetic mean — matching Python's np.mean(values, axis=0).
// ---------------------------------------------------------------------------
std::vector<double> Lepton_Brem_Distribution::phi_avg_a23(
        double E, double x, double theta,
        const std::vector<double>& t_nodes,
        const std::vector<double>& phi_nodes) const {

    const int n_t   = static_cast<int>(t_nodes.size());
    const int n_phi = static_cast<int>(phi_nodes.size());
    std::vector<double> result(n_t, 0.0);

    const double M   = target_mass;
    const double ml  = parent_mass;
    const double mA_ = mA;

    double e_k   = x * E;
    double p_abs = std::sqrt(E*E - ml*ml);
    double k_abs = std::sqrt(e_k*e_k - mA_*mA_);
    double cos_th = std::cos(theta), sin_th = std::sin(theta);

    // Fixed 3-vectors (beam along +z)
    const double p_vec[3] = {0.0, 0.0, p_abs};
    const double k_vec[3] = {k_abs*sin_th, 0.0, k_abs*cos_th};
    double v_vec[3] = {k_vec[0]-p_vec[0], k_vec[1]-p_vec[1], k_vec[2]-p_vec[2]};
    double v_abs = norm3(v_vec);
    if (v_abs == 0.0) return result;

    double v_hat[3] = {v_vec[0]/v_abs, v_vec[1]/v_abs, v_vec[2]/v_abs};
    double e1[3], e2[3];
    unit_perp_basis(v_hat, e1, e2);

    // u_tilde is constant over phi and t
    const double u_tilde = 2.0*(p_abs*k_abs*cos_th - E*e_k) + mA_*mA_;

    for (int it = 0; it < n_t; ++it) {
        const double t = t_nodes[it];

        // Nuclear recoil kinematics
        double q_abs_t = std::sqrt(t + t*t/(4.0*M*M));
        double q0_t    = -t / (2.0*M);
        double e_prime = E - e_k + q0_t;          // outgoing lepton energy
        double pf_energy = M + t/(2.0*M);          // recoiling nucleus energy
        double p_big_0   = M + pf_energy;           // P_i^0 + P_f^0

        // Angle between nuclear recoil q and lepton-kinematics vector v
        double cos_qv = (q_abs_t*q_abs_t + v_abs*v_abs + ml*ml - e_prime*e_prime)
                       / (2.0*q_abs_t*v_abs);
        cos_qv = std::max(-1.0, std::min(1.0, cos_qv));
        double sin_qv = std::sqrt(std::max(0.0, 1.0 - cos_qv*cos_qv));

        double sum_a23 = 0.0;
        for (int ip = 0; ip < n_phi; ++ip) {
            double phi = phi_nodes[ip];
            double cp  = std::cos(phi), sp = std::sin(phi);

            // Nuclear recoil 3-vector: q = q_abs*(cos_qv*v_hat + sin_qv*(cp*e1+sp*e2))
            double q_vec[3];
            for (int d = 0; d < 3; ++d)
                q_vec[d] = q_abs_t*(cos_qv*v_hat[d] + sin_qv*(cp*e1[d] + sp*e2[d]));

            // Final lepton momentum: pprime = p + q - k
            double pprime_vec[3] = { p_vec[0]+q_vec[0]-k_vec[0],
                                     p_vec[1]+q_vec[1]-k_vec[1],
                                     p_vec[2]+q_vec[2]-k_vec[2] };
            double pprime0 = e_prime;

            // Nuclear current sum vector: P_big = P_i + P_f
            //   P_i = (M, 0,0,0),  P_f = (pf_energy, -q_vec)  [convention: q flows out]
            double p_big_vec[3] = {-q_vec[0], -q_vec[1], -q_vec[2]};

            // Mandelstam-like variables
            double pprime_dot_k = dot4(pprime0, pprime_vec, e_k,    k_vec);
            double s_tilde      = -2.0*pprime_dot_k + mA_*mA_;

            double pprime_dot_p = dot4(pprime0, pprime_vec, E, p_vec);
            double t2           = 2.0*pprime_dot_p + 2.0*ml*ml;

            double p2_big       = dot4(p_big_0, p_big_vec, p_big_0, p_big_vec);
            double p_dot_p      = dot4(p_big_0, p_big_vec, E,       p_vec);
            double p_dot_pprime = dot4(p_big_0, p_big_vec, pprime0, pprime_vec);

            // A23 for vector boson (Eq. 14 of Liu-Miller)
            double su = s_tilde * u_tilde;
            if (std::abs(su) < 1e-50 || std::abs(s_tilde + u_tilde) < 1e-50) continue;

            double common_plus = p2_big*t
                + 4.0*std::pow((u_tilde*p_dot_p + s_tilde*p_dot_pprime)
                               / (s_tilde + u_tilde), 2);
            double a23 =
                -2.0*(s_tilde*s_tilde + u_tilde*u_tilde)/su * p2_big
                - 8.0*t/su * (p_dot_p*p_dot_p + p_dot_pprime*p_dot_pprime
                              - 0.5*(t2 + mA_*mA_)*p2_big)
                - 2.0*(s_tilde+u_tilde)*(s_tilde+u_tilde)/(su*su)
                  * (mA_*mA_ + 2.0*ml*ml) * common_plus;

            sum_a23 += a23;
        }
        result[it] = (n_phi > 0) ? sum_a23 / n_phi : 0.0;
    }
    return result;
}

// ---------------------------------------------------------------------------
// dsigma_dx_dcos_impl: exact d^2sigma / (eps^2 dx d(cos theta)) in GeV^-2.
//
// Matches dark_boson_cross_sections.py::dsigma_dx_dcos("vector", ...).
// Uses GL quadrature in log t with n_t points and phi average with n_phi GL nodes.
// ---------------------------------------------------------------------------
double Lepton_Brem_Distribution::dsigma_dx_dcos_impl(double E, double x, double theta,
                                                       int n_t, int n_phi) const {
    const double M   = target_mass;
    const double ml  = parent_mass;
    const double mA_ = mA;

    double e_k = x * E;
    if (E <= ml || e_k <= mA_ || x >= 1.0) return 0.0;

    double p_abs  = std::sqrt(E*E - ml*ml);
    double k_abs  = std::sqrt(e_k*e_k - mA_*mA_);
    double cos_th = std::cos(theta);

    double q_min, q_max;
    if (!q_limits_exact(E, x, theta, q_min, q_max)) return 0.0;

    // t = 2M(sqrt(M^2+q^2) - M) (four-momentum transfer squared to nucleus)
    double t_min = std::max(2.0*M*(std::sqrt(M*M + q_min*q_min) - M), 1.0e-30);
    double t_max = 2.0*M*(std::sqrt(M*M + q_max*q_max) - M);
    if (t_max <= t_min || !std::isfinite(t_min + t_max)) return 0.0;

    // GL nodes in log(t) space
    std::vector<double> log_t_nodes, log_t_weights;
    gl_interval(n_t, std::log(t_min), std::log(t_max), log_t_nodes, log_t_weights);
    std::vector<double> t_nodes(n_t);
    for (int i = 0; i < n_t; ++i) t_nodes[i] = std::exp(log_t_nodes[i]);

    // GL phi nodes on [0, 2*pi] — arithmetic mean used (matches Python np.mean)
    std::vector<double> phi_nodes, phi_weights;
    gl_interval(n_phi, 0.0, 2.0*pi, phi_nodes, phi_weights);

    // Phi-averaged A23 for each t node
    std::vector<double> a_avg = phi_avg_a23(E, x, theta, t_nodes, phi_nodes);

    // |p_vec - k_vec|
    double v_abs = std::sqrt(p_abs*p_abs + k_abs*k_abs - 2.0*p_abs*k_abs*cos_th);
    if (v_abs <= 0.0) return 0.0;

    // prefactor = alpha^3 / (8 M^2) * k_abs * E / p_abs
    double prefactor = alphaEM*alphaEM*alphaEM / (8.0*M*M) * k_abs * E / p_abs;

    // Integrate F^2(t)/t * a_avg in log(t) space.
    // The formula is ∫ F^2(t)/t^2 * a_avg(t) dt.  With u = log(t):
    //   dt = t du  =>  ∫ F^2/t^2 * a_avg dt = ∫ F^2/t * a_avg du
    // So the log-space integrand is F^2(t)/t * a_avg (includes the 1/t).
    double integral = 0.0;
    for (int i = 0; i < n_t; ++i) {
        double t    = t_nodes[i];
        double a2t  = a_screen*a_screen*t;
        double gel  = a2t / (1.0 + a2t);
        double gnuc = 1.0 / (1.0 + t/d_param);
        double F2   = target_Z*target_Z * gel*gel * gnuc*gnuc;
        integral   += log_t_weights[i] * F2 * a_avg[i] / t;
    }

    return prefactor * integral / v_abs;
}

// ---------------------------------------------------------------------------
// d2N_lepton_brem: wrapper converting (x, pT) -> (x, theta) with Jacobian.
// Returns d^2N / (dx dpT^2) per parent lepton per nucleus [GeV^-4], including eps^2.
//
// Jacobian: d(cos theta) = -d(pT^2) / (2 k_abs^2 cos theta)
//   => d^2N/(dx dpT^2) = eps^2 * dsigma/(eps^2 dx d(cos theta)) / (2 k_abs^2 cos theta)
// ---------------------------------------------------------------------------
double Lepton_Brem_Distribution::d2N_lepton_brem(double x, double pT, double E_parent) const {
    double EA = x * E_parent;
    if (EA <= mA) return 0.0;
    double k_abs = std::sqrt(EA*EA - mA*mA);
    if (pT >= k_abs || k_abs <= 0.0) return 0.0;

    double sin_th  = pT / k_abs;
    double cos_th  = std::sqrt(1.0 - sin_th*sin_th);
    double theta   = std::asin(sin_th);

    double dsig = dsigma_dx_dcos_impl(E_parent, x, theta);
    if (dsig <= 0.0) return 0.0;

    // Jacobian factor: 1/(2 k_abs^2 cos_theta)
    return epsilon*epsilon * dsig / (2.0 * k_abs*k_abs * cos_th);
}

// ---------------------------------------------------------------------------
// load_spectrum
// ---------------------------------------------------------------------------
void Lepton_Brem_Distribution::load_spectrum(const std::string& path, double z_start) {
    std::ifstream infile(path);
    if (!infile.is_open())
        throw std::runtime_error("Lepton_Brem_Distribution: cannot open spectrum file: " + path);

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#' || line[0] == 'z' || line[0] == 'Z') continue;
        double zm, dzm, Egev, rate;
        {
            std::istringstream ss(line);
            char comma;
            if (!(ss >> zm >> comma >> dzm >> comma >> Egev >> comma >> rate)) {
                std::istringstream ss2(line);
                if (!(ss2 >> zm >> dzm >> Egev >> rate)) continue;
            }
        }
        bool accept = (z_start <= 0.0) || (std::abs(zm - z_start) < 0.01);
        if (accept && Egev > parent_mass && rate > 0.0)
            spectrum.push_back({zm, dzm, Egev, rate * dzm});
    }

    if (spectrum.empty())
        cerr << "Lepton_Brem_Distribution: WARNING - no spectrum rows loaded from "
             << path << " (z_start=" << z_start << ")\n";
}

// ---------------------------------------------------------------------------
// calc_prod_rate
// ---------------------------------------------------------------------------
void Lepton_Brem_Distribution::calc_prod_rate() {
    vprodrate = 0.0;
    pmax_per_row.clear();
    cum_weight.clear();

    if (spectrum.empty() || ZMIN >= ZMAX) return;

    double cumulative = 0.0;
    for (const auto& row : spectrum) {
        double E = row.E_gev;
        auto f2d = [this, E](double x, double pt) {
            return this->d2N_lepton_brem(x, pt, E);
        };
        std::function<double(double,double)> f2d_fn = f2d;

        // Coarse grid probe: find envelope maximum and check for any non-zero value.
        // This must run before the cubature to avoid the 0/0 convergence loop in
        // SimpsonCubature_adapt when the integrand is identically zero.
        double pm = 0.0;
        const int Nx = 8, Np = 8;
        for (int ix = 0; ix <= Nx; ++ix) {
            double x = ZMIN + (ZMAX - ZMIN) * ix / Nx;
            for (int ip = 0; ip <= Np; ++ip) {
                double pt = PTMIN + (PTMAX - PTMIN) * ip / Np;
                double v = d2N_lepton_brem(x, pt, E);
                if (v > pm) pm = v;
            }
        }
        pmax_per_row.push_back(pm > 0.0 ? pm * 2.0 : 1e-300);

        double integral = 0.0;
        if (PTMIN < PTMAX && pm > 0.0)
            integral = SimpsonCubature_adapt(f2d_fn, ZMIN, ZMAX, 30, PTMIN, PTMAX, 30, 0.02);

        double row_contrib = row.weight * integral;
        vprodrate += row_contrib;

        cumulative += row_contrib;
        cum_weight.push_back(cumulative);
    }

    if (vprodrate > 0.0)
        pmax = *std::max_element(pmax_per_row.begin(), pmax_per_row.end());

    cout << "Lepton_Brem_Distribution: vprodrate = " << vprodrate
         << "  (rows=" << spectrum.size() << ")" << endl;
}

// ---------------------------------------------------------------------------
// sample_particle
// ---------------------------------------------------------------------------
void Lepton_Brem_Distribution::sample_particle(Particle& part) {
    if (spectrum.empty()) return;

    // Step 1: sample row
    double u1 = Random::Flat(0.0, cum_weight.back());
    int idx = static_cast<int>(
        std::lower_bound(cum_weight.begin(), cum_weight.end(), u1) - cum_weight.begin());
    if (idx >= static_cast<int>(spectrum.size())) idx = spectrum.size() - 1;

    const SpectrumRow& row = spectrum[idx];
    double E_parent = row.E_gev;
    double z_vertex = row.z_m + Random::Flat(0.0, row.dz_m);

    // Step 2: rejection sample (x, pT)
    double row_pmax = pmax_per_row[idx];
    double x, pT, hold;
    int attempts = 0;
    while (true) {
        x  = Random::Flat(ZMIN, ZMAX);
        pT = Random::Flat(PTMIN, PTMAX);
        hold = d2N_lepton_brem(x, pT, E_parent);
        if (hold >= Random::Flat() * row_pmax) {
            if (row_pmax < hold) { pmax_per_row[idx] = hold * 2.0; row_pmax = hold * 2.0; }
            break;
        }
        if (++attempts > 1000000) break;
    }

    // Step 3: set 4-momentum
    double EA  = x * E_parent;
    double k_abs = (EA > mA) ? std::sqrt(EA*EA - mA*mA) : 0.0;
    double pz  = k_abs * std::sqrt(std::max(0.0, 1.0 - (pT/std::max(k_abs,1e-30))*(pT/std::max(k_abs,1e-30))));
    double phi = Random::Flat(0.0, 2.0*pi);
    double theta = std::atan2(pT, pz);
    double pmom = std::sqrt(pz*pz + pT*pT);

    part.Set_Mass(mA);
    part.ThreeMomentumPolar(pmom, theta, phi);
    part.Set_Origin(0.0, 0.0, z_vertex);
}
