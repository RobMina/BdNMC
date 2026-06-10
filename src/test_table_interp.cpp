#include "Lepton_Brem_Distribution.h"
#include "constants.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

double arg_double(int argc, char** argv, int idx, double def) {
    return (idx < argc) ? std::atof(argv[idx]) : def;
}

int arg_int(int argc, char** argv, int idx, int def) {
    return (idx < argc) ? std::atoi(argv[idx]) : def;
}

double percentile(std::vector<double>& values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double pos = q * (values.size() - 1);
    const int lo = static_cast<int>(std::floor(pos));
    const int hi = static_cast<int>(std::ceil(pos));
    if (lo == hi) return values[lo];
    const double frac = pos - lo;
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

double axis_value(double lo, double hi, int index, int bins, double power) {
    const double frac = static_cast<double>(index) / bins;
    return lo + (hi - lo) * std::pow(frac, power);
}

} // namespace

int main(int argc, char** argv) {
    const double E_mu = arg_double(argc, argv, 1, 20.0);
    const double mA = arg_double(argc, argv, 2, 0.150);
    const int nx = std::max(2, arg_int(argc, argv, 3, 24));
    const int npt = std::max(2, arg_int(argc, argv, 4, 24));
    const int samples = std::max(1, arg_int(argc, argv, 5, 200));
    const double xmin = arg_double(argc, argv, 6, 0.05);
    const double xmax = arg_double(argc, argv, 7, 0.95);
    const double ptmin = arg_double(argc, argv, 8, 0.0);
    const double ptmax = arg_double(argc, argv, 9, 1.0);
    const double target_Z = arg_double(argc, argv, 10, 26.0);
    const double target_A = arg_double(argc, argv, 11, 56.0);
    const double min_rel_to_grid_max = arg_double(argc, argv, 12, 0.0);
    const double pt_power = std::max(1.0, arg_double(argc, argv, 13, 3.0));

    Lepton_Brem_Distribution dist(
        MASS_MUON, mA, 1.0e-3, 0.5,
        target_Z, target_A,
        xmin, xmax, ptmin, ptmax
    );

    std::vector<double> pt_nodes(npt + 1, 0.0);
    for (int ip = 0; ip <= npt; ++ip) {
        pt_nodes[ip] = axis_value(ptmin, ptmax, ip, npt, pt_power);
    }

    std::vector<double> grid((nx + 1) * (npt + 1), 0.0);
    double grid_max = 0.0;
    for (int ix = 0; ix <= nx; ++ix) {
        const double x = xmin + (xmax - xmin) * ix / nx;
        for (int ip = 0; ip <= npt; ++ip) {
            const double pt = pt_nodes[ip];
            const double value = dist.eval_d2N(x, pt, E_mu);
            grid[ix * (npt + 1) + ip] = value;
            if (value > grid_max) grid_max = value;
        }
    }

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> ux(xmin, xmax);
    std::uniform_real_distribution<double> upt(ptmin, ptmax);

    std::vector<double> rel_errors;
    rel_errors.reserve(samples);
    double sum_sq = 0.0;
    double sum_exact = 0.0;
    double sum_abs_diff = 0.0;
    double sum_signed_diff = 0.0;
    double sum_weighted_sq = 0.0;
    double max_rel = 0.0;
    int skipped = 0;
    const double min_abs = 1.0e-300;
    const double min_value = std::max(min_abs, min_rel_to_grid_max * grid_max);

    for (int i = 0; i < samples; ++i) {
        const double x = ux(rng);
        const double pt = upt(rng);
        const double exact = dist.eval_d2N(x, pt, E_mu);
        if (std::abs(exact) <= min_value) {
            skipped++;
            continue;
        }

        double gx = (x - xmin) / (xmax - xmin) * nx;
        int ix = std::max(0, std::min(nx - 1, static_cast<int>(std::floor(gx))));
        int ip = static_cast<int>(
            std::upper_bound(pt_nodes.begin(), pt_nodes.end(), pt) - pt_nodes.begin()) - 1;
        ip = std::max(0, std::min(npt - 1, ip));
        double tx = gx - ix;
        double tp = (pt - pt_nodes[ip]) / (pt_nodes[ip + 1] - pt_nodes[ip]);

        const double f00 = grid[ix * (npt + 1) + ip];
        const double f10 = grid[(ix + 1) * (npt + 1) + ip];
        const double f01 = grid[ix * (npt + 1) + ip + 1];
        const double f11 = grid[(ix + 1) * (npt + 1) + ip + 1];
        const double interp =
            (1.0 - tx) * (1.0 - tp) * f00 +
            tx * (1.0 - tp) * f10 +
            (1.0 - tx) * tp * f01 +
            tx * tp * f11;

        const double rel = std::abs(interp - exact) / std::abs(exact);
        rel_errors.push_back(rel);
        sum_sq += rel * rel;
        sum_exact += std::abs(exact);
        sum_abs_diff += std::abs(interp - exact);
        sum_signed_diff += interp - exact;
        sum_weighted_sq += std::abs(exact) * rel * rel;
        if (rel > max_rel) max_rel = rel;
    }

    std::vector<double> copy50 = rel_errors;
    std::vector<double> copy90 = rel_errors;
    std::vector<double> copy99 = rel_errors;
    const double rms = rel_errors.empty() ? 0.0 : std::sqrt(sum_sq / rel_errors.size());
    const double weighted_abs = sum_exact > 0.0 ? sum_abs_diff / sum_exact : 0.0;
    const double weighted_rms = sum_exact > 0.0 ? std::sqrt(sum_weighted_sq / sum_exact) : 0.0;
    const double uniform_yield_bias = sum_exact > 0.0 ? sum_signed_diff / sum_exact : 0.0;

    std::cout << std::setprecision(8)
              << "E_mu,mA,nx,npt,samples,used,skipped,min_rel_to_grid_max,grid_max,"
              << "pt_power,"
              << "rms_rel,p50_rel,p90_rel,p99_rel,max_rel,"
              << "weighted_abs_rel,weighted_rms_rel,uniform_yield_bias\n"
              << E_mu << "," << mA << "," << nx << "," << npt << ","
              << samples << "," << rel_errors.size() << "," << skipped << ","
              << min_rel_to_grid_max << "," << grid_max << ","
              << pt_power << ","
              << rms << ","
              << percentile(copy50, 0.50) << ","
              << percentile(copy90, 0.90) << ","
              << percentile(copy99, 0.99) << ","
              << max_rel << ","
              << weighted_abs << ","
              << weighted_rms << ","
              << uniform_yield_bias << "\n";
    return 0;
}
