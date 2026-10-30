// V0 harness: print stock BdNMC function values, with their inputs, at full precision (%.17g).
// tests/test_v0.py reads the file and compares the JAX port function by function (gpu-bdnmc-plan.md, V0).
// Inputs are fixed formulas, not random numbers, so the file is the same on every build.
// Build and run: see v0/README.md.
#include "DMscattering.h"
#include "detector.h"
#include "branchingratios.h"
#include "sanfordwang.h"
#include "BMPT_dist.h"
#include "Particle.h"
#include "Proton_Brem_Distribution.h"
#include "constants.h"

#include <cmath>
#include <cstdio>
#include <string>

static void ray(int k, double scale, double &px, double &py, double &pz) {
    px = scale * 0.01 * std::sin(1.7 * k);
    py = scale * 0.01 * std::cos(0.9 * k) - scale * 0.004;
    pz = scale * (1.0 + 0.1 * std::sin(0.3 * k));
}

int main() {
    const double m = 0.005, mv = 0.1, kap = 1e-3, alD = 0.1;
    // Electron scattering on an (E, Ee) grid.
    for (int i = 0; i < 40; i++) {
        double E = 0.02 * std::pow(1.2, i);
        double emax = EeTMax(E, m), emin = EeTMin(E, m);
        std::printf("EeTMax %.17g %.17g\n", E, emax);
        std::printf("EeTMin %.17g %.17g\n", E, emin);
        std::printf("sigma2 %.17g %.17g %.17g %.17g\n", E, std::min(0.9, emax), std::max(0.035, emin),
                    sigma2(E, m, mv, kap, alD, std::min(0.9, emax), std::max(0.035, emin)));
        for (int j = 0; j <= 10; j++) {
            double Ee = emin + (emax - emin) * j / 10.0;
            std::printf("dsigmadEe %.17g %.17g %.17g\n", Ee, E, dsigmadEe(Ee, E, m, mv, kap, alD));
            if (Ee > MASS_ELECTRON)
                std::printf("ThetaEe %.17g %.17g %.17g\n", Ee, E, ThetaEe(Ee, E, m));
        }
    }
    // Detector path lengths on fixed rays from the origin.
    detector_sphere sph(0.0, -1.9, 491.0, 5.0);
    detector_cuboid cub(0.0, -2.0, 500.0, 3.0, 3.0, 10.0, 0.0, 0.0, 0.0);
    detector_cuboid rot(0.5, -2.0, 500.0, 3.0, 4.0, 10.0, 0.3, 0.05, -0.2);
    for (int k = 0; k < 2000; k++) {
        double px, py, pz;
        ray(k, 1.0 + 0.01 * k, px, py, pz);
        Particle dm(m);
        dm.ThreeMomentum(px, py, pz);
        std::printf("Ldet_sphere %.17g %.17g %.17g %.17g\n", px, py, pz, sph.Ldet(dm));
        std::printf("Ldet_cuboid %.17g %.17g %.17g %.17g\n", px, py, pz, cub.Ldet(dm));
        std::printf("Ldet_rotated %.17g %.17g %.17g %.17g\n", px, py, pz, rot.Ldet(dm));
    }
    // Branching ratios (the Vnum normalisation).
    for (double mx : {0.001, 0.005, 0.02, 0.045}) {
        std::printf("brpi0_to_gamma_dm_dm %.17g %.17g\n", mx, brpi0_to_gamma_dm_dm(mv, mx, kap, alD));
        std::printf("brpi0toVgamma %.17g %.17g\n", mx, brpi0toVgamma(mv, mx, kap, alD));
        std::printf("brV_to_dm_dm %.17g %.17g\n", mx, brV_to_dm_dm(mv, mx, kap, alD));
    }
    // Parent densities.
    sanfordwang sw("pi0_sanfordwang");
    BMPT bmpt(120.0, 12, PDG::PI0);
    for (int i = 1; i <= 40; i++) {
        for (int j = 1; j <= 40; j++) {
            double p = 7.89 * i / 40.0, th = (pi / 2) * j / 40.0;
            std::printf("sanfordwang %.17g %.17g %.17g\n", p, th, std::sin(th) * (sw.swpim(p, th) + sw.swpip(p, th)));
            double pb = 110.0 * i / 40.0, tb = 0.2 * std::pow(j / 40.0, 2);
            std::printf("bmpt_ics %.17g %.17g %.17g\n", pb, tb, bmpt.Invariant_Cross_Section(pb, tb));
        }
    }
    // Proton bremsstrahlung: the splitting function, d2N (called with pT in its pT^2 slot, as stock's sampler and
    // rate integral call it) and V_prod_rate for the two generic brem windows. The constructor prints its mode.
    for (int i = 0; i <= 20; i++)
        for (int j = 0; j <= 20; j++)
            for (double mA : {0.02, 0.1, 0.5}) {
                double z = 0.05 + 0.9 * i / 20.0, pt2 = 1.5 * j / 20.0;
                std::printf("wpp %.17g %.17g %.17g %.17g\n", z, pt2, mA, wpp(z, pt2, mA));
            }
    struct Window { double E, zmin, zmax, ptmax; };
    for (Window w : {Window{8.9, 0.3, 0.7, 0.2}, Window{120.0, 0.1, 0.9, 1.0}})
        for (double mA : {0.02, 0.1, 0.3}) {
            std::string mode = "proton_brem";
            Proton_Brem_Distribution pbd(w.E, kap, mA, w.ptmax, w.zmax, w.zmin, alD, mode, 0.0);
            std::printf("V_prod_rate %.17g %.17g %.17g %.17g %.17g %.17g\n", w.E, w.zmin, w.zmax, w.ptmax, mA,
                        pbd.V_prod_rate());
            for (int i = 0; i <= 10; i++)
                for (int j = 0; j <= 10; j++) {
                    double z = w.zmin + (w.zmax - w.zmin) * i / 10.0, pt = w.ptmax * j / 10.0;
                    std::printf("d2N %.17g %.17g %.17g %.17g %.17g\n", w.E, mA, z, pt,
                                pbd.d2N_proton_brem_to_V(z, pt));
                }
        }
    // eta decay: its branching ratios, and the K0 parent laws used for eta (k0_sanfordwang, bpmt_k0), on the grids
    // used for the pi0 laws above.
    for (double mx : {0.001, 0.005, 0.02, 0.045}) {
        std::printf("breta_to_gamma_dm_dm %.17g %.17g\n", mx, breta_to_gamma_dm_dm(mv, mx, kap, alD));
        std::printf("bretatoVgamma %.17g %.17g\n", mx, bretatoVgamma(mv, mx, kap, alD));
    }
    sanfordwang swk("k0_sanfordwang");
    BMPT bmptk(120.0, 12, PDG::K0);
    for (int i = 1; i <= 40; i++) {
        for (int j = 1; j <= 40; j++) {
            double p = 7.89 * i / 40.0, th = (pi / 2) * j / 40.0;
            std::printf("sanfordwang_k0 %.17g %.17g %.17g\n", p, th, std::sin(th) * swk.swK(p, th));
            double pb = 110.0 * i / 40.0, tb = 0.2 * std::pow(j / 40.0, 2);
            std::printf("bmpt_k0_ics %.17g %.17g %.17g\n", pb, tb, bmptk.Invariant_Cross_Section_Kaon(pb, tb));
        }
    }
    // off-shell decays: the off-shell branching ratio and its (s, theta) density, for V above the meson mass, V
    // below 2 m_chi, and (pi0) the V pole inside the window; 20 s x 10 theta points, open at the window's ends.
    struct OffShellPoint { const char* meson; double mv, mx; };
    for (OffShellPoint o : {OffShellPoint{"pi0", 0.2, 0.005}, OffShellPoint{"pi0", 0.05, 0.03},
                            OffShellPoint{"pi0", 0.1, 0.01}, OffShellPoint{"eta", 0.6, 0.02},
                            OffShellPoint{"eta", 0.3, 0.2}}) {
        bool eta = std::string(o.meson) == "eta";
        double mm = eta ? MASS_ETA : MASS_PION_NEUTRAL, lo = 4 * o.mx * o.mx, hi = mm * mm;
        std::printf("br_offshell_%s %.17g %.17g %.17g\n", o.meson, o.mv, o.mx,
                    eta ? breta_to_gamma_dm_dm(o.mv, o.mx, kap, alD) : brpi0_to_gamma_dm_dm(o.mv, o.mx, kap, alD));
        for (int i = 1; i <= 20; i++)
            for (int j = 1; j <= 10; j++) {
                double s = lo + (hi - lo) * i / 21.0, th = pi * j / 11.0;
                std::printf("d2br_offshell_%s %.17g %.17g %.17g %.17g %.17g\n", o.meson, o.mv, o.mx, s, th,
                            eta ? d2breta_to_gamma_dm_dm(o.mv, o.mx, kap, alD, s, th)
                                : d2brpi0_to_gamma_dm_dm(o.mv, o.mx, kap, alD, s, th));
            }
    }
    return 0;
}
