#ifndef GUARD_BdNMC_capi_h
#define GUARD_BdNMC_capi_h

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BdNMChandle BdNMChandle;

/* One meson (or other parent) row in BdNMC beam-frame units:
   momentum GeV, position meters, time seconds. */
typedef struct {
	double px, py, pz, E;
	double x, y, z, t;
} BdNMCMesonRow;

/* Structured meson NCE configure (no card file / no card string). */
typedef struct {
	double mass_dm, mass_dp, epsilon, alpha_D;
	int samplesize, burn_max, burn_timeout, seed; /* seed < 0 → default RNG */
	double POT, pi0_per_POT, meson_per_pi0, beam_energy;
	const char* production_channel; /* "pi0_decay" | "eta_decay" */
	const char* signal_channel;     /* typically "NCE_electron" */
	int particle_list_position;     /* nonzero → use x,y,z,t from rows */
	double min_scatter_energy, max_scatter_energy;
	double min_scatter_angle, max_scatter_angle;
	double efficiency;
	double det_x, det_y, det_z, width, height, length;
	double det_phi, det_theta, det_psi;
	double number_density, proton_number, neutron_number, electron_number, mass;
} BdNMCMesonConfig;

BdNMChandle* bdnmc_create(void);
void bdnmc_destroy(BdNMChandle* h);

/* write_events/write_summary: 0=off, nonzero=on. Default after create is off/off. */
void bdnmc_set_write_files(BdNMChandle* h, int write_events, int write_summary);

/* Returns 0 on success. Path-based configure (CLI / regression). */
int bdnmc_configure(BdNMChandle* h, const char* card_path);

/* Structured in-memory configure for meson particle_list production. */
int bdnmc_configure_meson(BdNMChandle* h,
	const BdNMCMesonConfig* cfg,
	const BdNMCMesonRow* rows, int nrows);

int bdnmc_run(BdNMChandle* h);

int bdnmc_nevents(const BdNMChandle* h);
int bdnmc_nparticles(const BdNMChandle* h, int event_index);

/* name_buf must hold name_buf_len bytes including null terminator.
   Returns 0 on success. */
int bdnmc_get_event_particle(const BdNMChandle* h, int event_index, int particle_index,
	char* name_buf, int name_buf_len,
	double* E, double* px, double* py, double* pz, double* m,
	double origin[4], double end[4]);

int bdnmc_nchannels(const BdNMChandle* h);

/* Returns 0 on success. name_buf must hold name_buf_len bytes. */
int bdnmc_get_channel_summary(const BdNMChandle* h, int channel_index,
	char* name_buf, int name_buf_len,
	double* predicted_signal, double* Vnum, long* trials, double* scat_count, int* NDM);

int bdnmc_get_summary(const BdNMChandle* h,
	char* run_name, int run_name_len,
	char* model_name, int model_name_len,
	char* signal_channel, int signal_channel_len,
	double* mass_dp, double* mass_dm, double* epsilon, double* alpha_D,
	double* POT, double* efficiency, double* Vnumtot, double* predicted_signal_total,
	int* samplesize, int* nevent, long long* trials, double* pMax,
	int* NDM, int* scattot);

#ifdef __cplusplus
}
#endif

#endif
