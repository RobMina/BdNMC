#ifndef GUARD_BdNMC_capi_h
#define GUARD_BdNMC_capi_h

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BdNMChandle BdNMChandle;

BdNMChandle* bdnmc_create(void);
void bdnmc_destroy(BdNMChandle* h);

/* write_events/write_summary: 0=off, nonzero=on. Default after create is off/off. */
void bdnmc_set_write_files(BdNMChandle* h, int write_events, int write_summary);

/* Returns 0 on success. */
int bdnmc_configure(BdNMChandle* h, const char* card_path);
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
