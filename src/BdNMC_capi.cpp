#include "BdNMC_capi.h"
#include "BdNMC_Session.h"

#include <cstring>
#include <string>

struct BdNMChandle {
	BdNMCSession session;
};

static void copy_cstr(char* dst, int dst_len, const std::string& src){
	if(!dst || dst_len<=0)
		return;
	if(src.size() >= (size_t)dst_len){
		std::memcpy(dst, src.data(), (size_t)(dst_len-1));
		dst[dst_len-1] = '\0';
	}
	else{
		std::memcpy(dst, src.c_str(), src.size()+1);
	}
}

BdNMChandle* bdnmc_create(void){
	return new BdNMChandle();
}

void bdnmc_destroy(BdNMChandle* h){
	delete h;
}

void bdnmc_set_write_files(BdNMChandle* h, int write_events, int write_summary){
	if(!h) return;
	h->session.set_write_files(write_events!=0, write_summary!=0);
}

int bdnmc_configure(BdNMChandle* h, const char* card_path){
	if(!h || !card_path) return -1;
	return h->session.configure_from_card(card_path);
}

int bdnmc_configure_meson(BdNMChandle* h,
	const BdNMCMesonConfig* cfg,
	const BdNMCMesonRow* rows, int nrows){
	if(!h || !cfg || !rows || nrows <= 0) return -1;
	return h->session.configure_meson(*cfg, rows, nrows);
}

int bdnmc_run(BdNMChandle* h){
	if(!h) return -1;
	return h->session.run();
}

int bdnmc_nevents(const BdNMChandle* h){
	if(!h) return 0;
	return (int)h->session.events().size();
}

int bdnmc_nparticles(const BdNMChandle* h, int event_index){
	if(!h) return 0;
	const std::vector<BdNMCEvent>& evs = h->session.events();
	if(event_index < 0 || event_index >= (int)evs.size()) return 0;
	return (int)evs[event_index].particles.size();
}

int bdnmc_get_event_particle(const BdNMChandle* h, int event_index, int particle_index,
	char* name_buf, int name_buf_len,
	double* E, double* px, double* py, double* pz, double* m,
	double origin[4], double end[4]){
	if(!h) return -1;
	const std::vector<BdNMCEvent>& evs = h->session.events();
	if(event_index < 0 || event_index >= (int)evs.size()) return -1;
	const BdNMCEvent& ev = evs[event_index];
	if(particle_index < 0 || particle_index >= (int)ev.particles.size()) return -1;
	const BdNMCParticle& p = ev.particles[particle_index];
	copy_cstr(name_buf, name_buf_len, p.name);
	if(E) *E = p.E;
	if(px) *px = p.px;
	if(py) *py = p.py;
	if(pz) *pz = p.pz;
	if(m) *m = p.m;
	if(origin){
		for(int i=0;i<4;i++) origin[i]=p.origin[i];
	}
	if(end){
		for(int i=0;i<4;i++) end[i]=p.end[i];
	}
	return 0;
}

int bdnmc_nchannels(const BdNMChandle* h){
	if(!h) return 0;
	return (int)h->session.summary().channels.size();
}

int bdnmc_get_channel_summary(const BdNMChandle* h, int channel_index,
	char* name_buf, int name_buf_len,
	double* predicted_signal, double* Vnum, long* trials, double* scat_count, int* NDM){
	if(!h) return -1;
	const BdNMCRunSummary& s = h->session.summary();
	if(channel_index < 0 || channel_index >= (int)s.channels.size()) return -1;
	const BdNMCChannelSummary& ch = s.channels[channel_index];
	copy_cstr(name_buf, name_buf_len, ch.channel_name);
	if(predicted_signal) *predicted_signal = ch.predicted_signal;
	if(Vnum) *Vnum = ch.Vnum;
	if(trials) *trials = ch.trials;
	if(scat_count) *scat_count = (double)ch.scat_count;
	if(NDM) *NDM = ch.NDM;
	return 0;
}

int bdnmc_get_summary(const BdNMChandle* h,
	char* run_name, int run_name_len,
	char* model_name, int model_name_len,
	char* signal_channel, int signal_channel_len,
	double* mass_dp, double* mass_dm, double* epsilon, double* alpha_D,
	double* POT, double* efficiency, double* Vnumtot, double* predicted_signal_total,
	int* samplesize, int* nevent, long long* trials, double* pMax,
	int* NDM, int* scattot){
	if(!h) return -1;
	const BdNMCRunSummary& s = h->session.summary();
	copy_cstr(run_name, run_name_len, s.run_name);
	copy_cstr(model_name, model_name_len, s.model_name);
	copy_cstr(signal_channel, signal_channel_len, s.signal_channel);
	if(mass_dp) *mass_dp = s.mass_dp;
	if(mass_dm) *mass_dm = s.mass_dm;
	if(epsilon) *epsilon = s.epsilon;
	if(alpha_D) *alpha_D = s.alpha_D;
	if(POT) *POT = s.POT;
	if(efficiency) *efficiency = s.efficiency;
	if(Vnumtot) *Vnumtot = s.Vnumtot;
	if(predicted_signal_total) *predicted_signal_total = s.predicted_signal_total;
	if(samplesize) *samplesize = s.samplesize;
	if(nevent) *nevent = s.nevent;
	if(trials) *trials = s.trials;
	if(pMax) *pMax = s.pMax;
	if(NDM) *NDM = s.NDM;
	if(scattot) *scattot = s.scattot;
	return 0;
}
