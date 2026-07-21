#ifndef GUARD_BdNMC_Types_h
#define GUARD_BdNMC_Types_h

#include <string>
#include <vector>

struct BdNMCParticle {
	std::string name;
	double E=0, px=0, py=0, pz=0, m=0;
	double origin[4] = {0,0,0,0};
	double end[4] = {0,0,0,0};
};

struct BdNMCEvent {
	int event_index=0;
	double weight=0;
	std::string channel_name;
	std::vector<BdNMCParticle> particles;
};

struct BdNMCChannelSummary {
	std::string channel_name;
	double predicted_signal=0;
	double Vnum=0;
	long trials=0;
	long double scat_count=0;
	int NDM=0;
};

struct BdNMCRunSummary {
	std::string run_name;
	std::string model_name;
	std::string signal_channel;
	double mass_dp=0;
	double mass_dm=0;
	double epsilon=0;
	double alpha_D=0;
	double POT=0;
	double efficiency=0;
	double Vnumtot=0;
	double predicted_signal_total=0;
	int samplesize=0;
	int nevent=0;
	long long trials=0;
	double pMax=0;
	int NDM=0;
	int scattot=0;
	std::vector<BdNMCChannelSummary> channels;
};

#endif
