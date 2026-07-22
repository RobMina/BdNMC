#ifndef GUARD_BdNMC_Session_h
#define GUARD_BdNMC_Session_h

#include "BdNMC_Types.h"
#include "Parameter.h"
#include "Model.h"
#include "Scatter.h"
#include "DMgenerator.h"
#include "Distribution.h"
#include "detector.h"
#include "Particle.h"

#include <memory>
#include <string>
#include <vector>
#include <list>
#include <functional>

class BdNMCSession {
	public:
		BdNMCSession();
		~BdNMCSession();

		//Returns 0 on success, negative on failure.
		int configure_from_card(const std::string& path);
		/* Structured meson particle_list configure (no card I/O). */
		int configure_meson(const BdNMCMesonConfig& cfg,
			const BdNMCMesonRow* rows, int nrows);
		void set_write_files(bool events, bool summary);
		//Returns 0 on success, negative on failure.
		int run();

		const std::vector<BdNMCEvent>& events() const {return events_;}
		const BdNMCRunSummary& summary() const {return summary_;}

	private:
		void reset_state();
		int finish_configure();
		void record_event(int event_index, double weight, const std::list<Particle>& partlist);
		BdNMCParticle particle_from(const Particle& p) const;

		bool write_events_;
		bool write_summary_;
		bool configured_;
		bool ran_;

		Parameter * par_;
		std::shared_ptr<Model> model_;
		std::shared_ptr<detector> det_;
		std::shared_ptr<Scatter> SigGen_;
		std::vector<std::shared_ptr<DMGenerator> > DMGen_list_;
		std::vector<std::shared_ptr<Distribution> > PartDist_list_;
		std::vector<double> Vnum_list_;
		std::vector<std::string> sig_part_vec_;
		std::function<double(Particle&)> det_int_;

		std::string outmode_;
		std::string output_filename_;
		std::string summary_filename_;
		std::string sigchoice_;
		double Vnumtot_;
		double timing_cut_;
		double POT_;
		int samplesize_;
		int chan_count_;
		int repeat_;

		std::vector<BdNMCEvent> events_;
		BdNMCRunSummary summary_;
};

#endif
