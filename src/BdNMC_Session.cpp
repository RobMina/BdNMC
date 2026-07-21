#include "BdNMC_Session.h"
#include "record.h"
#include "constants.h"
#include "Random.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>

using std::cout;    using std::endl;
using std::vector;  using std::string;
using std::bind;    using std::function;
using std::list;    
using std::exception; using std::next;
using std::cerr;

namespace {
double t_delay_fraction(double tcut, double pos, double speed){
	double tdelay=pos/speed/speed_of_light-pos/speed_of_light;
	if(tdelay>tcut)
		return 1;
	else
		return 1 - (tcut-tdelay)/tcut;
}
}

BdNMCSession::BdNMCSession()
	: write_events_(false), write_summary_(false), configured_(false), ran_(false),
	  par_(0), Vnumtot_(0), timing_cut_(0), POT_(0), samplesize_(0),
	  chan_count_(0), repeat_(1) {}

BdNMCSession::~BdNMCSession(){
	delete par_;
	par_ = 0;
}

void BdNMCSession::set_write_files(bool events, bool summary){
	write_events_ = events;
	write_summary_ = summary;
}

BdNMCParticle BdNMCSession::particle_from(const Particle& p) const {
	BdNMCParticle out;
	out.name = p.name;
	out.E = p.E;
	out.px = p.px;
	out.py = p.py;
	out.pz = p.pz;
	out.m = p.m;
	for(int i=0;i<4;i++){
		out.origin[i] = p.origin_coords[i];
		out.end[i] = p.end_coords[i];
	}
	return out;
}

void BdNMCSession::record_event(int event_index, double weight, const list<Particle>& partlist){
	BdNMCEvent ev;
	ev.event_index = event_index;
	ev.weight = weight;
	for(list<Particle>::const_iterator it = partlist.begin(); it != partlist.end(); ++it){
		ev.particles.push_back(particle_from(*it));
	}
	events_.push_back(ev);
}

int BdNMCSession::configure_from_card(const std::string& parameter_file){
	using namespace std::placeholders;

	delete par_;
	par_ = 0;
	model_.reset();
	SigGen_.reset();
	DMGen_list_.clear();
	PartDist_list_.clear();
	Vnum_list_.clear();
	sig_part_vec_.clear();
	events_.clear();
	summary_ = BdNMCRunSummary();
	configured_ = false;
	ran_ = false;

	std::ifstream parstream(parameter_file);
	try{
		par_ = new Parameter(parstream);
	} catch(int e_int){
		if(e_int==-1){
			cerr << "Could not open " << parameter_file << ". Terminating run." << endl;
			parstream.close();
			return -1;
		}
		if(e_int==-2){
			cerr << "No partic/le_list_file supplied for particle_list distribution. Terminating run." << endl;
			parstream.close();
			return -1;
		}
		else{
			cerr << "Something unexpected went wrong. Terminating run." << endl;
			parstream.close();
			return -1;
		}
	}
	catch(exception &e){
		cerr << "Exception in " << e.what() << " thrown. Terminating run." << endl;
		parstream.close();
		return -1;
	}
	if(par_->Integrity()!=0){
		cerr << "Missing parameters, terminating run. Please check previous error messages." << endl;
		parstream.close();
		return -1;
	}
	cout << "Parameter read successfully\n";

	if(par_->Seed()>=0)
		Random(par_->Seed());
	else
		Random();

	timing_cut_ = par_->Timing_Cut();

	output_filename_ = par_->Output_File();
	if(output_filename_ == ""){
		output_filename_ = "../Events/events.dat";
	}
	summary_filename_ = par_->Summary_File();
	if(summary_filename_ == ""){
		summary_filename_ = "../Events/summary.dat";
	}

	sigchoice_ = par_->Signal_Channel();
	string sig_part_name = "DM";

	outmode_ = par_->Output_Mode();
	parstream.close();

	samplesize_ = par_->Sample_Size();
	repeat_ = 1;
	POT_ = par_->Protons_on_Target();

	det_ = par_->Get_Detector();
	det_int_ = bind(&detector::Ldet,det_,_1);

	double beam_energy = par_->Beam_Energy();

	std::shared_ptr<list<production_channel> > prodlist = par_->Get_Production_List();
	chan_count_ = prodlist->size();

	double max_scatter_energy = par_->Max_Scatter_Energy();
	double min_scatter_energy = par_->Min_Scatter_Energy();
	double min_angle = par_->Min_Angle();
	double max_angle = par_->Max_Angle();

	cout << "Setting up model " << par_->Model_Name() << endl;
	if(par_->Model_Name()=="Pseudoscalar_Mediator"){
		model_=std::shared_ptr<Pseudoscalar>(new Pseudoscalar(*par_));
	}
	else if(par_->Model_Name()=="Axion_Dark_Photon"){
		model_ = std::shared_ptr<Axion_Dark_Photon>(new Axion_Dark_Photon(*par_));
	}
	else if(par_->Model_Name()=="Inelastic_Dark_Matter"){
		model_ = std::shared_ptr<Inelastic_Dark_Matter>(new Inelastic_Dark_Matter(*par_));
	}
	else if(par_->Model_Name()=="Dark_Photon" or par_->Model_Name()=="Dark_Photon_DM"){
		model_ = std::shared_ptr<Kinetic_Mixing>(new Kinetic_Mixing(*par_));
	}
	else if(par_->Model_Name()=="Dark_Scalar"){
		model_ = std::shared_ptr<Scalar_Mediator>(new Scalar_Mediator(*par_));
	}
	else{
		cerr << "Unrecognized model " << par_->Model_Name() << ". Terminating run." << endl;
		return -1;
	}
	model_->Prepare_Model(*par_);

	if(par_->Output_Mode()=="particle_list"){
		configured_ = true;
		return 0;
	}

	model_->get_DMGen(DMGen_list_);
	PartDist_list_=model_->get_Distribution();
	model_->get_first_SigGen(SigGen_);
	model_->get_Vnum(Vnum_list_);
	Vnumtot_=model_->get_Vnumtot();
	sig_part_vec_ = model_->get_sig_part_vec();

	cout << "Model assigned" << endl;

	if(Vnumtot_<=0){
		cout << "No DM production expected. Terminating run.\n";
		configured_ = true;
		return 0;
	}

	SigGen_->set_angle_limits(min_angle, max_angle);
	SigGen_->set_energy_limits(min_scatter_energy, max_scatter_energy);

	cout << "--------------------" << endl;
	cout << "Run parameters:" << endl;
	cout << "--------------------" << endl;
	cout << "Number of events to be generated = " << samplesize_  << endl;
	cout << "Model Name = " << par_->Model_Name() << endl;
	model_->Report_Model();

	for(int i = 0; i<chan_count_; i++){
		cout << "Production-Channel " << i+1 << " = " << DMGen_list_[i]->Channel_Name();
		if(DMGen_list_[i]->query_off_shell()){
			cout << " in Off-Shell mode.\n";
		}
		else
			cout << " in On-Shell mode.\n";
		cout << "Production Distribution " << i+1 << " = " << PartDist_list_[i]->get_name() << endl;
		cout << "Branching Ratio "  << i+1 << " = " << DMGen_list_[i]->BranchingRatio() << endl;
		cout <<	"Production events in channel " << i+1 << " = " << Vnum_list_[i] << endl;
	}
	cout << "Signal Channel = " << sigchoice_ << endl;
	cout << "Beam Energy = " << beam_energy << " GeV" << endl;

	cout << "Maximum Scattering Energy = " << max_scatter_energy << " GeV" << endl;
	cout << "Minimum Scattering Energy = " << min_scatter_energy << " GeV" <<  endl;
	cout << "Timing Cut = " << timing_cut_ << " s\n";

	////////////////////
	//START OF BURN-IN//
	////////////////////

	double BURN_MAX = par_->Burn_In();
	if(BURN_MAX<0){
		cout << "burn_max < 0 specified. Assuming default value burn_max=1000\n";
		BURN_MAX=1000;
	}
	if(outmode_=="dm_detector_distribution"){
		BURN_MAX = 0;
		cout << "Detector_Mode selected.\nSkipping Burn-In.\n";
	}
	if(par_->Weighted_Events()==true){
		BURN_MAX = 0;
		cout << "Weighted Events selected.\nSkipping Burn-In.\n";
	}

	if(sig_part_vec_.empty()){
		sig_part_vec_.push_back(sig_part_name);
	}

	double BURN_OVERRIDE = par_->Burn_Timeout();
	if(Vnumtot_< par_->Min_Event()){
		cout << "Fewer than " << par_->Min_Event() << " events possible, skipping BURN_IN and setting pMax=0 to skip run.\n";
		BURN_MAX=0;
	}

	for(int i=0;BURN_MAX!=0&&i<chan_count_; i++){
		int nburn = 0;
		if(Vnum_list_[i]==0){
			cout << "Skipping Channel " << i+1 << ", no events expected.\n";
			continue;
		}
		else{
			cout << "Begin Channel " << i+1 << " Burn-In" << endl;
		}
		list<Particle>::iterator burniter;
		list<Particle>::iterator nextit;
		for(int burnattempt=0; (nburn < BURN_MAX)&&(burnattempt<BURN_MAX*BURN_OVERRIDE); burnattempt++){
			list<Particle> vecburn;
			Particle dist_part (0);
			PartDist_list_[i]->Sample_Particle(dist_part);
			if(DMGen_list_[i]->GenDM(vecburn, det_int_, dist_part)){
				burniter = vecburn.begin();
				for(nextit=next(burniter); burniter != vecburn.end(); burniter=nextit++){
					if(std::find(sig_part_vec_.begin(),sig_part_vec_.end(), burniter->name)!=sig_part_vec_.end()){
						SigGen_->probscatter(det_, *burniter);
						nburn++;
					}
				}
			}
		}

		if(nburn < BURN_MAX){
			cout << "Burn-In timed out, scattering number=" << nburn << " scattering goal=" << BURN_MAX << endl;
		}
		cout << "pMax at end of Burn-In = " << SigGen_->get_pMax() << endl;
	}

	configured_ = true;
	return 0;
}

int BdNMCSession::run(){
	if(!configured_){
		cerr << "BdNMCSession::run called before successful configure_from_card.\n";
		return -1;
	}
	if(par_->Output_Mode()=="particle_list"){
		return 0;
	}
	if(Vnumtot_<=0){
		return 0;
	}
	if(!SigGen_){
		cerr << "BdNMCSession::run: SigGen is null.\n";
		return -1;
	}

	events_.clear();
	summary_ = BdNMCRunSummary();
	ran_ = false;

	std::unique_ptr<std::ofstream> comprehensive_out;
	std::unique_ptr<std::ofstream> summary_out;

	if(write_events_ && (outmode_=="comprehensive"||outmode_=="dm_detector_distribution")){
		comprehensive_out = std::unique_ptr<std::ofstream>(new std::ofstream(output_filename_));
		if(!comprehensive_out->is_open()){
			cout << "Unable to open output file: " << output_filename_ << endl;
			return -1;
		}
	}

	if(write_summary_){
		summary_out = std::unique_ptr<std::ofstream>(new std::ofstream(summary_filename_,std::ios::app));
		if(!summary_out->is_open()){
			cerr << "Unable to open output file: " << summary_filename_ << endl;
			return -1;
		}
	}

	cout << "Run " << par_->Run_Name()  << " Start" << endl;

	if(write_events_ && outmode_=="comprehensive" && comprehensive_out){
		*comprehensive_out << "Run " << par_->Run_Name() << endl;
	}

	long long int trials = 0;
	vector<long> trials_list(chan_count_,0);
	vector<long double> scat_list(chan_count_,0.0);
	int nevent=0;
	vector<int> NDM_list(chan_count_,0);

	bool scatter_switch;
	int trials_max = par_->Max_Trials();

	if(trials_max>0){
		cout << "Maximum Trials set to " << trials_max << endl;
	}

	if((SigGen_->get_pMax()<=0 || SigGen_->get_pMax()*Vnumtot_<=par_->Min_Event()) && outmode_!="dm_detector_distribution" && !par_->Weighted_Events()){
		cout << "pMax less than tolerance limit, skipping remainder of run\n";
	}
	else{
		for(; (nevent < samplesize_) && ((trials < trials_max)||(trials_max<=0)); trials++){
			int i;
			scatter_switch = false;
			double vrnd = Random::Flat(0.0,1)*Vnumtot_;
			for(i=0; i<chan_count_; i++){
				if(vrnd<=Vnum_list_[i]){
					break;
				}
				else{
					vrnd-=Vnum_list_[i];
				}
			}
			trials_list[i]++;
			list<Particle> vec;
			Particle dist_part (0);
			PartDist_list_[i]->Sample_Particle(dist_part);

			list<Particle>::iterator iter;
			list<Particle>::iterator nextit;

			if(DMGen_list_[i]->GenDM(vec, det_int_, dist_part)){
				iter = vec.begin();
				for(nextit=next(iter) ; iter != vec.end();iter=nextit++){
					if(std::find(sig_part_vec_.begin(),sig_part_vec_.end(), iter->name)!=sig_part_vec_.end()){
						NDM_list[i]++;
						if(outmode_=="dm_detector_distribution"){
							double ldet = det_->Ldet(*iter);
							if(ldet <= 0)
								continue;
							if(write_events_ && comprehensive_out){
								*comprehensive_out << DMGen_list_[i]->Channel_Name() << " " << det_->Ldet(*iter) << " ";
								iter->report(*comprehensive_out);
							}
							scatter_switch=true;
							continue;
						}
						if(par_->Weighted_Events()){
							SigGen_->set_pMax(0);
						}
						if(SigGen_->probscatter(det_, vec, iter)){
							double timing_prob_factor;
							if(timing_cut_>0){
								timing_prob_factor=t_delay_fraction(timing_cut_,sqrt(pow(iter->end_coords[0],2)+pow(iter->end_coords[1],2)+pow(iter->end_coords[2],2)),iter->Speed());
							}
							else{
								timing_prob_factor=1;
							}

							if(par_->Weighted_Events()){
								scat_list[i]+=SigGen_->get_pMax()*timing_prob_factor;
							}
							else{
								scat_list[i]+=timing_prob_factor;
							}
							scatter_switch = true;
						}
						else{
							iter = vec.erase(iter);
						}
					}
				}
			}
			if(scatter_switch&&outmode_=="comprehensive"){
				++nevent;
				double weight = 0;
				if(par_->Weighted_Events()){
					weight = SigGen_->get_pMax();
				}
				record_event(nevent, weight, vec);
				if(write_events_ && comprehensive_out){
					*comprehensive_out << "event " << nevent;
					if(par_->Weighted_Events()){
						*comprehensive_out << " " << SigGen_->get_pMax();
					}
					*comprehensive_out << endl;
					Record_Particles(*comprehensive_out, vec);
					*comprehensive_out << "endevent " << nevent << endl << endl;
				}
			}
			else if(scatter_switch){
				++nevent;
			}
		}
	}

	if(write_summary_ && summary_out &&
	   (outmode_=="summary"||outmode_=="dm_detector_distribution"||outmode_=="comprehensive")){
		*summary_out << "Run " << par_->Run_Name() << endl;
	}

	vector<double> signal_list(chan_count_,0.0);
	double signal = 0.0;
	int scattot = 0;
	int NDM = 0;

	cout << chan_count_ << endl;

	for(int i=0; i<chan_count_; i++){
		if(scat_list[i]==0)
			signal_list[i]=0;
		else{
			if(par_->Weighted_Events()){
				signal_list[i] = (double)scat_list[i]/(double)trials*Vnumtot_/repeat_*par_->Efficiency();
			}
			else{
				signal_list[i] = (double)scat_list[i]/(double)trials*Vnumtot_*SigGen_->get_pMax()/repeat_*par_->Efficiency();
			}
		}
		scattot+=scat_list[i];
		cout << DMGen_list_[i]->Channel_Name() << ": " << signal_list[i];
		if(par_->Weighted_Events()){
			cout << " Weight: " << scat_list[i] << " V_num: " << Vnum_list_[i] << endl;
		}
		else{
			cout << " Events: " << scat_list[i] << " V_num: " << Vnum_list_[i] << endl;
		}

		if(write_summary_ && summary_out &&
		   (outmode_=="summary"||outmode_=="dm_detector_distribution"||outmode_=="comprehensive")){
			*summary_out << DMGen_list_[i]->Channel_Name() << " ";
			model_->Report(*summary_out);
			*summary_out << signal_list[i] << " " << sigchoice_ << " " << POT_ << " " << par_->Efficiency() << " " << samplesize_ << " " << Vnum_list_[i] << " " << Vnumtot_ << endl;
		}
		NDM+=NDM_list[i];
		signal+=signal_list[i];
	}

	if(write_summary_ && summary_out){
		if(outmode_=="summary"||outmode_=="comprehensive"){
			*summary_out << "Total ";
			model_->Report(*summary_out);
			*summary_out << signal << " " << sigchoice_ << " " << POT_ << " " << par_->Efficiency() << " " << samplesize_ << " " << endl;
		}
		else if(outmode_=="dm_detector_distribution"){
			cout << "Outputting summary" << endl;
			*summary_out << "Total ";
			model_->Report(*summary_out);
			*summary_out << signal << " " << sigchoice_ << " " << POT_ << " " << par_->Efficiency() << " " << Vnumtot_ << " " << samplesize_ << " " << (double)NDM/(2*trials) << endl;
		}
	}

	if(write_events_ && comprehensive_out && outmode_=="dm_detector_distribution"){
		cout << "Outputting summary" << endl;
		*comprehensive_out << "Total ";
		model_->Report(*comprehensive_out);
		*comprehensive_out << signal << " " << sigchoice_ << " " << POT_ << " " << par_->Efficiency() << " " << Vnumtot_ << " " << samplesize_ << " " << (double)NDM/(2*trials) << endl;
	}

	if(write_events_ && comprehensive_out && outmode_ =="comprehensive"){
		*comprehensive_out << endl << " Summary " << signal << " ";
		model_->Report(*comprehensive_out);
		*comprehensive_out << sigchoice_ << " " << POT_ << " " <<    par_->Efficiency() << " " << samplesize_ << " " << endl;
	}

	if(summary_out)
		summary_out->close();
	if(comprehensive_out)
		comprehensive_out->close();

	cout << "Number of trials = " << trials << endl;
	cout << "Number of candidates intersecting detector = " << NDM << endl;
	cout << "Number of " << sigchoice_ <<  " = " << scattot << endl;
	for(int i=0; i<chan_count_;i++)
		cout << "Number of events from channel " << i+1 << " " << DMGen_list_[i]->Channel_Name() << " = " << scat_list[i] << " in " << trials_list[i] << " trials." << endl;
	cout << "Number of sample events generated = " << nevent << endl;
	cout << "Acceptance = " << (double)NDM/(2*trials) << endl;
	if(!par_->Weighted_Events()){
		cout << "Maximum event probability = "  << SigGen_->get_pMax() << endl;
		cout << "Average event probability = "  << (double)scattot/NDM*SigGen_->get_pMax() << endl;
	}
	else{
		cout << "Average event probability = " << scattot/(double)NDM << endl;
	}
	cout << "Predicted number of signal events = "  << signal << endl;
	for(int i=0; i<chan_count_;i++)
		cout << "Predicted number of signal events from channel " << i+1 << " " << DMGen_list_[i]->Channel_Name()  <<  " = "  << signal_list[i] << endl;

	if(write_events_ && (outmode_=="comprehensive"||outmode_=="dm_detector_distribution")){
		cout << "--------------------" << endl;
		cout << "Events stored in file " << output_filename_ << endl;
		cout << "--------------------" << endl;
	}
	if(write_summary_){
		cout << "--------------------" << endl;
		cout << "Summary appended to file " << summary_filename_ << endl;
		cout << "--------------------" << endl;
	}

	summary_.run_name = par_->Run_Name();
	summary_.model_name = par_->Model_Name();
	summary_.signal_channel = sigchoice_;
	summary_.mass_dp = par_->MassDP();
	summary_.mass_dm = par_->MassDM();
	summary_.epsilon = par_->Epsilon();
	summary_.alpha_D = par_->alD();
	summary_.POT = POT_;
	summary_.efficiency = par_->Efficiency();
	summary_.Vnumtot = Vnumtot_;
	summary_.predicted_signal_total = signal;
	summary_.samplesize = samplesize_;
	summary_.nevent = nevent;
	summary_.trials = trials;
	summary_.pMax = SigGen_->get_pMax();
	summary_.NDM = NDM;
	summary_.scattot = scattot;
	summary_.channels.clear();
	for(int i=0; i<chan_count_; i++){
		BdNMCChannelSummary ch;
		ch.channel_name = DMGen_list_[i]->Channel_Name();
		ch.predicted_signal = signal_list[i];
		ch.Vnum = Vnum_list_[i];
		ch.trials = trials_list[i];
		ch.scat_count = scat_list[i];
		ch.NDM = NDM_list[i];
		summary_.channels.push_back(ch);
	}

	ran_ = true;
	return 0;
}
