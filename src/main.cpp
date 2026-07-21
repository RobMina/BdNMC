#include "BdNMC_Session.h"

#include <iostream>
#include <string>

using std::cout; using std::endl;
using std::cerr; using std::string;

int main(int argc, char* argv[]){
	string parameter_file;
	if(argc==2)
		parameter_file = argv[1];
	else
		parameter_file = "parameter.dat";

	BdNMCSession session;
	//CLI preserves historical file output behavior.
	session.set_write_files(true, true);
	int cfg = session.configure_from_card(parameter_file);
	if(cfg!=0)
		return cfg;
	return session.run();
}
