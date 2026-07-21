CXXFLAGS = -std=c++14 -Wall -pedantic -Wunused-but-set-parameter -fPIC

VPATH = ../src:header:../BdNMC_LANL/src
#:../src/cteq_pdf_reader

CXX = g++

DEPDIR = .d
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

COMPILE.cpp = $(CXX) $(DEPFLAGS) $(CXXFLAGS)
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d
OUTPUT_OPTION = -o

# Library objects (everything except main)
LIB_OBJ = Position_Distributions.o DMscattering.o Random.o Kinematics.o sanfordwang.o Particle.o Parameter.o detector.o decay.o decay_gen_kinetic_mixing.o decay_gen_baryonic.o Integrator.o record.o branchingratios.o DMNscattering.o Electron_Scatter.o Nucleon_Scatter.o Particle_List.o partonsample.o parton_V_gen.o DMNscattering_Baryonic.o BurmanSmith.o BMPT_dist.o minimization.o Proton_Brem_Distribution.o Lepton_Brem_Distribution.o V_decay_gen.o Pion_Inelastic.o Inelastic_Nucleon_Scatter.o piminus_capture_gen.o SignalDecay.o Axion_Dark_Photon.o Elastic_Scatter.o Pseudoscalar.o Model.o BeamDistribution.o Proton_Brem.o Two_to_Two_Scatter.o Three_Body_Decay_Gen.o Scatter.o Inelastic_Dark_Matter.o Kinetic_Mixing.o Scalar_Mediator.o Generic_Bremsstrahlung.o General_Decay_Generator.o BdNMC_Session.o BdNMC_capi.o

PREFIX ?= /usr/local
INSTALL_LIBDIR = $(PREFIX)/lib
INSTALL_INCDIR = $(PREFIX)/include
INSTALL_BINDIR = $(PREFIX)/bin

.PHONY: all clean install
all: libBdNMC.so main

libBdNMC.so: $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^

main: main.o libBdNMC.so
	$(CXX) $(CXXFLAGS) -o $@ main.o -L. -lBdNMC -Wl,-rpath,'$$ORIGIN'

test_iww: ../src/test_iww.cpp Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o
	$(CXX) $(CXXFLAGS) -c -o test_iww.o ../src/test_iww.cpp
	$(CXX) $(CXXFLAGS) -o test_iww test_iww.o Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o

test_exact: ../src/test_exact.cpp Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o
	$(CXX) $(CXXFLAGS) -c -o test_exact.o ../src/test_exact.cpp
	$(CXX) $(CXXFLAGS) -o test_exact test_exact.o Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o

test_table_interp: ../src/test_table_interp.cpp Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o
	$(CXX) $(CXXFLAGS) -c -o test_table_interp.o ../src/test_table_interp.cpp
	$(CXX) $(CXXFLAGS) -o test_table_interp test_table_interp.o Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o

test_muon_brem_sigma: ../src/test_muon_brem_sigma.cpp Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o
	$(CXX) $(CXXFLAGS) -c -o test_muon_brem_sigma.o ../src/test_muon_brem_sigma.cpp
	$(CXX) $(CXXFLAGS) -o test_muon_brem_sigma test_muon_brem_sigma.o Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o

test_muon_brem_norm: ../src/test_muon_brem_norm.cpp Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o
	$(CXX) $(CXXFLAGS) -c -o test_muon_brem_norm.o ../src/test_muon_brem_norm.cpp
	$(CXX) $(CXXFLAGS) -o test_muon_brem_norm test_muon_brem_norm.o Lepton_Brem_Distribution.o Particle.o Random.o Integrator.o BeamDistribution.o Kinematics.o minimization.o

%.o : %.cpp $(DEPDIR)/%.d
	$(COMPILE.cpp) -c $(OUTPUT_OPTION) $@ $<
	$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(LIB_OBJ) main.o))

install: all
	mkdir -p $(INSTALL_LIBDIR) $(INSTALL_INCDIR) $(INSTALL_BINDIR) $(PREFIX)/share
	cp -f libBdNMC.so $(INSTALL_LIBDIR)/
	cp -f ../src/BdNMC_Types.h ../src/BdNMC_Session.h ../src/BdNMC_capi.h $(INSTALL_INCDIR)/
	cp -f main $(INSTALL_BINDIR)/BDNMC
	printf 'BdNMC shared library install\nPREFIX=%s\n' '$(PREFIX)' > $(PREFIX)/share/README

.PHONY: clean
clean:
	rm -f *.o libBdNMC.so main
	rm -rf .d
