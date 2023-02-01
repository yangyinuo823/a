import os
import torch
import torchani
from TANISample.sampletools import NormalModeSampler


model = torchani.models.ANI1x()
dir = os.getcwd()
infile = './DA_TS.xyz'
Nconf = 200
T = 300


# 1. use TorchANI model to calculate vib_modes
sampler = NormalModeSampler(infile, Nconf, T, ifCompMode=True, TANI_model = model)
print('1')

all_confs = sampler.normal_mode_sampling()
print('2')

# func1_1: filter out similar struct with rmsd < 0.01 and dE < 2e-3 
# default variables: rmsd_threshold=0.01, dE_threshold=2e-3, no-hydrogen=True
flt11_confs = sampler.remove_similar_struct(all_confs, rmsd_threshold=0.01, dE_threshold=2e-3)
print('3')

# func1_2: keep 80% most diverse structures by comparing aevs difference 
# default variables: aevsize=384, no-hydrogen=True
flt12_confs = sampler.get_diverse_confs(all_confs, 0.80) 
print('4')

# func2: filter out structures with qbc < 0.23 (default: qbc=0.23)
flt2_confs = sampler.QBC_check(flt12_confs, qbc=0.23) 
print('5')

# func3: write conformers to xyz files, (default: Onefile=False)
sampler.write_conformers(flt2_confs, './NMS1_onefile/', Onefile = True)
sampler.write_conformers(flt2_confs, './NMS1_multifiles/')



# 2. use vib_modes from gaussian log files
sampler = NormalModeSampler(infile, Nconf, T, ifCompMode=False, TANI_model = model, freqlogfile='./DA_TS.log')
all_confs = sampler.normal_mode_sampling()
flt1_confs = sampler.get_diverse_confs(all_confs, 0.80)
flt2_confs = sampler.QBC_check(flt1_confs, qbc=0.23)
sampler.write_conformers(flt2_confs, './NMS2_multifiles/')
