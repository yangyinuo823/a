
import torch
import torchani
import numpy as np
from rdkit.SimDivFilters import rdSimDivPickers
import scipy
import os
from ase.io import read
from .nmstools import *
import rmsd
from time import time
#from othertools import *
from .othertools import *


class NormalModeSampler:
    '''
    Do normal mode sampling for a single optimized structure with TorchANI model, below are functions
    1. compute_mode: compute vib modes with TorchANI model
    2. mode_from_log_file: load vib modes from Gaussian log file
    3. normal_mode_sampling: do NMS given vib_modes and vib_fconstants
    4. QBC_check: do QBC check with TorchANI model for sampled structures
    5. remove_similar_struct: remove structures with small delta_rmsd (<0.01) & with small delta_E(<0.02Har)
    6. get_diverse_confs: given ratios of conformations you want to keep
    '''

    def __init__(self, infile, Nconf, T, ifCompMode=False, TANI_model=None, freqlogfile=None, ifLinear=False):
        self.infile = infile              # infile should be a single optimized structure (eg. xyz/pdb etc)
        self.mol = read(self.infile)           
        self.Nconf = Nconf                # num of confs to sample
        self.spc = self.mol.get_atomic_numbers() # array of atomic index
        self.coord = self.mol.get_positions()    # 2D array of atomic positions
        self.Na = len(self.spc)           # num of atoms
        self.T = T                        # temp in K
        self.model = TANI_model           # TorchANI
        self.logfile = freqlogfile        # gaussian log file
        self.ifCompMode = ifCompMode      # set this to True if need to compute modes with TorchANI
        self.ifLinear = ifLinear          # set this to True if it is a linear molecule
        self.device = torch.device('cpu')

    
    def compute_modes(self):
        '''
        Use TorchANI vibrational_analysis function to compute MDN normal modes (Same as Gaussian), 
        Gen all 3*N modes, have to remove 5 or 6 modes with smallest absolute value of freq
        '''
        species = torch.tensor(self.spc, device=self.device, dtype=torch.long).unsqueeze(0)
        coordinates = torch.tensor(self.coord, device=self.device).float().unsqueeze(0).requires_grad_(True)
        masses = torchani.utils.get_atomic_masses(species)
        energies = self.model((species, coordinates)).energies
        hessian = torchani.utils.hessian(coordinates, energies = energies)
        freq, modes, fconstants, rmasses = torchani.utils.vibrational_analysis(masses, hessian, mode_type='MDN')
        _, indices = torch.sort(torch.abs(freq)) # get indices after sorting
        if self.ifLinear:
            useless_idx = indices[:5] 
        else:
            useless_idx = indices[:6] # default is non-linear molecule, lowest 6 freq modes are rotational/translational
        mask = np.array([False if i in useless_idx else True for i in range(3*self.Na)])
        return modes[mask], torch.abs(fconstants[mask])
            
    
    def modes_from_logfile(self):
        '''
        Extract modes from Gaussian log file (if do freq calculation)
        works for linear-shaped geometry
        '''
        loglines = open(self.logfile, 'r').readlines()
        record_ids = [lid for lid, line in enumerate(loglines) if 'Frc consts  --' in line] # find line_id records Force constants
        num_modes = self.Na*3-6 if len(record_ids) == self.Na-2 else self.Na*3-5            # detect num of vib_modes
        vib_modes = np.empty([num_modes, self.Na, 3], dtype=float)
        vib_fconst = np.empty(num_modes, dtype=float)
        for i in range(num_modes):
            lid = record_ids[i//3]  # represent line_id in loglines that have ith mode's force constant
            rid = i%3               # if i = 0/1/2/3/4/5, rid = 0/1/2/0/1/2, ith force constant is ridth fconst in the line
            vib_fconst[i] = float(loglines[lid].split()[rid + 3]) # update force constant
            for r in range(self.Na):
                line = loglines[lid + 3 + r]
                for c in range(3):
                    vib_modes[i][r][c] = float(line.split()[2 + 3*rid + c]) # update vib_modes coordinates
        return vib_modes, vib_fconst
    

    def normal_mode_sampling(self):
        '''
        This function only do simply NMS based on vib_modes and vib_fconsts
        '''
        if self.ifCompMode:
            vib_modes, vib_fconst = self.compute_modes()
        else:
            vib_modes, vib_fconst = self.modes_from_logfile()
        #breakpoint()
        nms = nmsgenerator(self.coord, vib_modes, vib_fconst, self.spc, self.T)
        conformers = nms.get_Nrandom_structures(self.Nconf)

        return conformers
    

    def QBC_check(self, conformers, qbc=0.23):
        '''
        Based on sampled conformers, do QBC check for active learning, default qbc is < 0.23 kcal/mol
        '''
        qbc_threshold = qbc / torchani.units.HARTREE_TO_KCALMOL
        # remove conformation with qbc < self.qbc_threshold
        Ngen = conformers.shape[0]
        mask = np.array([True]*Ngen)
        spc = torch.tensor(self.spc, device = self.device, dtype=torch.long).unsqueeze(0)
        for i, conf in enumerate(conformers):
            coord = torch.tensor(conf, device=self.device).float().unsqueeze(0)
            _, _, qbc = self.model.energies_qbcs((spc, coord))
            if qbc < qbc_threshold:
                mask[i] = False
        return conformers[mask]


    def remove_similar_struct(self, conformers, rmsd_threshold = 0.01, dE_threshold = 2e-3, no_hydrogen=True):
        '''
        remove stuctures with very similar structures based on rmsd and dE
        if two structures are very close (delta_rmsd < 0.01) and the surface is flat (dE < 2e-3 hartree)
        Slow when num_conf >= 2000, because alignments takes time
        '''
        Ngen = conformers.shape[0]
        if no_hydrogen:
            mask = np.array([False if self.spc[i]==1 else True for i in range(self.Na)])
        else:
            mask = np.array([True] * self.Na)
        
        keep_idx = set([i for i in range(Ngen)])
        species = torch.tensor(self.spc, device=self.device, dtype=torch.long).unsqueeze(0)
        for i in range(Ngen):
            for j in range(i+1, Ngen):
                if i in keep_idx and j in keep_idx:
                    struct_i, struct_j = conformers[i][mask], conformers[j][mask]
                    rmsd_ij = rmsd.kabsch_rmsd(struct_i, struct_j) 

                    if rmsd_ij < rmsd_threshold:
                        coord_i = torch.tensor(conformers[i], device=self.device).float().unsqueeze(0)
                        coord_j = torch.tensor(conformers[j], device=self.device).float().unsqueeze(0)
                        Ei = self.model((species, coord_i)).energies
                        Ej = self.model((species, coord_j)).energies
                        if abs(Ei - Ej) < dE_threshold:
                            keep_idx.remove(j)
                            
        
        keep_mask = np.array([True if i in keep_idx else False for i in range(Ngen)])
        return conformers[keep_mask]


    

    def get_diverse_confs(self, conformers, keep_ratio, no_hydrogen=True, aevsize=384):
        '''
        This function is edited from Justin's code in pyaniasetools.py
        find most diverse conformations in conformers by compare euclidean distance between aevs
        Use keep_ratio to decide how many confs to keep
        It is faster because there's no need to align the structures.
        '''
        Ngen = conformers.shape[0]
        Nkep = int(Ngen * keep_ratio)
        
        if no_hydrogen:
            atom_list = [i for i, s in enumerate(self.spc) if s != 1]  # not consider hydrogen aev
        else:
            atom_list = [i for i in range(self.Na)]                    # consider hydrogen aev

        spc = torch.tensor(self.spc, device = self.device, dtype=torch.long).unsqueeze(0)
        all_aevs = np.empty([Ngen, len(atom_list * aevsize)])
        aev_computer = self.model.aev_computer

        for i in range(Ngen):
            coord = torch.tensor(conformers[i], device=self.device).float().unsqueeze(0)
            spe_coord_convert = self.model.species_converter((spc, coord))
            _, aev = aev_computer(spe_coord_convert)
            for j, a_id in enumerate(atom_list):
                all_aevs[i][j*aevsize : (j+1)*aevsize] = aev[0][a_id]
        
        dist_mat = scipy.spatial.distance.pdist(all_aevs, 'sqeuclidean') # square enclidean distance
        picker = rdSimDivPickers.MaxMinPicker()
        seed_list = [i for i in range(Ngen)]
        np.random.shuffle(seed_list)
        Nkep_idx = list(picker.Pick(dist_mat, Ngen, Nkep, firstPicks=seed_list[:5]))
        Nkep_idx = set(Nkep_idx)
        mask = np.array([True if i in Nkep_idx else False for i in range(Ngen)])
        return conformers[mask]
    

    def get_distance(self, conformers, a1, a2):
        '''
        get atomic distance from a1 to a2, return an array of all distance in conformers
        '''
        dist = np.empty(conformers.shape[0])
        for i, conf in enumerate(conformers):
            dist[i] = np.sqrt(sum((conf[a1] - conf[a2]) ** 2))  # compute distance between two atoms
        return dist


    def write_conformers(self, conformers, write_dir, Onefile=False):
        '''
        write conformers in xyz format
        Onefile = True: write all confs in one xyz file
        Onefile = False: write confs in different xyz files with index (0 to 999,999 confs)
        '''
        if not os.path.exists(write_dir):
            os.mkdir(write_dir)
        basename = os.path.basename(self.infile).split('.')[0] + '_confs'
        spc = np.array([AtomicNumToSymbol(x) for x in self.spc])
        if Onefile:
            write_allxyz(os.path.join(write_dir, basename + '.xyz'), spc, conformers)
        else:
            for fn in os.listdir(write_dir):
                # if conformers with same prefix has been writen before, remove them
                if fn.startswith(basename):
                    os.remove(os.path.join(write_dir, fn))
            for i, conf in enumerate(conformers):
                write_xyz(os.path.join(write_dir, basename + str(i).zfill(6) + '.xyz'), spc, conf)


class FilterStructures:
    '''
    Given a fold of single structure xyz files, roughly remove xyzfile with similar structures
    Used after NMS on a trajectory of points (may has duplicates between different points' NMS)
    '''
    def __init__(self, confs_dir):
        self.dir = confs_dir
        self.filenames = os.listdir(self.dir)
        self.filenames.sort()
        self.filenames = np.array(self.filenames)
        self.Nconf = len(self.filenames)
        self.mol1 = read(os.path.join(self.dir, self.filenames[0]))
        self.spc = self.mol1.get_atomic_numbers()
        self.Na = len(self.spc)
    
    def _gen_structs(self):
        conformers = np.empty([self.Nconf, self.Na, 3])
        for i, fn in enumerate(self.filenames):
            mol = read(os.path.join(self.dir, fn))
            conformers[i] = mol.get_positions()
        return conformers

    
    def remove_similar_struct(self, rmsd_threshold = 0.01, no_hydrogen=True):

        if no_hydrogen:
            mask = np.array([False if self.spc[i]==1 else True for i in range(self.Na)])
        else:
            mask = np.array([True] * self.Na)
        
        conformers = self._gen_structs()
        keep_idx = set([i for i in range(self.Nconf)])
        for i in range(self.Nconf):
            for j in range(i+1, self.Nconf):
                if i in keep_idx and j in keep_idx:
                    struct_i, struct_j = conformers[i][mask], conformers[j][mask]
                    rmsd_ij = rmsd.rmsd(struct_i, struct_j) 
                    if rmsd_ij < rmsd_threshold:
                        keep_idx.remove(j)
                            
        remove_mask = np.array([False if i in keep_idx else True for i in range(self.Nconf)])
        remove_filenames = self.filenames[remove_mask]
        
        for fn in remove_filenames:
            os.remove(os.path.join(self.dir,fn))

        print('remove %s duplicated structures'%len(remove_filenames))
        
        return
    
    







                
            
            


    



        

    


