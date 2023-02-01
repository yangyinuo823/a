'''
Edit from Justin's code nmstools.py by Yinuo on 2023/01/23
Main changes: change Nf to Na in the NMS equation
'''
import numpy as np


# constants
mDynetoMet = 1.0e-5 * 1.0e-3 * 1.0e10  # 1 dyn = 10-5 kg*m*s-2, convert millidyn's meter to Angs
Kb = 1.38064852e-23  # in unit of m2 * kg * s-2 * K-1
MtoA = 1.0e10        # meter to Angstrom


# normal mode sampling generator
class nmsgenerator():
    # xyz = initial min structure
    # nmo = normal mode displacements
    # fcc = force constants
    # spc = atomic species list
    # T   = temperature of displacement
    def __init__(self,xyz,nmo,fcc,spc,T,minfc = 1.0E-3,maxd=2.0):
        self.xyz = np.array(xyz)           # in A
        self.nmo = np.array(nmo)           # unitless
        self.fcc = np.array([i if i > minfc else minfc for i in fcc]) # in Millidyne/Angs = 10-3 N/Ang = 10-8 kg*m*s-2/Ang
        self.chg = np.array(spc) # spc are represented by '1', '6', '7', '8'
        self.T = T               # in K
        self.Na = xyz.shape[0]   # num of atoms
        self.Nf = nmo.shape[0]   # num of normal modes (3Na-6 for non-linear mol)
        self.maxd = maxd         # max displacement
    

    # Generate a structure
    def __genrandomstruct__(self):
        rdt = np.random.random(self.Nf+1) # return Nf+1 random floats in [0.0, 1.0) interval
        rdt[0] = 0.0
        norm = np.random.random(1)[0] # gen 1 random float in [0.0, 1.0) internal
        rdt = norm*np.sort(rdt)       # scale rdt by a random float scalar [0.0, norm) interval
        rdt[self.Nf] = norm           # make rdt: [0, sorted_num, norm]

        oxyz = self.xyz.copy()

        for i in range(self.Nf):
            Ki = mDynetoMet * self.fcc[i] # convert to Kg*s-2
            ci = rdt[i+1]-rdt[i]          # to make sure sum(ci) in range of [0,1)
            Sn = -1.0 if np.random.binomial(1,0.5,1) else 1.0  # 50%, 50% chances to have 1.0 and -1.0 
            Ri = Sn * MtoA * np.sqrt((3.0 * ci * Kb * float(self.Na) * self.T)/(Ki)) # unit of Ri: Ang, paper uses Na, but Justin use Nf
            Ri = min([Ri,self.maxd])      # do not exceed maxd scalar (2.0 A)
            oxyz = oxyz + Ri * self.nmo[i]
        return oxyz

    # Checks for small atomic distances
    def __check_atomic_distances__(self,rxyz):
        for i in range(0,self.Na):
            for j in range(i+1,self.Na):
                Rij = np.linalg.norm(rxyz[i]-rxyz[j])
                if Rij < 0.006 * (self.chg[i] * self.chg[j]) + 0.6:
                    return True
        return False

    # Checks for large changes in atomic distance from eq
    def __check_distance_from_eq__(self,rxyz):
        for i in range(0,self.Na):
            Rii = np.linalg.norm(self.xyz[i]-rxyz[i])
            if Rii > 2.0:
                return True
        return False

    # Call this to return a random structure
    def get_random_structure(self):
        gs = True
        fnd = True
        count = 0
        while gs:
            rxyz = self.__genrandomstruct__()
            # make sure no very short atomic distance, and keep atomic_distance within 2A from equilibrium
            gs = self.__check_atomic_distances__(rxyz) or self.__check_distance_from_eq__(rxyz)
            if count > 100:
                fnd=False
                break
            count += 1
        return rxyz,fnd

    # Call this to return a random structure
    def get_Nrandom_structures(self, N):
        a_xyz = np.empty((N, self.Na, 3),dtype=np.float32)
        a_bool = np.empty((N),dtype=bool)
        for i in range(N):
            a_xyz[i],a_bool[i] = self.get_random_structure()
        return a_xyz[a_bool]  # use a_bool as a mask to filter out not qualified structures!
