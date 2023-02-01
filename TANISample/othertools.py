import numpy as np
import os



def AtomicNumToSymbol(x):
    dict = {1:'H', 6:'C', 7:'N', 8:'O'}
    x = int(x)
    if x not in dict:
        raise KeyError("Do not have atomic number %f"%x)
    else:
        return dict[x]


def write_xyz(filename, spc, conformer, comment=''):
    # write a single sturcture in xyz file
    f = open(filename, 'w')
    spc, conformer = np.array(spc), np.array(conformer)
    Na = len(spc)
    f.write(str(Na) + '\n' + comment + '\n')
    for i in range(Na):
        line = spc[i] + '  ' + '    '.join("{:.7f}".format(x) for x in conformer[i])
        f.write(line + '\n')
    f.close()


def write_allxyz(filename, spc, conformers):
    # write multiple structures in one xyz file
    f = open(filename, 'w')
    spc, conformers = np.array(spc), np.array(conformers)
    Na = len(spc)
    for i, conf in enumerate(conformers):
        comment = 'conf_' + str(i).zfill(6)
        f.write(str(Na) + '\n' + comment + '\n')
        for j in range(Na):
            line = spc[j] + '  ' + '    '.join("{:.7f}".format(x) for x in conformers[i][j ])
            f.write(line + '\n')
    f.close()
        
