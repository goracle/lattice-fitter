#!/usr/bin/python3
import sys
import os.path
from os import listdir
from os.path import isfile, join
from math import sqrt
import glob
import numpy as np
import h5py
import read_file as rf
from traj_list import traj_list


def main():
    trajl = traj_list()
    for traj in trajl:
        outname = str(traj)+'.dat'
        if isfile(outname):
            print("Skipping:", outname, "File exists.")
            continue
        outh = h5py.File(outname, 'w')
        for i, filen in enumerate(glob.glob('traj_'+str(traj)+'_*')):
            num = rf.numlines(filen)
            dim = sqrt(num)
            dimi = int(dim)
            if dim == dimi:
                data = np.zeros((dimi,dimi), dtype=np.complex)
                flag = 'sq'
            else:
                data = np.zeros((num), dtype=np.complex)
                flag = 'non'
            for line in open(filen, 'r'):
                lsp = line.split()
                if flag == 'sq':
                    data[int(lsp[0]),int(lsp[1])]=np.complex(float(
                        lsp[2]),float(lsp[3]))
                else:
                    data[int(lsp[0])]=np.complex(float(lsp[1]),float(lsp[2]))
            outh[filen]=data
            setatr(outh[filen], filen)
        outh.close()
        print("Done converting trajectory:", traj)
                
            
def setatr(dataset, filen):
    pol = rf.pol(filen)
    if pol is not None:
        dataset.attrs['pol'] =  pol
    else:
        dataset.attrs['pol'] =  False
    traj_here = rf.traj(filen)
    if traj_here is not None:
        dataset.attrs['traj'] = traj_here
    else:
        dataset.attrs['traj'] = False
    dataset.attrs['reverse_p'] = rf.reverse_p(filen)
    dataset.attrs['checkp'] = rf.checkp(filen)
    dataset.attrs['vecp'] = rf.vecp(filen)
    fig = rf.figure(filen)
    if fig is not None:
        dataset.attrs['figure'] = fig
    else:
        dataset.attrs['figure'] = False
    mom = rf.mom(filen)
    if mom is not None:
        dataset.attrs['mom'] = mom
    else:
        dataset.attrs['mom'] = False
    sep = rf.sep(filen)
    if sep is not None:
        dataset.attrs['sep'] = sep
    else:
        dataset.attrs['sep'] = False
    dataset.attrs['basename'] = rf.basename(filen)
    return dataset


if __name__ == '__main__':
    main()
