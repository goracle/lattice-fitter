#!/usr/bin/python3
"""Convert ascii files to hdf5"""
import sys
from os.path import isfile
import re
import glob
import numpy as np
import h5py
import read_file as rf
from traj_list import traj_list

LT = 64
# throws an exception when profile isn't defined
try:
    PROFILE
except NameError:
    def PROFILE(x):
        """Line profiler default."""
        return x


@PROFILE
def main():
    """should be run independently to do the conversion"""
    trajl = traj_list()
    for traj in trajl:
        outname = str(traj)+'.dat'
        if isfile(outname):
            print("Skipping:", outname, "File exists.")
            continue
        outh = h5py.File(outname, 'w')
        for i, filen in enumerate(glob.glob('traj_'+str(traj)+'_*')):
            if i:
                pass
            if re.search(r'Figure_', filen):
                data = np.zeros((LT), dtype=np.complex)
                squ = False
            else:
                data = np.zeros((LT, LT), dtype=np.complex)
                squ = True
            if squ:
                for line in open(filen, 'r'):
                    lsp = line.split()
                    try:
                        data[int(lsp[0]), int(lsp[1])] = np.complex(
                            float(lsp[2]), float(lsp[3]))
                    except IndexError:
                        print("Error: bad file format:", filen)
                        sys.exit(1)
            else:
                for line in open(filen, 'r'):
                    lsp = line.split()
                    data[int(lsp[0])] = np.complex(
                        float(lsp[1]), float(lsp[2]))
            outh[filen] = data
            setatr(outh[filen], filen)
        outh.close()
        print("Done converting trajectory:", traj)


@PROFILE
def setatr(dataset, filen):
    """Add metadata onto hdf5 datasets"""
    # pol = rf.pol(filen)
    # if pol is not None:
    #    pass
    #     dataset.attrs['pol'] =  pol
    # else:
    #    dataset.attrs['pol'] =  False
    # traj_here = rf.traj(filen)
    # if traj_here is not None:
    #    dataset.attrs['traj'] = traj_here
    # else:
    #    dataset.attrs['traj'] = False
    # dataset.attrs['reverse_p'] = rf.reverse_p(filen)
    # dataset.attrs['checkp'] = rf.checkp(filen)
    # dataset.attrs['vecp'] = rf.vecp(filen)
    # fig = rf.figure(filen)
    # if fig is not None:
    #    dataset.attrs['figure'] = fig
    # else:
    #    dataset.attrs['figure'] = False
    mom = rf.mom(filen)
    if mom is not None:
        dataset.attrs['mom'] = mom
    else:
        dataset.attrs['mom'] = False
    # sep = rf.sep(filen)
    # if sep is not None:
    #    dataset.attrs['sep'] = sep
    # else:
    #    dataset.attrs['sep'] = False
    dataset.attrs['basename'] = rf.basename(filen)
    return dataset


if __name__ == '__main__':
    main()
