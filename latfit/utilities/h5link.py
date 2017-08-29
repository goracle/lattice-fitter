#!/usr/bin/python3
import sys
import os
import re
import glob
import numpy as np
import h5py


def main():
    trajl = []
    for i, filen in enumerate(glob.glob('*hdf5')):
        m = re.search('traj_(\d+)_', filen)
        if not m:
            continue
        else:
            traj = int(m.group(1))
        trajl.append(traj)
    trajl = sorted(list(set(trajl)))
    for traj in trajl:
        file1 = 'traj_'+str(traj)+'.hdf5'
        if os.path.isfile(file1):
            fn = h5py.File(file1,'r+')
        else:
            fn = h5py.File(file1,'w')
        for j, file2 in enumerate(glob.glob('traj_'+str(traj)+'_*hdf5')):
            gn = h5py.File(file2, 'r')
            datal = []
            for data in gn:
                datal.append(str(data))
            gn.close()
            for data in datal:
                if not data in fn:
                    try:
                        fn[data]=h5py.ExternalLink(file2,data)
                    except ValueError:
                        print("Error.  link file, orig file, dataset name:")
                        print(file1,file2,data)
                        sys.exit(1)
            print("Done linking",file2,file1)
        fn.close()







if __name__ == '__main__':
    main()
