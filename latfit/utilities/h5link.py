#!/usr/bin/python3
"""Link a single config's hdf5 files into
one hdf5 file for the whole config"""
import sys
import re
import os
import re
import glob
import h5py


def get_trajl():
    """Get trajectory list"""
    trajl = []
    for i, filen in enumerate(glob.glob('*hdf5')):
        if i:
            pass
        mat = re.search(r'traj_(\d+)_', filen)
        if not mat:
            continue
        else:
            traj = int(mat.group(1))
        trajl.append(traj)
    trajl = sorted(list(set(trajl)))
    return trajl


def main():
    """Link hdf5 files"""
    trajl = get_trajl()
    for traj in trajl:
        file1 = 'traj_'+str(traj)+'.hdf5'
        if os.path.isfile(file1):
            fn1 = h5py.File(file1, 'r+')
        else:
            fn1 = h5py.File(file1, 'w')
        for j, file2 in enumerate(glob.glob('traj_'+str(traj)+'_*hdf5')):
            if 'exact' in file2:
                continue
            if j:
                pass
            try:
                gn1 = h5py.File(file2, 'r')
            except OSError:
                print("hdf5 linker error")
                print("problem file=", file2)
                sys.exit(1)
            datal = []
            try:
                for data in gn1:
                    datal.append(str(data))
            except RuntimeError:
                print("Bad symbol table in:", file2)
                print("Dataset right before failure:", datal)
                print("trying to link to", file1)
                continue
                raise
            gn1.close()
            for data in datal:
                #if '4542' in data:
                #    data_ln = re.sub('4541', '4540', data)
                #else:
                data_ln = data
                if data_ln not in fn1:
                    try:
                        fn1[data_ln] = h5py.ExternalLink(file2, data)
                    except ValueError:
                        print("Error.  link file, orig file, dataset name:")
                        print(file1, file2, data)
                        print("link name:", data_ln)
                        sys.exit(1)
            print("Done linking", file2, file1)
        fn1.close()


if __name__ == '__main__':
    main()
