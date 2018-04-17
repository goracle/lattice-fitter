#!/usr/bin/python3
"""Average several hdf5 datasets (pipipbc project)"""
import sys
import re
from pathlib import Path
import numpy as np
import h5py

OUTNAME = ''

def main(*args):
    """Average the datasets from the command line"""
    if not args:
        args = sys.argv
    else:
        args = [sys.argv[0], *args]
    assert len(args) > 1, "Input list of files to average"
    norm = 1.0/(len(args)-1)
    avg = np.array([])
    for i, data in enumerate(args):
        if i == 0:
            continue
        try:
            fn = h5py.File(data, 'r')
        except OSError:
            print("Not including:", data, "File doesn't exit.")
            norm = 1.0/(1.0/norm-1)
            continue
        print("Including", data)
        for k in fn:
            isospin = k
            for l in fn[k]:
                try:
                    setname = k+'/'+l
                except TypeError:
                    isospin = ''
                    setname = k
                break
            break
        if i == 1:
            avg = np.zeros(fn[setname].shape, dtype=np.complex128)
        avg += np.array(fn[setname])
    print("multiplying by norm=", norm)
    avg *= norm
    name = str(input("output name?")) if not OUTNAME else OUTNAME
    name = re.sub('.jkdat', '', name)
    pathname = Path(name+'.jkdat')
    if not pathname.is_file():
        fn = h5py.File(name+'.jkdat', 'w')
        fn[isospin+'/'+name.split('/')[-1]] = avg
        fn.close()
        print("done.")
    else:
        print(name+'.jkdat', "exists. Skipping.")

            


if __name__ == '__main__':
    main()
