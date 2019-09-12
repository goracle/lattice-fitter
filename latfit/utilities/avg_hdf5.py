#!/usr/bin/python3
"""Average several hdf5 datasets (pipipbc project)"""
import sys
import re
from pathlib import Path
import numpy as np
import h5py
from latfit.utilities import exactmean as em

OUTNAME = ''

def main(*args):
    """Average the datasets from the command line"""
    if not args:
        args = sys.argv
    else:
        args = [sys.argv[0], *args]
    assert len(args) > 1, "Input list of files to average"
    norm = 1.0/(len(args)-1)
    for i, data in enumerate(args):
        if i == 0:
            continue
        try:
            fn1 = h5py.File(data, 'r')
        except OSError:
            print("Not including:", data, "File doesn't exist.")
            norm = 1.0/(1.0/norm-1)
            sys.exit(1)
        print("Including", data)
        for k in fn1:
            isospin = k
            for item in fn1[k]:
                try:
                    setname = k+'/'+item
                except TypeError:
                    isospin = ''
                    setname = k
                break
            break
        print('adding in dataset=', setname, "in file=", data, 'i=', i)
        if i == 1:
            avg = []
        avg.append(np.asarray(fn1[setname], dtype=np.complex128))
    avg = em.acsum(np.asarray(avg))
    print("multiplying by norm=", norm)
    avg *= norm
    name = str(input("output name?")) if not OUTNAME else OUTNAME
    name = re.sub('.jkdat', '', name)
    pathname = Path(name+'.jkdat')
    if not pathname.is_file():
        fn1 = h5py.File(name+'.jkdat', 'w')
        fn1[isospin+'/'+name.split('/')[-1]] = avg
        fn1.close()
        print("done.")
    else:
        print(name+'.jkdat', "exists. Skipping.")




if __name__ == '__main__':
    main()
