#!/usr/bin/python3
"""Average several hdf5 datasets (pipipbc project)"""
import sys
import re
from pathlib import Path
import numpy as np
import h5py

OUTNAME = ''

# from here
# https://github.com/numpy/numpy/issues/8786
def kahan_sum(a, axis=0):
    a = np.asarray(a)
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/a/42817610/353337
        y = a[(slice(None), ) * axis + (i, )] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()
    return s

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
            fn = h5py.File(data, 'r')
        except OSError:
            print("Not including:", data, "File doesn't exist.")
            norm = 1.0/(1.0/norm-1)
            sys.exit(1)
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
        print('adding in dataset=', setname, "in file=", data, 'i=', i)
        if i == 1:
            avg = []
        avg.append(np.asarray(fn[setname]))
    avg = kahan_sum(avg)
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
