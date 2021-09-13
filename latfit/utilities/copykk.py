#!/usr/bin/python3
"""Copy off-diagonal KK->x operator files to x->KK"""

import sys
import re
from glob import glob
import h5py
import numpy as np


def main():
    """main"""
    for fname in glob('kk*jkdat'):
        if 'pisq' in fname:
            continue
        print("processing:", fname)
        assert '_A_1PLUS_mom000.jkdat' in fname, fname
        fname1 = re.sub('S_pipi', 'S@pipi', fname)
        spl = fname1.split('_')[0]
        spl = re.sub('S@pipi', 'S_pipi', spl)
        assert 'kk' == spl[0:2], (spl, fname)
        other_op = spl[2:]

        fn1 = h5py.File(fname, 'r')
        assert len(fn1) == 1, len(fn1)
        for i in fn1:
            iso = i
            break
        assert len(fn1[iso]) == 1, len(fn1[iso])
        for i in fn1[iso]:
            dat = i
            break
        arr = np.array(fn1[iso][dat], dtype=np.complex128)
        fn1.close()
        fn1 = h5py.File(other_op+'kk_A_1PLUS_mom000.jkdat', 'w')
        dname = iso+'/'+other_op+'kk_A_1PLUS_mom000'
        fn1[dname] = arr
        fn1.close()
    print("done.")

if __name__ == '__main__':
    main()
