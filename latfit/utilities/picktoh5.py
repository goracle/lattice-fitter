#!/usr/bin/python3
"""convert pickled numpy array to hdf5"""

import re
import sys
import pickle
import numpy as np
import h5py


def main(fname):
    """main"""
    oname = re.sub(r'.p$', r'.hdf5', fname)
    name = re.sub(r'.p$', '', fname)
    print("converting", fname, "to", oname)
    fn1 = open(fname, 'rb')
    arr = pickle.load(fn1)
    arr = np.array(arr)
    fn1 = h5py.File(oname)
    fn1[name] = arr
    fn1.close()
    print("done with conversion.")


if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv
    main(sys.argv[1])
