#!/usr/bin/python3
"""Convert simple (one dataset) hdf5 file to pickle file"""

import re
import sys
import pickle
import numpy as np
import h5py


def main(fname):
    """main"""
    oname = re.sub(r'.hdf5$', r'.p', fname)
    if oname[-2:] != '.p':
        oname = oname + '.p'
    name = re.sub(r'.hdf5$', '', fname)
    print("converting", fname, "to", oname)
    fn1 = h5py.File(fname, 'r')
    arr = [fn1[dat] for dat in fn1][0]
    arr = np.array(arr)
    fn1.close()
    fn1 = open(oname, 'wb')
    pickle.dump(arr, fn1)
    fn1.close()
    print("done with conversion.")


if __name__ == '__main__':
    assert len(sys.argv) == 2, sys.argv
    main(sys.argv[1])

