#!/usr/bin/python3
"""Average several hdf5 datasets (pipipbc project)"""
import sys
import numpy as np
import h5py

def main():
    """Average the datasets from the command line"""
    assert len(sys.argv) > 1, "Input list of files to average"
    norm = 1.0/(len(sys.argv)-1)
    avg = np.array([])
    for i, data in enumerate(sys.argv):
        if i == 0:
            continue
        print("Including", data, "with norm", norm)
        fn = h5py.File(data, 'r')
        for k in fn:
            isospin = k
            for l in fn[k]:
                setname = k+'/'+l
                break
            break
        if i == 1:
            avg = np.zeros(fn[setname].shape, dtype=np.complex128)
        avg += np.array(fn[setname])*norm
    name = str(input("output name?"))
    fn = h5py.File(name+'.jkdat', 'w')
    fn[isospin+'/'+name] = avg
    fn.close()
    print("done.")

            


if __name__ == '__main__':
    main()
