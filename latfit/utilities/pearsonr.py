#!/usr/bin/python3
"""Calculate correlation coefficients"""
import sys
import h5py
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr

def dataset(fn1):
    """Collapse a flat numpy file"""
    dataset_var = None
    ret = None
    prefix = '/'
    while dataset_var is None:
        for i in fn1[str(prefix)]:
            i = prefix + '/' + i
            if isinstance(fn1[str(i)], h5py._hl.group.Group):
                prefix = i
            else:
                dataset_var = i
                ret = np.asarray(fn1[str(dataset_var)])
    return ret

def rcoeff(file1, file2):
    """Find the pearson r coefficient"""
    fn1 = h5py.File(str(file1), 'r')
    gn1 = h5py.File(str(file2), 'r')
    adat = dataset(fn1)
    adat = np.real(adat)
    bdat = dataset(gn1)
    bdat = np.real(bdat)
    lent = len(adat[0])
    for i in range(lent):
        print(i, pearsonr(adat[:, i], bdat[:, i]),
              spearmanr(adat[:, i], bdat[:, i]))

if __name__ == '__main__':
    rcoeff(sys.argv[1], sys.argv[2])
