#!/usr/bin/python3

import sys
import h5py
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


   

def dataset(fn1):
    """Collapse a flat numpy file"""
    dataset = None
    ret = None
    prefix = '/'
    while dataset is None:
        for i in fn1[str(prefix)]:
            i = prefix + '/' + i
            if isinstance(fn1[str(i)], h5py._hl.group.Group):
                prefix = i
            else:
                dataset = i
                ret = np.asarray(fn1[str(dataset)])
    return ret
    
 
def rcoeff(file1, file2):
    """Find the pearson r coefficient"""
    fn1=h5py.File(str(file1), 'r')
    gn1=h5py.File(str(file2), 'r')
    a=dataset(fn1)
    a=np.real(a)
    b=dataset(gn1)
    b=np.real(b)
    LT = len(a[0])
    for i in range(LT):
        print(i, pearsonr(a[:, i], b[:, i]), spearmanr(a[:, i], b[:, i]))

if __name__ == '__main__':
    rcoeff(sys.argv[1], sys.argv[2])
