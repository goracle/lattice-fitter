"""Get covariance matrix element, blocks, coordinates"""
import sys
from collections import namedtuple
import numpy as np

from latfit.extract.getblock import getblock
from latfit.extract.get_coventry import get_coventry

from latfit.config import BINNUM

def proc_ijfile(ifile_tup, jfile_tup, reuse=None):
    """Process the current file.
    Return covariance matrix entry I, indexj in the case of multi-file
    structure.
    Return the covariance matrix for single file.
    """
    if reuse is None:
        reuse = {}
    rets = namedtuple('rets', ['coord', 'covar', 'returnblk'])

    #do i
    try:
        num_configs = len(reuse['i'])
    except TypeError:
        reuse['i'] = getblock(ifile_tup, reuse)
        num_configs = len(reuse['i'])
    avg_i = np.mean(reuse['i'], axis=0)

    #print num of configs
    if proc_ijfile.CONFIGSENT != 0:
        print("Number of configurations to average over:", num_configs)
        if BINNUM != 1:
            print("Configs per bin:", BINNUM)
        proc_ijfile.CONFIGSENT = 0

    #check if we're on the same block (i==j)
    sameblk = np.array_equal(ifile_tup, jfile_tup)

    if sameblk:
        reuse['j'] = reuse['i']
    else:
        #do j
        try:
            #check to make sure i, j have the same number of lines
            if not num_configs == len(reuse['j']):
                print("***ERROR***")
                print("Number of configs not equal for i and j")
                print("Offending files:", ifile_tup, "\nand", jfile_tup)
                sys.exit(1)
        except TypeError:
            reuse['j'] = getblock(jfile_tup, reuse)

    return rets(coord=avg_i, covar=get_coventry(reuse, sameblk, avg_i), returnblk=reuse['j'])

#test to see if we've already printed the number of configurations
proc_ijfile.CONFIGSENT = object()
