"""Get covariance matrix element, blocks, coordinates"""
import sys
from collections import namedtuple, deque
import numpy as np

from latfit.extract.getblock import getblock
from latfit.extract.get_coventry import get_coventry

def gevp_proc_ijfile(ifile_tup, jfile_tup, time_arr,reuse=None):
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
        num_configs=len(reuse['i'])
    except TypeError:
        reuse['i']=deque()
        getblock(ifile_tup, reuse, 'i')
        num_configs=len(reuse['i'])
    avgI=np.mean(reuse['i'], axis=0)

    #print num of configs
    if gevp_proc_ijfile.CONFIGSENT != 0:
        print("Number of configurations to average over:",num_configs)
        gevp_proc_ijfile.CONFIGSENT = 0

    #check if we're on the same block (i==j)
    sameblk = np.array_equal(ifile_tup,jfile_tup)

    if sameblk:
        reuse['j']=reuse['i']
    else:
        #do j
        try:
            #check to make sure i, j have the same number of lines
            if not num_configs==len(reuse['j']):
                print("***ERROR***")
                print("Number of configs not equal for i and j")
                print("GEVP covariance matrix entry:",time_arr)
                sys.exit(1)
        except TypeError:
            reuse['j']=deque()
            getblock(jfile_tup, reuse, 'j')

    return rets(coord=avgI, covar=get_coventry(reuse, sameblk, avgI),returnblk=reuse['j'])
gevp_proc.CONFIGSENT = object()
