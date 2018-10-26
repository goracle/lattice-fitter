"""Bin configs"""
import sys
import numpy as np

# from latfit.mathfun.elim_jkconfigs import elim_jkconfigs

from latfit.config import BINNUM
from latfit.config import JACKKNIFE


def binconf(jkblk):
    """Dynamic binning of configs.  BINNUM configs per bin.
    """
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to bin configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    if BINNUM == 1:
        return jkblk

    assert len(jkblk)%BINNUM == 0,\
        "non divisible BINNUM not supported:"+str(len(jkblk))
    inv = np.sum(jkblk, axis=0)-(len(jkblk)-1)*jkblk
    inv2 = np.array([np.mean(inv[j*BINNUM:(j+1)*BINNUM],
                             axis=0) for j in range(int(len(jkblk)/BINNUM))])
    newblk = np.array([np.mean(np.delete(inv2, j, axis=0), axis=0)
                       for j in range(len(inv2))])
    return newblk


#    first_len = len(jkblk)
#    elim_list=list(
#        range(first_len)[-(first_len % BINNUM):])
#    print(elim_list)
#    sys.exit(0)
#    newblk = elim_jkconfigs(jkblk, elim_list=list(
#        range(first_len)[-(first_len % BINNUM):]))
#    newblk = jkblk
#    alen = len(newblk)
#    sum_blk = np.sum(newblk, axis=0)
#    newblk = np.array([np.sum(newblk[j*BINNUM:(j+1)*BINNUM], axis=0)
#                       for j in range(int(alen/BINNUM))])
#    newblk *= (alen-1.0)
#    newblk -= (BINNUM-1.0)*sum_blk
#    newblk /= (alen-BINNUM)*1.0
