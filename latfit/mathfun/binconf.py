"""Bin configs"""
import sys
import numpy as np

from latfit.config import BINNUM
from latfit.config import JACKKNIFE
import latfit.finalout.mkplot
from latfit.utilities import exactmean as em


def binconf(jkblk, binnum=BINNUM):
    """Dynamic binning of configs.  BINNUM configs per bin.
    """
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to bin configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    if binnum == 1:
        ret = jkblk

    else:
        assert len(jkblk)%binnum == 0,\
            "non divisible binnum not supported:"+str(len(jkblk))
        inv = em.acsum(jkblk, axis=0)-(len(jkblk)-1)*jkblk
        inv2 = np.array([em.acmean(inv[j*binnum:(j+1)*binnum],
                                   axis=0) for j in range(int(
                                       len(jkblk)/binnum))])
        ret = np.array([em.acmean(np.delete(inv2, j, axis=0), axis=0)
                        for j in range(len(inv2))])
    # print("updating title number of configs (after binning) to:", len(ret))
    latfit.finalout.mkplot.NUM_CONFIGS = len(ret)
    return ret


#    first_len = len(jkblk)
#    elim_list=list(
#        range(first_len)[-(first_len % BINNUM):])
#    print(elim_list)
#    sys.exit(0)
#    newblk = elim_jkconfigs(jkblk, elim_list=list(
#        range(first_len)[-(first_len % BINNUM):]))
#    newblk = jkblk
#    alen = len(newblk)
#    sum_blk = em.acsum(newblk, axis=0)
#    newblk = np.array([em.acsum(newblk[j*BINNUM:(j+1)*BINNUM], axis=0)
#                       for j in range(int(alen/BINNUM))])
#    newblk *= (alen-1.0)
#    newblk -= (BINNUM-1.0)*sum_blk
#    newblk /= (alen-BINNUM)*1.0
