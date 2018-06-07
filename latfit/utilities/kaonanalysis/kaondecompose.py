"""Decompose kaon results container and mix containers
from 1d arrays to multi-dimensional"""

from h5jack import LT as LT_CHECK
print("Imported kaondecompose with LT=", LT_CHECK)

def decomposeMix(mixarr, avgTk=False):
    """Decompose mix array into pieces indexed by
    (fidx, tdis, timek)
    """
    ret = {}
    idx_prev = -1
    for fidx in range(2):
        for tdis in range(LT_CHECK):
            for timek in range(LT_CHECK):
                idx = mapMix(fidx, tdis, timek)
                assert idx == idx_prev+1, "Index is running too fast."
                if avgTk:
                    if timek == 0:
                        ret[(fidx, tdis)] = mixarr[idx]
                    else:
                        ret[(fidx, tdis)] += mixarr[idx]
                    else:
                        ret[(fidx, tdis, timek)] = mixarr[idx]
                    idx_prev = idx
                if avgTk:
                    ret[(fidx, tdis, timek)] /= LT_CHECK
    return ret

def mapMix(fidx, tdis, timek):
    """Map to 1d array (non-mix)"""
    return fidx + 2*( tdis + LT_CHECK*( timek ) )


def decompose(array, ncontract, avgTk=False, tstep=1):
    """Decompose array into pieces indexed by
    (ncontract, muAx, tdis, timek)
    """
    ret = {}
    idx_prev = -1
    for condix in range(ncontract):
        for gcombidx in range(4):
            for tdis in range(LT_CHECK):
                for timek in range(LT_CHECK):
                    idx = mapres(condix, gcombidx, tdis, timek, ncontract)
                    assert idx == idx_prev+1, "Index is running too fast."
                    if avgTk:
                        if timek == 0:
                            ret[(condix, gcombidx, tdis)] = array[idx]
                        else:
                            ret[(conidx, gcombidx, tdis)] += array[idx]
                    else:
                        ret[(condix, gcombidx, tdis, timek)] = array[idx]
                    idx_prev = idx
                if avgTk:
                    ret[(conidx, gcombidx, tdis)] /= LT_CHECK/tstep
    return ret

def mapres(conidx, gcombidx, tdis, timek, ncontract):
    """Map to 1d array (non-mix)"""
    return conidx + ncontract*( gcombidx + 4*( tdis + LT_CHECK*( timek ) ) ) 
    
#inline int map(const int tk, const int t_dis, const int con_idx, const int gcombidx, const int thread) const{
#  return con_idx + ncontract*( gcombidx + 4*( t_dis + Lt*( tk + Lt*thread) ) );
#}

  
