"""Calculate weight sum for list of best fits

We "drop down" onto the plateau.
The first point we land on is our answer
(the first point on the plateau)
"""

from scipy import stats
import numpy as np
import gvar
from latfit.utilities import exactmean as em
from latfit.analysis.superjack import jack_mean_err


# the weight sum

def weight_sum(sorted_block_list):
    """Calculate the weight sum
    return the weighted average
    use list of jackknife blocks
    index runs towards earlier (more goal-like) times
    """
    assert list(sorted_block_list), len(sorted_block_list)
    if len(sorted_block_list) > 1:
        retblk = non_triv_block(sorted_block_list)
    else:
        retblk = sorted_block_list[0]
    # jackknife average
    mean, err = jack_mean_err(retblk)
    return mean, err

def non_triv_block(sorted_block_list):
    """If the number of steps is in the walk back
    is greater than 1, get this non-trivial block"""
    retblk = []
    norm = [] # sum of weights
    for idx, blk in enumerate(sorted_block_list):
        np1list = sorted_block_list[:idx+1]
        fact2 = 1-chained_probs(np1list)
        if idx:
            nlist = sorted_block_list[:idx]
            fact1 = chained_probs(nlist)
        else:
            fact1 = fact2
        weight = fact1*fact2
        norm.append(weight)
        retblk.append(blk*weight)
    # perform the weight sum here
    retblk = np.array(retblk)
    norm = np.array(norm)
    retblk = em.acsum(retblk, axis=0)*em.acsum(1/norm, axis=0)
    return retblk


def chained_probs(nlist):
    """Get the probablity that all blocks in nlist
    have the same mean.
    return a jackknifed answer to that question.
    """
    assert list(nlist)
    ret = np.ones(len(nlist[0]))
    for idx, blk1 in enumerate(nlist):
        for jdx, blk2 in enumerate(nlist):
            if jdx <= idx:
                continue
            assert not np.allclose(blk1, blk2, rtol=1e-12)
            nprob = prob_blk(blk1, blk2)
            nprob = np.asarray(nprob)
            ret *= nprob
    return ret


def invert_jk(blk):
    """Invert jackknife block"""
    asum = em.acsum(blk, axis=0)
    blk2 = blk*len(blk-1)
    ret = asum-blk2
    return ret

def varidx(blk, idx):
    """Calculate the variance of block with index idx
    deleted"""
    invblk = invert_jk(blk)
    dblk = np.delete(invblk, idx, axis=0)
    ret = em.acstd(dblk, axis=0, ddof=1)**2
    return ret

def tstat(blk1, blk2, idx):
    """Calculate the sample statistic
    Dependent t-test for paired samples
    (correlated)
    """
    avg1 = blk1[idx]
    avg2 = blk2[idx]
    avg = avg1-avg2
    var = varidx(blk1-blk2, idx)
    lblk = len(blk1)
    ret = avg
    ret /= np.sqrt(var/lblk)
    dof = lblk-1
    return ret, dof


def prob_blk(blk1, blk2):
    """Calculate probablity that the two blocks are equal"""
    res = []
    assert len(blk1) == len(blk2), (len(blk1), len(blk2))
    lblk = len(blk1)
    for idx in range(lblk):
        tval, dof = tstat(blk1, blk2, idx)
        pval = 1 - stats.t.cdf(tval, dof)
        res.append(pval)
    return res

def main():
    """Test function using gaussian noise"""
    llen = 200
    tavg = []
    for _ in range(3):
        arr1 = np.ones(llen)+np.array([
            np.random.normal(0, 0.01*2**(i+1)) for i in range(llen)])
        tavg.append(arr1)
    tavg = list(reversed(tavg))
    last = np.ones(llen)
    last += np.array([np.random.normal(0.1, 0.01) for _ in range(llen)])
    tavg.append(last)
    mean, err = weight_sum(tavg)
    print("test array weighted sum:")
    print(gvar.gvar(mean, err))




if __name__ == '__main__':
    main()
