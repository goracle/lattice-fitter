"""Block the ensemble by dropping JACKKNIFE_BLOCK_SIZE configs"""

import sys
import copy
import random
import numpy as np
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.extract.inverse_jk import inverse_jk
from latfit.utilities import exactmean as em
from latfit.utilities.postprod.h5jack import dojackknife
from latfit.utilities import exactmean as em
from latfit.config import BOOTSTRAP_BLOCK_SIZE, NBOOT
from latfit.config import RANDOMIZE_ENERGIES, GEVP, EFF_MASS
from latfit.config import SYS_ENERGY_GUESS
import latfit.config

print("Using bootstrap block size:", BOOTSTRAP_BLOCK_SIZE)

def build_choices_set(block, nconfigs, nboot=0):
    """Find allowed indices for non-overlapping block
    bootstrap"""
    ret = np.arange(0, nconfigs, block)
    ret = list(ret)
    try:
        assert not (nconfigs % block),\
            "blocking with a remainder is complicated and not supported yet"
    except AssertionError:
        remainder = nconfigs-(nconfigs % block)
        print(nconfigs)
        print(block)
        print(remainder)
        raise
    # balanced bootstrap
    if nboot and build_choices_set.choices is None:
        ret2 = []
        for i in range(nboot):
            for j in ret:
                ret2.append(j)
        np.random.shuffle(ret2)
        # make a generator so we don't need to track the index
        build_choices_set.choices = (n for n in ret2)
    return ret
build_choices_set.choices = None

def allzero(arr):
    """Test if all zero"""
    assert np.allclose(arr, 0.0, rtol=1e-14),\
        " zero:\n"+str(arr)

if GEVP and EFF_MASS:
    
    def test_avgs(reuse_inv):
        """if EFF_MASS and GEVP,
        then test to make sure all the averages have been shifted
        to be constant with respect to time
        """
        avg = em.acmean(reuse_inv)
        zero = []
        for i, tavg in enumerate(avg):
            if i < test_avgs.start_index:
                continue
            if i > test_avgs.stop_index:
                continue
            zero.append(test_avgs.avg[i]-tavg)
        zero = np.array(zero)
        assert zero.shape
        allzero(zero)
        if EFF_MASS and GEVP and not SYS_ENERGY_GUESS:
            allzero(avg-avg[0])
    test_avgs.avg = {}
    test_avgs.start_index = None
    test_avgs.stop_index = None

else:
    def test_avgs(_):
        """if EFF_MASS and GEVP,
        then test to make sure all the averages have been shifted
        to be constant with respect to time
        """


def bootstrap_ensemble(reuse_inv, avg, reuse_blocked):
    """Generate a bootstrapped version of the ensemble
    with replacement, then jackknife it
    """
    if latfit.config.BOOTSTRAP:
        reuse_inv = np.array(copy.deepcopy(np.array(reuse_inv)))
        test_avgs(reuse_inv)
        retblk = np.zeros(reuse_inv.shape, dtype=reuse_inv.dtype)
        block = BOOTSTRAP_BLOCK_SIZE
        assert block, str(block)
        choices = build_choices_set(block, len(reuse_inv), nboot=NBOOT)
        idx = 0
        while idx < len(reuse_inv):

            # pick random choice
            try:
                choice = next(build_choices_set.choices)
            except StopIteration:
                print("rebuilding choices for new bootstrap")
                build_choices_set.choices = None
                choices = build_choices_set(block, len(reuse_inv),
                                            nboot=NBOOT)
                choice = next(build_choices_set.choices)
            assert choice in choices, str(choice)+" "+str(choices)

            # append the configs to our boostrap container
            for j in range(block):
                retblk[idx+j] = reuse_inv[choice+j]

            # increment
            idx += block


        # check to see we've filled the whole ensemble
        for i, item in enumerate(retblk):
            assert np.all(item != 0), str(item)+" "+str(i)

        # find bootstrap average
        ret = copy.deepcopy(np.array(retblk, dtype=reuse_inv.dtype))
        mean = em.acmean(ret, axis=0)

        # jackknife in prep for covariance matrix
        if not RANDOMIZE_ENERGIES:
            ret = dojackknife(ret)
        assert np.allclose(mean, em.acmean(ret, axis=0), rtol=1e-12)
        avg = copy.deepcopy(np.asarray(avg))
        assert avg.shape[0] == mean.shape[0],\
            "time extent does not match"

        # store the bootstrap average coordinates
        # use 'avg' variable only for x (time) coordinates
        avg2 = [[avg[i][0], mean[i]] for i, _ in enumerate(mean)]
        avg = copy.deepcopy(np.array(avg2))
    else:
        # do nothing
        ret = reuse_blocked
    return ret, avg

if JACKKNIFE_BLOCK_SIZE == 1:
    def block_ensemble(_, reuse):
        """Do nothing in this case;
        we already have the blocked ensemble
        since usual jackknife has block size = 1
        """
        return reuse
else:
    def block_ensemble(num_configs, reuse, bsize=JACKKNIFE_BLOCK_SIZE):
        """Block the ensemble
        (for block jackknife)
        eliminate bsize configs
        """
        if bsize > 1:
            # original data, obtained by reversing single jackknife procedure
            assert not isinstance(reuse, dict),\
                "dict passed to ensemble blocker"
            reuse_inv = inverse_jk(reuse, num_configs)
            assert len(reuse_inv) == bsize*num_configs,\
                "array mismatch:"+str(bsize)+" "+str(num_configs)+" "+str(
                    len(reuse_inv))
            assert isinstance(bsize, int),\
                "jackknife block size should be integer"
            assert bsize > 1,\
                "jackknife block size should be greater than one"+\
                " for this setting"

            # blocked
            retblked = []
            for i in range(num_configs):
                newblk = delblock(i, reuse_inv, bsize)
                retblked.append(em.acmean(newblk, axis=0))
            assert len(retblked) == num_configs, "bug"
            ret = np.array(retblked, dtype=np.array(reuse).dtype)
            # ret = dojackknife(ret)
            assert ret.shape[1:] == reuse.shape[1:],\
                str(ret.shape)+" "+str(reuse.shape)
        else:
            ret = reuse
        return ret

def delblock(config_num, reuse_inv, bsize=JACKKNIFE_BLOCK_SIZE):
    """Delete JACKKNIFE_BLOCK_SIZE configs
    at block position config_num
    """
    assert isinstance(config_num, int) or int(config_num) == config_num
    config_num = int(config_num)
    origl = len(reuse_inv)
    ret = np.delete(reuse_inv,
                    np.arange(config_num*bsize, (config_num+1)*bsize, 1),
                    axis=0)
    if not latfit.config.BOOTSTRAP:
        assert origl-bsize == len(ret),\
            str(origl)+" "+str(bsize)+" "+str(len(ret))
    return ret
