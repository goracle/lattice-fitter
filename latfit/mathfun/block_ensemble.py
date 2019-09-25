"""Block the ensemble by dropping JACKKNIFE_BLOCK_SIZE configs"""

import sys
import copy
import numpy as np
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.extract.inverse_jk import inverse_jk
from latfit.utilities import exactmean as em
from latfit.utilities.postprod.h5jack import dojackknife
from latfit.utilities import exactmean as em
from latfit.config import BOOTSTRAP_BLOCK_SIZE
import latfit.config

print("Using bootstrap block size:", BOOTSTRAP_BLOCK_SIZE)

def bootstrap_ensemble(reuse_inv, avg, reuse_blocked):
    """Generate a bootstrapped version of the ensemble
    with replacement, then jackknife it
    """
    if latfit.config.BOOTSTRAP:
        reuse_inv = np.array(copy.deepcopy(np.array(reuse_inv)))
        reuse_inv_mean = em.acmean(reuse_inv, axis=0)
        choices = list(range(len(reuse_inv)))
        np.random.shuffle(choices)
        retblk = np.zeros(reuse_inv.shape, dtype=reuse_inv.dtype)
        idx = 0
        for _ in choices:
            block = BOOTSTRAP_BLOCK_SIZE
            if idx+block > len(reuse_inv):
                block = len(reuse_inv)-idx
            #choice = np.random.randint(0, len(reuse_inv)-block)
            if block == len(reuse_inv):
                choice = 0
            else:
                choice = np.random.randint(0, len(reuse_inv)-block+1)
            for j in range(block):
                retblk[idx+j] = reuse_inv[choice+j]
            idx += block
        for i, item in enumerate(retblk):
            assert np.all(item != 0), str(item)+" "+str(i)
        ret = copy.deepcopy(np.array(retblk, dtype=reuse_inv.dtype))
        mean = em.acmean(ret, axis=0)
        ret = dojackknife(ret)
        assert np.allclose(mean, em.acmean(ret, axis=0), rtol=1e-12)
        #assert np.mean(np.delete(reuse_inv, config_num, axis=0), axis=0) == avg[:,1], str(avg[:,1])+" "+str(np.mean(np.delete(reuse_inv, config_num, axis=0), axis=0))
        avg = copy.deepcopy(np.asarray(avg))
        assert avg.shape[0] == mean.shape[0], "time extent does not match"
        avg2 = [[avg[i][0], mean[i]] for i, _ in enumerate(mean)]
        avg = copy.deepcopy(np.array(avg2))
    else:
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
        # original data, obtained by reversing single jackknife procedure
        assert not isinstance(reuse, dict), "dict passed to ensemble blocker"
        reuse_inv = inverse_jk(reuse, num_configs)
        assert len(reuse_inv) == bsize*num_configs, "array mismatch:"+str(
            bsize)+" "+str(num_configs)+" "+str(len(reuse_inv))
        assert isinstance(bsize, int),\
            "jackknife block size should be integer"
        assert bsize > 1,\
            "jackknife block size should be greater than one for this setting"

        # blocked
        retblked = []
        for i in range(num_configs):
            newblk = delblock(i, reuse_inv, bsize)
            retblked.append(em.acmean(newblk, axis=0))
        assert len(retblked) == num_configs, "bug"
        ret = np.array(retblked, dtype=reuse.dtype)
        ret = dojackknife(ret)
        assert ret.shape[1:] == reuse.shape[1:],\
            str(ret.shape)+" "+str(reuse.shape)
        return retblked

def delblock(config_num, reuse_inv, bsize=JACKKNIFE_BLOCK_SIZE):
    """Delete JACKKNIFE_BLOCK_SIZE configs
    at block position config_num
    """
    assert isinstance(config_num, int) or int(config_num) == config_num
    config_num = int(config_num)
    ret = np.delete(reuse_inv,
                    np.arange(config_num*bsize, (config_num+1)*bsize, 1),
                    axis=0)
    return ret
