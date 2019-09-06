"""Block the ensemble by dropping JACKKNIFE_BLOCK_SIZE configs"""

import sys
import numpy as np
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.config import BOOTSTRAP
from latfit.extract.inverse_jk import inverse_jk
from latfit.utilities import exactmean as em
from latfit.utilities.h5jack import dojackknife
import latfit.config

def bootstrap_ensemble(reuse_inv, reuse_blocked):
    """Generate a bootstrapped version of the ensemble
    with replacement, then jackknife it
    """
    if latfit.config.BOOTSTRAP:
        choices = list(range(len(reuse_inv)))
        retblk = []
        for i in choices:
            elem = reuse_inv[np.random.choice(choices)]
            elem += CONST_SHIFT
            retblk.append(elem)
        ret = np.array(retblk, dtype=reuse.dtype)
        ret = dojackknife(ret)
    else:
        ret = reuse_blocked
    return ret

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
        reuse_inv = inverse_jk(reuse, num_configs)
        assert len(reuse_inv) == bsize*num_configs, "array mismatch"
        assert isinstance(bsize, int), "jackknife block size should be integer"
        assert bsize > 1,\
            "jackknife block size should be greater than one for this setting"

        # blocked
        retblked = []
        for i in range(num_configs):
            newblk = delblock(i, reuse_inv)
            retblked.append(em.acmean(newblk, axis=0))
        assert len(retblked) == num_configs, "bug"
        ret = np.array(retblked, dtype=reuse.dtype)
        assert ret.shape == reuse.shape, "reuse blocked size != reuse size"
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

