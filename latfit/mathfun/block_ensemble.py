"""Block the ensemble by dropping JACKKNIFE_BLOCK_SIZE configs"""

import numpy as np
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.extract.inverse_jk import inverse_jk
from latfit.utilities import exactmean as em

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
        assert bsize > 1, "jackknife block size should be greater than one for this setting"

        # blocked
        retblked = []
        for i in in range(num_configs):
            newblk = delblock(i, reuse_inv)
            retblked.append(em.acmean(newblk, axis=0))
        assert len(retblked) == num_configs, "bug"
        ret = np.array(retblked, dtype=reuse.dtype)
        return retblked

def delblock(config_num, reuse_inv, bsize=JACKKNIFE_BLOCK_SIZE):
    """Delete JACKKNIFE_BLOCK_SIZE configs
    at block position config_num
    """
    assert isinstance(config_num, int)
    ret = np.delete(reuse_inv,
                    np.arange(config_num*bsize, (config_num+1)*bsize, 1),
                    axis=0)
    return ret

