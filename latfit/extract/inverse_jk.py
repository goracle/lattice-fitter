"""get the original data from the jackknife blocks"""
import numpy as np
from latfit.utilities import exactmean as em
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.utilities.postprod.h5jack import dojackknife
import latfit.config

def inverse_jk(reuse, num_configs=-1):
    """get the original data from the jackknife blocks
    """
    # reuse[config][time]
    if latfit.config.BOOTSTRAP or JACKKNIFE_BLOCK_SIZE > 1:
        num_configs = len(reuse)
    assert num_configs == len(reuse)
    if num_configs < 0:
        num = len(reuse)-1
    else:
        num = num_configs-1
    reuse_inv = em.acsum(reuse, axis=0)-num*reuse
    assert np.allclose(dojackknife(reuse_inv), reuse, rtol=1e-12)
    return reuse_inv
