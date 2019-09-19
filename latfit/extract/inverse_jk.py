"""get the original data from the jackknife blocks"""
from latfit.utilities import exactmean as em
from latfit.config import JACKKNIFE_BLOCK_SIZE
import latfit.config

def inverse_jk(reuse, num_configs=-1):
    """get the original data from the jackknife blocks
    """
    # reuse[config][time]
    if latfit.config.BOOTSTRAP or JACKKNIFE_BLOCK_SIZE > 1:
        num_configs = len(reuse)
    else:
        assert num_configs == len(reuse)
    if num_configs < 0:
        num = len(reuse)-1
    else:
        num = num_configs-1
    reuse_inv = em.acsum(reuse, axis=0)-num*reuse
    return reuse_inv
