"""get the original data from the jackknife blocks"""
import numpy as np

def inverse_jk(reuse, time_range, num_configs=-1):
    """get the original data from the jackknife blocks
    first arg is block,
    then the latter two specify the dimensions of the block
    """
    #reuse[config][time]
    if num_configs < 0:
        num = len(reuse)-1
    else:
        num = num_configs-1
    reuse_inv = np.zeros((num_configs, len(time_range)))
    reuse_inv = np.sum(reuse, 0)-num*reuse
    return reuse_inv
