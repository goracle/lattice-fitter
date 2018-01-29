"""Gets rid of some configs in extracted jackknife blocks"""
import sys
import numpy as np

from latfit.config import ELIM_JKCONF_LIST
from latfit.config import JACKKNIFE


def elim_jkconfigs(jkblk, elim_list=None):
    """Takes a jackknife block as an argument, eliminates configs
    corresponding to ELIM_JKCONF_LIST, then returns the new jackknife block.
    """
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to eliminate configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    try:
        if not elim_list:
            elim_list = tuple(ELIM_JKCONF_LIST)
    except (NameError, TypeError):
        print("***ERROR***")
        print("Not eliminating any configs because of misconfigured")
        print("list of configs to elimiante.")
        print("Check config and rerun.")
        sys.exit(1)
    num_configs = len(jkblk)
    k_elim = len(elim_list)
    if k_elim == 0:
        new_jkblk = jkblk
    else:
        skip_sum = np.sum([jkblk[skip]
                           for skip in elim_list], axis=0)
        sum_blk = np.sum(jkblk, axis=0)
        new_jkblk = 1.0/(num_configs-1-k_elim)*((
            num_configs-1)*(np.delete(
                jkblk, elim_list, axis=0)+skip_sum)-k_elim*sum_blk)
    return new_jkblk
