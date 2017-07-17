"""Gets rid of some configs in extracted jackknife blocks"""
import sys
import numpy as np

from latfit.config import ELIM_JKCONF_LIST
from latfit.config import JACKKNIFE

def elim_jkconfigs(jkblk):
    """Takes a jackknife block as an argument, eliminates configs
    corresponding to ELIM_JKCONF_LIST, then returns the new jackknife block.
    """
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to eliminate configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    if not isinstance(ELIM_JKCONF_LIST, list) and not isinstance(
            ELIM_JKCONF_LIST, np.ndarray) and not isinstance(
                ELIM_JKCONF_LIST[0], int):
        print('***ERROR***')
        print("Not eliminating any configs because of")
        print("misconfigured list of configs to eliminate.")
        print("Check config and rerun.")
        sys.exit(1)
    else:
        num_configs = len(jkblk)
        k = len(ELIM_JKCONF_LIST)
        if k == 0:
            new_jkblk = jkblk
        else:
            skip_sum = np.sum([jkblk[skip]
                               for skip in ELIM_JKCONF_LIST], axis=0)
            sum_blk = np.sum(jkblk, axis=0)
            new_jkblk = 1.0/(num_configs-1-k)*((num_configs-1)*(np.delete(
                jkblk, ELIM_JKCONF_LIST, axis=0)+skip_sum)-k*sum_blk)
    return new_jkblk
