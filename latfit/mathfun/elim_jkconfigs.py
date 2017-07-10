"""Gets rid of some configs in extracted jackknife blocks"""
import sys
import numpy as np

from latfit.config import elim_jkconf_list
from latfit.config import JACKKNIFE

def elim_jkconfigs(jkblk):
    """Takes a jackknife block as an argument, eliminates configs
    corresponding to elim_jkconf_list, then returns the new jackknife block.
    """
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to eliminate configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    else:
        try:
            elim_list = tuple(elim_jkconf_list)
        except (NameError, TypeError):
            print("***ERROR***")
            print("Not eliminating any configs because of misconfigured")
            print("list of configs to elimiante.")
            print("Check config and rerun.")
            sys.exit(1)
        n_conf = len(jkblk)
        k_elim = len(elim_list)
        if k_elim == 0:
            new_jkblk = jkblk
        else:
            skip_sum = np.sum([jkblk[skip] for skip in
                               elim_list], axis=0)
            sum_blk = np.sum(jkblk, axis=0)
            new_jkblk = 1.0/(n_conf-1-k_elim)*((n_conf-1)*(np.delete(
                jkblk, elim_list, axis=0)+skip_sum)-k_elim*sum_blk)
    return new_jkblk
