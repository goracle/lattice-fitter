import numpy as np
import sys

from latfit.config import ELIM_JKCONF_LIST
from latfit.config import JACKKNIFE

def elim_jkconfigs(jkblk):
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to eliminate configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    if not type(ELIM_JKCONF_LIST) is list and not type(ELIM_JKCONF_LIST) is np.ndarray and not type(ELIM_JKCONF_LIST[0]) is int:
        print('***ERROR***')
        print("Not eliminating any configs because of misconfigured list of configs to elimiante.")
        print("Check config and rerun.")
        sys.exit(1)
    else:
        n=len(jkblk)
        k=len(ELIM_JKCONF_LIST)
        if k==0:
            new_jkblk=jkblk
        else:
            skip_sum=np.sum([jkblk[skip] for skip in ELIM_JKCONF_LIST],axis=0)
            sum_blk=np.sum(jkblk,axis=0)
            new_jkblk=1.0/(n-1-k)*((n-1)*(np.delete(jkblk,ELIM_JKCONF_LIST,axis=0)+skip_sum)-k*sum_blk)
    return new_jkblk
            
