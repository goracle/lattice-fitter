"""Gets rid of some configs in extracted jackknife blocks"""
import sys
import numpy as np
import accupy

ELIM_JKCONF_LIST = None
HALF = ''
try:
    from latfit.config import JACKKNIFE
except ImportError:
    JACKKNIFE = True


def elim_jkconfigs(jkblk, elim_list=None):
    """Takes a jackknife block as an argument, eliminates configs
    corresponding to ELIM_JKCONF_LIST, then returns the new jackknife block.
    """
    assert HALF, str(HALF)
    if not JACKKNIFE:
        print("***ERROR***")
        print("Attempting to eliminate configurations from jackknife blocks,")
        print("but jackknife correction to covariance matrix is not enabled.")
        sys.exit(1)
    try:
        assert elim_list or ELIM_JKCONF_LIST is not None, "missing elimination list"+str(ELIM_JKCONF_LIST)+" "+str(elim_list)
        if elim_list is None:
            elim_list = list(ELIM_JKCONF_LIST)
        if ELIM_JKCONF_LIST is not None and HALF == 'full':
            assert np.all(elim_list == ELIM_JKCONF_LIST), "list mismatch:"+str(elim_list)
    except (NameError, TypeError):
        print("***ERROR***")
        print("Not eliminating any configs because of misconfigured")
        print("list of configs to elimiante.")
        print("Check config and rerun.")
        sys.exit(1)
    num_configs = len(jkblk)
    k_elim = len(elim_list)
    if k_elim == 0 or len(jkblk) == 1:
        new_jkblk = jkblk
    else:

        # each of the unwanted configs appears k_elim-1 times in skip_sum
        # each of the wanted configs appears k_elim times in skip_sum
        skip_sum = accupy.fsum([jkblk[skip]
                           for skip in elim_list])

        sum_blk = accupy.fsum(jkblk) # same as sum over original set, no norm

        # delete the blocks corresponding to the unwanted configs
        inner = np.delete(jkblk, elim_list, axis=0)+skip_sum

        # check for precision loss, plausibly
        assert np.allclose(inner - skip_sum,
                           np.delete(jkblk, elim_list, axis=0), rtol=1e-12)

        # unormalize; only sums over configs now
        inner *= (num_configs-1)
        
        # subtract the wanted configs we added in skip_sum
        # subtract the unwanted configs we added in skip_sum, plus one extra
        # this extra being the copy which shows up in the non-deleted blocks
        final_diff = inner-k_elim*sum_blk

        # check for precision loss
        assert np.allclose(final_diff + k_elim*sum_blk, inner, rtol=1e-12)

        # normalize
        new_jkblk = final_diff/(num_configs-1-k_elim)

    return new_jkblk
