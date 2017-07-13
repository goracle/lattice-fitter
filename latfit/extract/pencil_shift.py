"""arXiv:1010.0202 see eqn. 10"""
import numpy as np

from latfit.extract.gevp_getfiles_onetime import gevp_getfiles_onetime

from latfit.config import PENCIL_SHIFT
from latfit.config import NUM_PENCILS
from latfit.config import GEVP_DIRS

DIMOPS = len(GEVP_DIRS)
def pencil_shift_lhs(time, xstep):
    """pencil shift the lhs of the gevp"""
    lhs_penarr = np.array([0])
    lhs_files = np.zeros((NUM_PENCILS+1, DIMOPS, DIMOPS), dtype=object)
    lhs_files[0] = gevp_getfiles_onetime(time)
    for i in range(NUM_PENCILS):
        lhs_files[i+1] = gevp_getfiles_onetime(time+(i+1)*PENCIL_SHIFT*xstep)
        lhs_penarr = np.array(np.append(lhs_penarr, lhs_penarr+1))
    lhs_files_arr = lhs_files[0]
    for i in lhs_penarr:
        if i == 0:
            continue
        lhs_files_arr = np.append(lhs_files_arr, lhs_files[i], axis=1)
    files = lhs_files_arr
    for _ in range(NUM_PENCILS):
        files = np.append(files, lhs_files_arr, axis=0)
    return files

def pencil_shift_rhs(time2, xstep):
    """pencil shift the rhs of the gevp"""
    rhs_penarr = np.array([0])
    rhs_files = np.zeros((NUM_PENCILS+1, DIMOPS, DIMOPS), dtype=object)
    rhs_files[0] = gevp_getfiles_onetime(time2)
    for i in range(NUM_PENCILS):
        rhs_files[i+1] = gevp_getfiles_onetime(time2-(i+1)*PENCIL_SHIFT*xstep)
        rhs_penarr = np.array(np.append(rhs_penarr, rhs_penarr+1))
    rhs_files_arr = rhs_files[0]
    for j in rhs_penarr:
        if j == 0:
            continue
        rhs_files_arr = np.append(rhs_files_arr, rhs_files[j], axis=0)
    files2 = rhs_files_arr
    for _ in range(NUM_PENCILS):
        files2 = np.append(files2, rhs_files_arr, axis=1)
    return files2
