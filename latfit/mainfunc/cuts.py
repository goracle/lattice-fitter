"""Fit cuts"""
import mpi4py
from mpi4py import MPI

from latfit.config import VERBOSE, PVALUE_MIN, CALC_PHASE_SHIFT, MULT, NOLOOP
from latfit.config import PHASE_SHIFT_ERR_CUT, ISOSPIN, GEVP

import latfit.mainfunc.fit_range_sort as frsort

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


@PROFILE
def cutresult(result_min, min_arr, overfit_arr, param_err):
    """Check if result of fit to a
    fit range is acceptable or not, return true if not acceptable
    (result should be recorded or not)
    """
    ret = False
    if VERBOSE:
        print("p-value = ", result_min.pvalue.val, "rank:", MPIRANK)
    # reject model at 10% level
    if result_min.pvalue.val < PVALUE_MIN:
        print("Not storing result because p-value"+\
              " is below rejection threshold. number"+\
              " of non-overfit results so far =", len(min_arr))
        print("number of overfit results =", len(overfit_arr))
        ret = True

    # is this justifiable?
    if not ret and frsort.skip_large_errors(result_min.energy.val,
                                            param_err):
        print("Skipping fit range because param errors"+\
                " are greater than 100%")
        ret = True

    # is this justifiable?
    if not ret and CALC_PHASE_SHIFT and MULT > 1 and not NOLOOP:
        if any(result_min.phase_shift.err > PHASE_SHIFT_ERR_CUT):
            if all(result_min.phase_shift.err[
                    :-1] < PHASE_SHIFT_ERR_CUT):
                if VERBOSE:
                    print("warning: phase shift errors on "+\
                          "last state very large")
                ret = True if ISOSPIN == 2 and GEVP else ret
            else:
                if VERBOSE:
                    print("phase shift errors too large")
                ret = True
    return ret
