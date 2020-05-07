"""formatted print functions used in main()"""
import numpy as np
import mpi4py
from mpi4py import MPI
import gvar
from latfit.config import MULT, CALC_PHASE_SHIFT
from latfit.config import ALTERNATIVE_PARALLELIZATION

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

DOWRITE = ALTERNATIVE_PARALLELIZATION and not MPIRANK\
    or not ALTERNATIVE_PARALLELIZATION


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def print_phaseshift(result_min):
    """Print the phase shift info from a single fit range"""
    for i in range(MULT):
        if CALC_PHASE_SHIFT and DOWRITE:
            print("phase shift of state #")
            if np.isreal(result_min.phase_shift.val[i]):
                print(i, gvar.gvar(
                    result_min.phase_shift.val[i],
                    result_min.phase_shift.err[i]))
            else:
                # temperr = result_min.phase_shift.err[i]
                try:
                    assert np.isreal(result_min.phase_shift.err[i]),\
                        "phase shift error is not real"
                except AssertionError:
                    pass
                    #temperr = np.imag(result_min.phase_shift.err[i])
                print(result_min.phase_shift.val[i], "+/-")
                print(result_min.phase_shift.err[i])
                #print(i, gvar.gvar(
                #          np.imag(result_min.phase_shift.val[i]),
                #          temperr), 'j')

@PROFILE
def inverse_excl(meta, excl):
    """Get the included fit points from excluded points"""
    full = meta.actual_range()
    ret = [np.array(full) for _ in range(len(excl))]
    for idx, excldim in enumerate(excl):
        try:
            inds = [int(full.index(i)) for i in excldim]
        except ValueError:
            print("excluded point(s) is(are) not in fit range.")
            inds = []
            for j in excldim:
                if j not in full:
                    print("point is not in fit range:", j)
                else:
                    inds.append(int(full.index(j)))
        ret[idx] = list(np.delete(ret[idx], inds))
    return ret


@PROFILE
def print_fit_results(meta, min_arr):
    """ Print the fit results
    """
    if DOWRITE:
        print("Fit results:  pvalue, energies,",
              "err on energies, included fit points")
        res = []
        for i in min_arr:
            res.append((getattr(i[0], "pvalue").val,
                        getattr(i[0], 'energy').val,
                        i[1], inverse_excl(meta, i[2])))
        res = sorted(res, key=lambda x: x[0])
        for i in res:
            print(i)
