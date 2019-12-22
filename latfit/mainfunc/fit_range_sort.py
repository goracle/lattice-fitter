"""Utilities to sort fit ranges, skip fit ranges"""
import mpi4py
from mpi4py import MPI
import numpy as np
import latfit.analysis.sortfit as sortfit
from latfit.config import ISOSPIN, GEVP, MAX_RESULTS
from latfit.config import SKIP_LARGE_ERRORS, ERR_CUT
from latfit.config import ONLY_SMALL_FIT_RANGES, VERBOSE
from latfit.config import MULT, BIASED_SPEEDUP, MAX_ITER
import latfit.config
from latfit.singlefit import singlefit
import latfit.singlefit

assert not BIASED_SPEEDUP

MPIRANK = MPI.COMM_WORLD.rank
#MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def augment_excl(*xargs, **kwargs):
    """dummy"""
    assert None
    if xargs or kwargs:
        pass
    return []

@PROFILE
def skip_large_errors(result_param, param_err):
    """Skip on parameter errors greater than 100%
    (fit range is too noisy)
    return a bool if we should skip this fit range
    """
    ret = False
    result_param = np.asarray(result_param)
    param_err = np.asarray(param_err)
    if result_param.shape:
        for i, j in zip(result_param, param_err):
            assert j >= 0, "negative error found:"+\
                str(result_param)+" "+str(param_err)
            if j:
                ret = abs(i/j) < 1
            else:
                ret = False
    return ret if SKIP_LARGE_ERRORS else False




def fit_range_combos(meta, plotdata):
    """Generate fit range combinations"""
    # generate all possible points excluded from fit range
    prod, sampler = meta.generate_combinations()

    # length of possibilities is useful to know,
    # update powerset if brute force solution is possible
    prod = meta.length_fit(prod, sampler)

    tsorted = get_tsorted(plotdata)
    sorted_fit_ranges = sort_fit_ranges(meta, tsorted, sampler)
    return prod, sorted_fit_ranges

@PROFILE
def exitp(meta, min_arr, overfit_arr, idx):
    """Test to exit the fit range loop"""
    ret = False
    if meta.skip_loop():
        print("skipping loop")
        ret = True

    if not ret and meta.random_fit:
        if len(min_arr) >= MAX_RESULTS or (
                len(overfit_arr) >= MAX_RESULTS
                and not min_arr):
            ret = True
            print("a reasonably large set of indices"+\
                " has been checked, exiting fit range loop."+\
                " (number of fit ranges checked:"+str(idx+1)+")")
            print("rank :", MPIRANK, "exiting fit loop")
    if not len(min_arr) + len(overfit_arr) and idx >= MAX_ITER:
        print("Maximum iteration count", MAX_ITER,
              "exceeded with no results")
        print("rank :", MPIRANK, "exiting fit loop")
        ret = True
    elif idx >= 10*MAX_ITER:
        print("Maximum iteration count * 10", 10 * MAX_ITER, "exceeded.")
        print("rank :", MPIRANK, "exiting fit loop")
        ret = True
    return ret
@PROFILE
def keyexcl(excl):
    """Make a unique id for a set of excluded points"""
    return str(list(excl))



@PROFILE
def get_one_fit_range(meta, prod, idx, samp_mult, checked):
    """Choose one fit range from all combinations"""
    key = None
    if not meta.random_fit:
        excl = prod[idx]
    else: # large fit range, try to get lucky
        if idx == 0:
            excl = latfit.config.FIT_EXCL
        else:
            excl = [np.random.choice(
                samp_mult[i][1], p=samp_mult[i][0])
                    for i in range(MULT)]
    # add user info
    excl = augment_excl([list(j) for j in excl])

    key = keyexcl(excl)
    ret = None
    if key in checked:
        if VERBOSE:
            print("key checked, continuing:", key)
        ret = None
    else:
        ret = excl
        checked.add(key)
    return ret, checked

# generate all possible points excluded from fit range

@PROFILE
def sort_fit_ranges(meta, tsorted, sampler):
    """Sort fit ranges by likelihood of success
    (if bias this introduces is not an issue)"""
    samp_mult = []
    if meta.random_fit:
        # go in a random order if lenprod is small
        # (biased by how likely fit will succeed),
        for i in range(MULT):
            #if MULT == 1:
            #    break
            if i == 0 and MPIRANK == 0 and BIASED_SPEEDUP:
                print("Setting up biased sorting of"+\
                        " (random) fit ranges")
            if BIASED_SPEEDUP:
                probs, sampi = sortfit.sample_norms(
                    sampler, tsorted[i], meta.lenfit)
            else:
                probs = None
                sampi = sorted(list(sampler))
            samp_mult.append([probs, sampi])
    else:
        for i in range(MULT):
            if meta.lenprod == 1:
                break
            #if MULT == 1 or lenprod == 1:
            #    break
            #if i == 0 and MPIRANK == 0:
            if not i:
                print("Setting up sorting of exhaustive "+\
                        "list of fit ranges")
            sampi = sortfit.sortcombinations(
                sampler, tsorted[i], meta.lenfit)
            samp_mult.append(sampi)
    return samp_mult

@PROFILE
def get_tsorted(plotdata):
    """Get list of best times (most likely to yield good fits)"""
    tsorted = []
    for i in range(MULT):
        #if MULT == 1:
        #    break
        if not i:
            if BIASED_SPEEDUP:
                print("Finding best times ("+\
                      "most likely to give small chi^2 (t^2) contributions)")
        if MULT > 1 or GEVP:
            coords = np.array([j[i] for j in plotdata.coords[:, 1]])
        else:
            coords = np.asarray(plotdata.coords[:, 1])
        times = np.array(list(plotdata.coords[:, 0]))
        if MULT > 1 or GEVP:
            tsorted.append(sortfit.best_times(
                coords, plotdata.cov[:, :, i, i], i, times))
        else:
            tsorted.append(
                sortfit.best_times(coords, plotdata.cov, 0, times))
    return tsorted
