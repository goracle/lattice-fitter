"""Utilities to sort fit ranges, skip fit ranges"""
import mpi4py
from mpi4py import MPI
import numpy as np
import latfit.analysis.sortfit as sortfit
from latfit.config import ISOSPIN, GEVP, MAX_RESULTS
from latfit.config import SKIP_LARGE_ERRORS, ERR_CUT
from latfit.config import ONLY_SMALL_FIT_RANGES
from latfit.config import MULT, BIASED_SPEEDUP, MAX_ITER
import latfit.config
from latfit.mainfunc.metaclass import filter_sparse
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

def earlier(already_cut, jdx, kdx):
    """If a time slice in dimension kdx is cut,
    all the later time slices should also be cut"""
    ret = False
    for tup in already_cut:
        jinit, kinit = tup
        if kinit == kdx and jdx > jinit:
            ret = True
    return ret


def cut_on_growing_exp(meta):
    """Growing exponential is a signal for around the world contamination"""
    err = singlefit.error2
    coords = singlefit.coords_full
    assert singlefit.error2 is not None, "Bug in the acquiring error bars"
    #assert GEVP, "other versions not supported yet"+str(
    # err.shape)+" "+str(coords.shape)
    start = str(latfit.config.FIT_EXCL)
    actual_range = meta.actual_range()
    already_cut = set()
    for i, _ in enumerate(coords):
        for j, _ in enumerate(coords):
            if i >= j:
                continue
            excl_add = coords[j][0]
            actual_range = meta.actual_range()
            if excl_add not in actual_range:
                continue
            if MULT > 1:
                for k in range(len(coords[0][1])):
                    if (j, k) in already_cut:
                        continue
                    merr = max(err[i][k], err[j][k])
                    assert merr > 0, str(merr)
                    sig = np.abs(coords[i][1][k]-coords[j][1][k])/merr
                    earlier_cut = earlier(already_cut, j, k)
                    if (sig > 1.5 and coords[j][1][k] > coords[i][1][k]) or\
                    earlier_cut:
                        print("(max) err =", merr, "coords =",
                              coords[i][1][k], coords[j][1][k])
                        print("cutting dimension", k,
                              "for time slice", excl_add, "(exp grow cut)")
                        print("err/coords > diff cut =", sig)
                        latfit.config.FIT_EXCL[k].append(excl_add)
                        latfit.config.FIT_EXCL[k] = list(set(
                            latfit.config.FIT_EXCL[k]))
                        already_cut.add((j, k))
            else:
                if j in already_cut:
                    continue
                merr = max(err[i], err[j])
                if np.abs(coords[i][1]-coords[j][1])/merr > 1.5:
                    print("(max) err =", merr, "coords =",
                          coords[i][1], coords[j][1])
                    print("cutting dimension", 0, "for time slice",
                          excl_add, "(exp grow cut)")
                    print("err/coords > diff cut =", 1.5)
                    latfit.config.FIT_EXCL[0].append(excl_add)
                    latfit.config.FIT_EXCL[0] = list(set(
                        latfit.config.FIT_EXCL[0]))
                    already_cut.add(j)
    ret = start == str(latfit.config.FIT_EXCL)
    return ret

@PROFILE
def cut_on_errsize(meta):
    """Cut on the size of the error bars on individual points"""
    err = singlefit.error2
    coords = singlefit.coords_full
    assert singlefit.error2 is not None, "Bug in the acquiring error bars"
    #assert GEVP, "other versions not supported yet"+str(
    # err.shape)+" "+str(coords.shape)
    start = str(latfit.config.FIT_EXCL)
    for i, _ in enumerate(coords):
        excl_add = coords[i][0]
        actual_range = meta.actual_range()
        if excl_add not in actual_range:
            continue
        if MULT > 1:
            for j in range(len(coords[0][1])):
                if err[i][j]/coords[i][1][j] > ERR_CUT:
                    print("err =", err[i][j], "coords =", coords[i][1][j])
                    print("cutting dimension", j,
                          "for time slice", excl_add)
                    print("err/coords > ERR_CUT =", ERR_CUT)
                    latfit.config.FIT_EXCL[j].append(excl_add)
                    latfit.config.FIT_EXCL[j] = list(set(
                        latfit.config.FIT_EXCL[j]))
        else:
            if err[i]/coords[i][1] > ERR_CUT:
                print("err =", err[i], "coords =", coords[i][1])
                print("cutting dimension", 0, "for time slice", excl_add)
                print("err/coords > ERR_CUT =", ERR_CUT)
                latfit.config.FIT_EXCL[0].append(excl_add)
                latfit.config.FIT_EXCL[0] = list(set(
                    latfit.config.FIT_EXCL[0]))
    ret = start == str(latfit.config.FIT_EXCL)
    return ret

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
def excl_inrange(meta, excl):
    """Find the excluded points in the actual fit window"""
    ret = []
    fullrange = meta.actual_range()
    for _, exc in enumerate(excl):
        newexc = []
        for point in exc:
            if point in fullrange:
                newexc.append(point)
        ret.append(newexc)
    return ret

def skipped_all(meta, excl):
    """Skipped all points in a given GEVP dim"""
    ret = False
    win = meta.actual_range()
    for i in excl:
        check = []
        for j in i:
            if j in win:
                check.append(j)
        if len(check) >= len(win):
            ret = True
    return ret

def onlynpts(meta, excl, npts):
    """Fit dim contains only n points"""
    ret = False
    win = meta.actual_range()
    for i in excl:
        check = []
        for j in i:
            if j in win:
                check.append(j)
        if len(check) + 1 == len(win):
            ret = True
    return ret

@PROFILE
def toosmallp(meta, excl):
    """Skip a fit range if it has too few points"""
    ret = False
    excl = excl_inrange(meta, excl)
    # each energy should be included
    if skipped_all(meta, excl):
        print("skipped all the data points for a GEVP dim, "+\
                "so continuing.")
        ret = True

    # each fit curve should be to more than one data point
    if onlynpts(meta, excl, 1) and not ONLY_SMALL_FIT_RANGES:
        print("warning:  only one data point in fit curve")
        # ret = True

    if not ret and onlynpts(meta, excl, 2) and not ONLY_SMALL_FIT_RANGES:
        print("warning: only two data points in fit curve")
        # allow for very noisy excited states in I=0
        if not (ISOSPIN == 0 and GEVP):
            #ret = True
            pass

    #cut on arithmetic sequence
    if not ret and len(filter_sparse(
            excl, meta.fitwindow, xstep=meta.options.xstep)) != len(excl):
        print("warning:  not an arithmetic sequence")
    ret = False if ISOSPIN == 0 and GEVP else ret
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
