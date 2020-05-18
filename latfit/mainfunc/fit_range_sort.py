"""Utilities to sort fit ranges, skip fit ranges"""
import mpi4py
from mpi4py import MPI
import numpy as np
import latfit.analysis.sortfit as sortfit
from latfit.config import GEVP, MAX_RESULTS
from latfit.config import SKIP_LARGE_ERRORS
from latfit.config import VERBOSE, NOLOOP
from latfit.config import ALTERNATIVE_PARALLELIZATION
from latfit.config import MULT, BIASED_SPEEDUP, MAX_ITER
import latfit.config
import latfit.singlefit

assert not BIASED_SPEEDUP

MPIRANK = MPI.COMM_WORLD.rank
#MPISIZE = MPI.COMM_WORLD.Get_size()
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
def exitp(meta, min_arr, overfit_arr, idx, noprint=False):
    """Test to exit the fit range loop"""
    ret = False
    if meta.skip_loop():
        if VERBOSE and not noprint:
            print("skipping loop")
        ret = True

    if not ret and meta.random_fit:
        if len(min_arr) >= MAX_RESULTS or (
                len(overfit_arr) >= MAX_RESULTS
                and not min_arr):
            ret = True
            if not noprint:
                print("a reasonably large set of indices",
                      "has been checked, exiting fit range loop.",
                      "(number of fit ranges checked:"+str(idx+1)+")")
                print("rank :", MPIRANK, "exiting fit loop")

    # check to see if max iter count and no results
    mix = get_chunked_max(3)
    assert isinstance(mix, int), mix
    if not len(min_arr) + len(overfit_arr) and idx >= mix and meta.random_fit:
        if not noprint:
            print("Maximum iteration count", mix,
                  "exceeded with no results")
            print("rank :", MPIRANK, "exiting fit loop")
        ret = True
    # check on loop progress in 6 chunks; if not making progress, exit.
    for chunk in range(6):
        if ret:
            break
        mix = get_chunked_max(chunk)
        thr, rstr = threshold(chunk)
        if idx >= mix and len(min_arr) < thr and meta.random_fit:
            if not noprint:
                print("Maximum iteration count", mix, "exceeded.")
                print("and results needed are <", rstr, "of", MAX_RESULTS)
                print("results:", len(min_arr))
                print("rank :", MPIRANK, "exiting fit loop")
            ret = True
    # check to see if 4*max iter count (absolute upper bound)
    mix = get_chunked_max(6)
    if idx >= mix:
        if not noprint:
            print("Maximum iteration count * 4", mix, "exceeded.")
            print("results:", len(min_arr))
            print("rank :", MPIRANK, "exiting fit loop")
        ret = True
    return ret

@PROFILE
def keyexcl(excl):
    """Make a unique id for a set of excluded points"""
    return str(list(excl))



@PROFILE
def get_one_fit_range(meta, idx, checked, combo_data):
    """Choose one fit range from all combinations"""
    prod, samp_mult = combo_data
    key = None
    if not meta.random_fit:
        excl = prod[idx]
        excl = list(excl)
    else: # large fit range, try to get lucky
        if idx == 0:
            excl = list(latfit.config.FIT_EXCL)
        else:
            excl = tuple(np.random.choice(
                samp_mult[i][1], p=samp_mult[i][0])
                         for i in range(MULT))
    # add user info
    excl = augment_excl([list(j) for j in excl])
    excl = list(excl)

    key = keyexcl(excl)
    ret = None
    if key in checked:
        if VERBOSE:
            print("key checked, continuing:", key)
        ret = None
    else:
        ret = list(excl)
        checked.add(key)
    return ret, checked

def setup_checked(fit_range_init):
    """Init the set of checked fit ranges"""
    ret = set()
    ret.add(fit_range_init)
    return ret

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
            if not i and VERBOSE:
                print("Setting up sorting of exhaustive "+\
                        "list of fit ranges")
            sampi = sortfit.sortcombinations(
                sampler, tsorted[i], meta.lenfit)
            samp_mult.append(sampi)
    return samp_mult

def combo_data_to_fit_ranges(meta, combo_data, chunk, checked=None):
    """Get sets of fit ranges;
    chunked based on progress points
    """
    if checked is None:
        checked = []

    # get maximum number of fit ranges
    # "mix"
    mix = meta.lenprod
    if meta.random_fit and not NOLOOP:
        mix2, chunk = get_chunked_max(
            chunk, procs=meta.options.procs, start=len(checked))
        mix2 = np.ceil(mix2)
        mix = min(mix, mix2)
    assert int(mix) == mix, (mix, meta.lenprod)
    mix = int(mix)

    ret = []
    for idx in range(mix-len(checked)):
        excl, checked = get_one_fit_range(
            meta, idx, checked, combo_data)
        if excl is not None:
            excl = list(excl)
            if latfit.singlefit.toosmallp(meta, excl):
                excl = None
        ret.append(excl)
    return ret, chunk


def set_fit_range(meta, excl):
    """check fit range; then set the fit range globally"""
    # get one fit range, check it
    if excl is not None:
        excl = list(excl)
    skip = False
    if excl is None:
        skip = True
    if not skip:
        if latfit.singlefit.toosmallp(meta, excl):
            assert None, "bug; inconsistent"
            skip = True
        if not skip:
            # update global info about excluded points
            latfit.config.FIT_EXCL = list(excl)
    return skip


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

def threshold(idx):
    """Number of results needed at
    threshold indexed by idx:
    0: MAX_RESULTS/16
    1: MAX_RESULTS/8
    2: MAX_RESULTS/4
    3: MAX_RESULTS/2
    4: MAX_RESULTS*2/3
    5: MAX_RESULTS*3/4

    These levels are arbitrary,
    and have not been tuned
    """
    thr = MAX_RESULTS
    rstr = ""
    if not idx:
        ret = thr/16
        rstr = "1/16"
    elif idx == 1:
        ret = thr/8
        rstr = "1/8"
    elif idx == 2:
        ret = thr/4
        rstr = "1/4"
    elif idx == 3:
        ret = thr/2
        rstr = "1/2"
    elif idx == 4:
        ret = thr*2/3
        rstr = "2/3"
    elif idx == 5:
        ret = thr*3/4
        rstr = "3/4"
    else:
        assert None, (
            "bad threshold index specified:", idx)
    return ret, rstr


def get_chunked_max(idx, procs=None, start=0):
    """
    get upper bound based on chunk index
    0: MAX_ITER/8
    1: MAX_ITER/4
    2: MAX_ITER/2
    3: MAX_ITER
    4: MAX_ITER*2
    5: MAX_ITER*3
    6: MAX_ITER*4

    These levels are arbitrary,
    and have not been tuned
    """
    mix = MAX_ITER
    ret = 0
    # number of processes to split fit ranges over
    procs = np.inf if procs is None else procs
    # 100 fit ranges for each proc seems a good amount
    # of work to do
    while ret-start < procs:
        if not idx:
            ret = mix/8
        elif idx == 1:
            ret = mix/4
        elif idx == 2:
            ret = mix/2
        elif idx == 3:
            ret = mix
        elif idx == 4:
            ret = mix*2
        elif idx == 5:
            ret = mix*3
        elif idx == 6:
            ret = mix*4
        elif idx > 6 and procs < np.inf:
            break
        else:
            assert None, (
                "bad max iter chunk index specified:", idx)
        if procs == np.inf:
            break
        idx += 1
    ret = (ret, idx) if procs != np.inf else ret
    return ret
