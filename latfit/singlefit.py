"""Standard fit branch"""
import sys
import os
from collections import namedtuple
import cloudpickle
from numpy import sqrt
import numpy as np

# package modules
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.makemin.dof_errchk import dof_errchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr
from latfit.mathfun.covinv_avg import covinv_avg
from latfit.jackknife_fit import jackknife_fit
from latfit.analysis.get_fit_params import get_fit_params
from latfit.mathfun.block_ensemble import block_ensemble
from latfit.mathfun.binconf import binconf
from latfit.utilities import exactmean as em
from latfit.analysis.errorcodes import NoConvergence, PrecisionLossError
from latfit.analysis.errorcodes import XmaxError

# import global variables
from latfit.config import FIT, NBOOT, fit_func
from latfit.config import JACKKNIFE_FIT, JACKKNIFE_BLOCK_SIZE
from latfit.config import JACKKNIFE, NOLOOP, BOOTSTRAP_PVALUES
from latfit.config import PRINT_CORR
from latfit.config import GEVP, RANDOMIZE_ENERGIES
import latfit.config
import latfit.analysis.result_min as resmin
import latfit.jackknife_fit as jack_module
import latfit.mathfun.block_ensemble as blke

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def avgtime_replace(params, avg):
    """Replace all the time slice data in the average
    with average over time"""
    tavg = em.acmean(avg, axis=0)
    assert params.dimops == len(tavg)
    for i, _ in enumerate(avg):
        avg[i] = tavg
    return avg

def randomize_data(params, reuse, reuse_blocked, coords):
    """Replace data by avg + gaussian noise"""
    if isinstance(reuse, dict):
        reuse = rearrange_reuse_dict(params, reuse)
    if isinstance(reuse_blocked, dict):
        reuse_blocked = rearrange_reuse_dict(params, reuse_blocked)
    dev = em.acstd(reuse)*np.sqrt(len(reuse)-1)*np.sqrt(len(reuse))
    avg = avgtime_replace(params, em.acmean(reuse, axis=0))
    avgblkd = avgtime_replace(params, em.acmean(reuse_blocked, axis=0))
    nconfigs = len(reuse)
    assert nconfigs == len(reuse_blocked),\
        str(nconfigs)+" "+str(len(reuse_blocked))
    for i in range(nconfigs):
        noise = np.random.normal(0, dev, avgblkd.shape)
        reuse[i] = avg + noise
        reuse_blocked[i] = avgblkd + noise
        assert np.all(reuse[i] == reuse_blocked[i]),\
            str(avg)+" "+str(avgblkd)
    new_avg = em.acmean(reuse, axis=0)
    # new_avgblked = em.acmean(reuse_blocked, axis=0)
    for i, _ in enumerate(coords):
        coords[i] = [coords[i][0], new_avg[i]]
    coords = np.array(coords)
    return reuse, reuse_blocked, coords


@PROFILE
def singlefit(input_f, fitrange, xmin, xmax, xstep):
    """Get data to fit
    and minimized params for the fit function (if we're fitting)
    """
    # test to see if file/folder exists
    inputexists(input_f)

    # process the file(s)
    if singlefit.reuse is None:
        singlefit.coords_full, singlefit.cov_full, singlefit.reuse = extract(
            input_f, xmin, xmax, xstep)
    coords_full, cov_full, reuse = singlefit.coords_full,\
        singlefit.cov_full, singlefit.reuse

    # Now that we have the data to fit, do pre-proccess it
    params = namedtuple('fit_params', ['dimops', 'num_configs',
                                       'prefactor', 'time_range'])
    params = get_fit_params(cov_full, reuse, xmin, fitrange, xstep)

    # make reuse into an array, rearrange
    reuse = rearrange_reuse_dict(params, reuse)

    # block the ensemble
    if singlefit.reuse_blocked is None or RANDOMIZE_ENERGIES:
        singlefit.reuse_blocked = block_ensemble(params.num_configs, reuse)
        chec = binconf(reuse, binnum=JACKKNIFE_BLOCK_SIZE)
        try:
            assert np.allclose(chec, singlefit.reuse_blocked, rtol=1e-14)
        except AssertionError:
            try:
                raise PrecisionLossError
            except PrecisionLossError:
                singlefit_reset()
                raise XmaxError(problemx=xmax)



    # correct covariance matrix for jackknife factor
    if singlefit.sent is None:
        cov_full *= params.prefactor
        singlefit.sent = object()

    # debug branch
    debug_print(coords_full, cov_full)

    # select subset of data for fit
    coords, cov = fit_select(coords_full, cov_full,
                             index_select(xmin, xmax, xstep,
                                          fitrange, coords_full))

    # error handling for Degrees of Freedom <= 0 (it should be > 0).
    # number of points plotted = len(cov).
    # DOF = len(cov) - START_PARAMS
    dof_errchk(len(cov_full), params.dimops)
    dof_errchk(len(cov), params.dimops)

    # we have data 6ab
    # at this point we have the covariance matrix, and coordinates

    if GEVP:
        singlefit.error2 = np.array([np.sqrt(np.diag(
            cov_full[i][i])) for i in range(len(coords_full))]) if\
            singlefit.error2 is None else singlefit.error2
        #print("(Rough) scale of errors in data points = ",
        #np.sqrt(np.diag(cov[0][0])))
    else:
        singlefit.error2 = np.array([np.sqrt(cov_full[i][i])
                                     for i in range(len(coords_full))]) if\
                                         singlefit.error2 is None else\
                                         singlefit.error2
        print("(Rough) scale of errors in data points = ", sqrt(cov[0][0]))

    if RANDOMIZE_ENERGIES:
        reuse, singlefit.reuse_blocked, coords = randomize_data(
            params, reuse, singlefit.reuse_blocked, coords)

    if FIT:
        if JACKKNIFE_FIT and JACKKNIFE == 'YES':

            # initial fit
            reset_bootstrap()
            latfit.config.BOOTSTRAP = False
            if os.path.isfile("result_min.p") and NOLOOP:
                result_min, param_err = cloudpickle.load(
                    open("result_min.p", "rb"))
            else:
                try:
                    result_min, param_err = jackknife_fit(
                        params, reuse, singlefit.reuse_blocked, coords)
                except PrecisionLossError:
                    singlefit_reset()
                    raise XmaxError(problemx=xmax)
                cloudpickle.dump((result_min, param_err),
                                 open("result_min.p", "wb"))
                if BOOTSTRAP_PVALUES:
                    result_min = bootstrap_pvalue(params, reuse,
                                                  coords, result_min)
        else:
            result_min, param_err = non_jackknife_fit(params, cov, coords)

        result_min = error_bar_scheme(result_min, fitrange, xmin, xmax)

        ret = (result_min, param_err, coords_full, cov_full)
    else:
        ret = (coords, cov)
    return ret
singlefit.reuse = None
singlefit.coords_full = None
singlefit.cov_full = None
singlefit.sent = None
singlefit.error2 = None
singlefit.reuse_blocked = None

def non_jackknife_fit(params, cov, coords):
    """Compute using a very old fit style"""
    covinv = covinv_compute(params, cov)
    result_min = mkmin(covinv, coords)
    # compute errors 8ab, print results (not needed for plot part)
    param_err = geterr(result_min, covinv, coords)
    return result_min, param_err

def covinv_compute(params, cov):
    """Compute inverse covariance matrix"""
    try:
        covinv = covinv_avg(cov, params.dimops)
    except np.linalg.linalg.LinAlgError:
        covinv = np.zeros(cov.shape)
        for i, _ in enumerate(covinv):
            for j, _ in enumerate(covinv):
                covinv[i][j] = np.nan
    return covinv

def error_bar_scheme(result_min, fitrange, xmin, xmax):
    """use a consistent error bar scheme;
    if fitrange isn't max use conventional,
    otherwise use the new double jackknife estimate
    """
    if xmin != fitrange[0] or xmax != fitrange[1]:
        try:
            result_min.misc.error_bars = None
        except AttributeError:
            pass
    return result_min

def bootstrap_pvalue(params, reuse, coords, result_min):
    """Get bootstrap p-values"""
    # fit to find the null distribution
    if result_min.misc.dof not in bootstrap_pvalue.result_minq:
        latfit.config.BOOTSTRAP = True
        set_bootstrap_shift(result_min)
        # total_configs = JACKKNIFE_BLOCK_SIZE*params.num_configs
        nconfig = int(params.num_configs)
        params.num_configs = NBOOT
        print("starting computation of null distribution from bootstrap")
        print("NBOOT =", NBOOT)
        try:
            result_minq, _ = jackknife_fit(
                params, reuse, singlefit.reuse_blocked, coords)
        except NoConvergence:
            print("minimizer failed to converge during bootstrap")
            assert None
        assert next(blke.build_choices_set.choices, None) is None
        print("done computing null dist.")
        assert result_min.misc.dof == result_minq.misc.dof
        bootstrap_pvalue.result_minq[result_min.misc.dof] = result_minq
        resmin.NULL_CHISQ_ARRS[result_min.misc.dof] = result_minq.chisq.arr
        params.num_configs = nconfig
    else:
        result_minq = bootstrap_pvalue.result_minq[result_min.misc.dof]

    # overwrite initial fit with the accurate p-value info
    result_min.pvalue.arr = resmin.chisq_arr_to_pvalue_arr(
        result_minq.misc.dof, params.num_configs,
        result_minq.chisq.arr, result_min.chisq.arr)
    result_min.pvalue.val = em.acmean(result_min.pvalue.arr)
    result_min.pvalue.err = em.acmean((
        result_min.pvalue.arr-result_min.pvalue.val)**2)
    result_min.pvalue.err *= np.sqrt((len(
        result_min.pvalue.arr)-1)/len(result_min.pvalue.arr))
    return result_min
bootstrap_pvalue.result_minq = {}


def set_bootstrap_shift(result_min):
    """Subtract any systematic difference
    to get the Null distribution for p-values
    This function informs the bootstrapping function
    of the shift
    """
    coords = singlefit.coords_full
    assert coords is not None
    shift = {}
    for i, ctime in enumerate(coords[:, 0]):
        part1 = fit_func(ctime, result_min.min_params.val)
        blke.test_avgs.avg[i] = part1
        part1 = np.array(part1, dtype=np.float128)
        part2 = coords[i][1]
        if not set_bootstrap_shift.printed:
            print("avg coords:", coords)
            set_bootstrap_shift.printed = True
        part2 = np.array(part2, dtype=np.float128)
        try:
            shift[int(ctime)] = part1 - part2
        except ValueError:
            print("could not sum part1 and part2")
            print("part1 =", part1)
            print("part2 =", part2)
            raise
    print("setting bootstrap shift to fit function with value:", shift)
    jack_module.CONST_SHIFT = shift
set_bootstrap_shift.printed = False

def reset_bootstrap():
    """Set const. shift to 0
    (for initial fit)
    and reset the index list for the bootstrap ensembles
    """
    jack_module.CONST_SHIFT = 0
    blke.build_choices_set.choices = None

def debug_print(coords_full, cov_full):
    """Debug print
    """
    if PRINT_CORR:
        print(coords_full)
        if GEVP:
            print([sqrt(np.diag(cov_full[i][i])) for i in range(
                len(cov_full))])
        else:
            print([sqrt(cov_full[i][i]) for i in range(len(cov_full))])
        sys.exit(0)

def singlefit_reset():
    """reset all the internal singlefit function variables"""
    singlefit.reuse = None
    singlefit.coords_full = None
    singlefit.cov_full = None
    singlefit.sent = None
    singlefit.error2 = None
    singlefit.reuse_blocked = None


@PROFILE
def index_select(xmin, xmax, xstep, fitrange, coords_full):
    """Get the starting and ending indices
    for the fitted subset of the data"""
    start_index = int((fitrange[0]-xmin)/xstep)
    stop_index = int(len(coords_full)-1-(xmax-fitrange[1])/xstep)
    blke.test_avgs.start_index = start_index
    blke.test_avgs.stop_index = stop_index
    return start_index, stop_index


@PROFILE
def fit_select(coords_full, cov_full, selection):
    """Select portion of data to fit with"""
    # select part of data to fit
    start_index = selection[0]
    stop_index = selection[1]
    coords = coords_full[start_index:stop_index+1]
    cov = cov_full[start_index:stop_index+1, start_index:stop_index+1]
    return coords, cov


# do this so reuse goes from reuse[time][config]
# to more convenient reuse[config][time]
@PROFILE
def rearrange_reuse_dict(params, reuse, bsize=JACKKNIFE_BLOCK_SIZE):
    """reuse = swap(reuse, 0, 1), turn it into an array
    detail:
    make reuse, the original unjackknifed data,
    into a numpy array, swap indices
    """
    total_configs = bsize*params.num_configs
    assert int(total_configs) == total_configs
    total_configs = int(total_configs)
    return np.array([[reuse[time][config]
                      for time in params.time_range]
                     for config in range(total_configs)])
