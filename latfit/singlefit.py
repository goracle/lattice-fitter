"""Standard fit branch"""
import sys
import os
from collections import namedtuple
import mpi4py
from mpi4py import MPI
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
from latfit.utilities.tuplize import list_mat, tupl_mat
from latfit.analysis.errorcodes import NoConvergence, PrecisionLossError
from latfit.analysis.errorcodes import XmaxError, FitFail
from latfit.mainfunc.metaclass import filter_sparse

# import global variables
from latfit.config import FIT, NBOOT, fit_func, ONLY_SMALL_FIT_RANGES
from latfit.config import JACKKNIFE_FIT, JACKKNIFE_BLOCK_SIZE
from latfit.config import JACKKNIFE, NOLOOP, BOOTSTRAP_PVALUES
from latfit.config import PRINT_CORR, MULT, ERR_CUT, ISOSPIN
from latfit.config import GEVP, RANDOMIZE_ENERGIES, VERBOSE, DIMSELECT
import latfit.config
import latfit.analysis.result_min as resmin
import latfit.jackknife_fit as jack_module
import latfit.mathfun.block_ensemble as blke

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
def singlefit(meta, input_f):
    """Get data to fit
    and minimized params for the fit function (if we're fitting)
    """
    fitwindow, xmin, xmax, xstep = meta.fitwindow, meta.options.xmin,\
        meta.options.xmax, meta.options.xstep
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
    params = get_fit_params(cov_full, reuse, xmin, fitwindow, xstep)

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
                                          fitwindow, coords_full))

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

    fiduc_point_cuts(meta)
    print("new excl:", latfit.config.FIT_EXCL)
    if not toosmallp(meta, latfit.config.FIT_EXCL) and FIT:
        if JACKKNIFE_FIT and JACKKNIFE == 'YES':

            # initial fit
            reset_bootstrap()
            if os.path.isfile("result_min.p") and NOLOOP and DIMSELECT is None:
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

        result_min = error_bar_scheme(result_min, fitwindow, xmin, xmax)

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

def fiduc_point_cuts(meta):
    """Perform fiducial cuts on individual effective mass points"""
    if FIT:
        samerange = cut_on_errsize(meta)
        samerange = cut_on_growing_exp(meta) and samerange
        if not samerange:
            if toosmallp(meta, latfit.config.FIT_EXCL):
                if VERBOSE:
                    print("fiducial cuts leave nothing to fit.  rank:", MPIRANK)
                raise FitFail

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

def error_bar_scheme(result_min, fitwindow, xmin, xmax):
    """use a consistent error bar scheme;
    if fit window isn't max use conventional,
    otherwise use the new double jackknife estimate
    """
    if xmin != fitwindow[0] or xmax != fitwindow[1]:
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
    bootstrap_pvalue.result_minq = {}
    resmin.NULL_CHISQ_ARRS = {}
    latfit.config.BOOTSTRAP = False

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
def index_select(xmin, xmax, xstep, fitwindow, coords_full):
    """Get the starting and ending indices
    for the fitted subset of the data"""
    start_index = int((fitwindow[0]-xmin)/xstep)
    stop_index = int(len(coords_full)-1-(xmax-fitwindow[1])/xstep)
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

def earlier(already_cut, jdx, kdx):
    """If a time slice in dimension kdx is cut,
    all the later time slices should also be cut"""
    ret = False
    for tup in already_cut:
        jinit, kinit = tup
        if kinit == kdx and jdx > jinit:
            ret = True
    return ret


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

@PROFILE
def cut_on_growing_exp(meta):
    """Growing exponential is a signal for around the world contamination"""
    err = singlefit.error2
    coords = singlefit.coords_full
    assert singlefit.error2 is not None, "Bug in the acquiring error bars"
    #assert GEVP, "other versions not supported yet"+str(
    # err.shape)+" "+str(coords.shape)
    start = str(latfit.config.FIT_EXCL)
    excl = list_mat(latfit.config.FIT_EXCL)
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
                        if VERBOSE:
                            print("(max) err =", merr, "coords =",
                                  coords[i][1][k], coords[j][1][k])
                            print("cutting dimension", k,
                                  "for time slice", excl_add,
                                  "(exp grow cut)")
                            print("time slices:", coords[i][0], coords[j][0])
                            print("err/coords > diff cut =", sig)
                        excl[k].append(excl_add)
                        excl[k] = list(set(excl[k]))
                        already_cut.add((j, k))
            else:
                if j in already_cut:
                    continue
                merr = max(err[i], err[j])
                if np.abs(coords[i][1]-coords[j][1])/merr > 1.5:
                    if VERBOSE:
                        print("(max) err =", merr, "coords =",
                              coords[i][1], coords[j][1])
                        print("cutting dimension", 0, "for time slice",
                              excl_add, "(exp grow cut)")
                        print("err/coords > diff cut =", 1.5)
                    excl[0].append(excl_add)
                    excl[0] = list(set(excl[0]))
                    already_cut.add(j)
    excl = tupl_mat(excl)
    ret = start == str(excl)
    latfit.config.FIT_EXCL = excl
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
    excl = list_mat(latfit.config.FIT_EXCL)
    for i, _ in enumerate(coords):
        excl_add = coords[i][0]
        actual_range = meta.actual_range()
        if excl_add not in actual_range:
            continue
        if MULT > 1:
            for j in range(len(coords[0][1])):
                if err[i][j]/coords[i][1][j] > ERR_CUT:
                    if VERBOSE:
                        print("err =", err[i][j], "coords =",
                              coords[i][1][j])
                        print("cutting dimension", j,
                              "for time slice", excl_add)
                        print("err/coords > ERR_CUT =", ERR_CUT)
                    excl[j].append(excl_add)
                    excl[j] = list(set(excl[j]))
        else:
            if err[i]/coords[i][1] > ERR_CUT:
                if VERBOSE:
                    print("err =", err[i], "coords =", coords[i][1])
                    print("cutting dimension", 0, "for time slice", excl_add)
                    print("err/coords > ERR_CUT =", ERR_CUT)
                excl[0].append(excl_add)
                excl[0] = list(set(excl[0]))
    excl = tupl_mat(excl)
    ret = start == str(excl)
    latfit.config.FIT_EXCL = excl
    return ret

@PROFILE
def toosmallp(meta, excl):
    """Skip a fit range if it has too few points"""
    ret = False
    excl = excl_inrange(meta, excl)
    # each energy should be included
    if skipped_all(meta, excl):
        if VERBOSE:
            print("skipped all the data points for a GEVP dim, "+\
                  "so continuing.")
        ret = True

    # each fit curve should be to more than one data point
    if onlynpts(meta, excl, 1) and not ONLY_SMALL_FIT_RANGES:
        if not (ISOSPIN == 0 and GEVP):
            if VERBOSE:
                print("skip: only one data point in fit curve")
            ret = True
        else:
            print("warning: only one data point in fit curve")

    if not ret and onlynpts(meta, excl, 2) and not ONLY_SMALL_FIT_RANGES:
        # allow for very noisy excited states in I=0
        if not (ISOSPIN == 0 and GEVP):
            if VERBOSE:
                print("skip: only two data points in fit curve")
            ret = True
        else:
            print("warning: only two data points in fit curve")

    #cut on arithmetic sequence
    if not ret and len(filter_sparse(
            excl, meta.fitwindow, xstep=meta.options.xstep)) != len(excl):
        if not (ISOSPIN == 0 and GEVP):
            if VERBOSE:
                print("skip: not an arithmetic sequence")
            ret = True
        else:
            if VERBOSE:
                print("warning: not an arithmetic sequence")

    ret = False if ISOSPIN == 0 and GEVP else ret
    if ret:
        if VERBOSE:
            print('excl:', excl, 'is too small')
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
