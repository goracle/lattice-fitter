"""The main fit code"""
import sys
import os
from math import sqrt
from collections import namedtuple
import time
import numpy as np
import mpi4py
from mpi4py import MPI

from latfit.config import TLOOP, TSEP_VEC, TLOOP_START, LATTICE_ENSEMBLE
from latfit.config import VERBOSE, FIT, MATRIX_SUBTRACTION, GEVP_DERIV
from latfit.config import BINNUM, USE_LATE_TIMES, BIASED_SPEEDUP, ADD_CONST
from latfit.config import MULT, METHOD, JACKKNIFE, GEVP, GEVP_DEBUG
from latfit.config import TSTEP, CALC_PHASE_SHIFT, SKIP_OVERFIT
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT, MAX_RESULTS
from latfit.config import INCLUDE

# errors
from latfit.analysis.errorcodes import XmaxError, XminError
from latfit.analysis.errorcodes import NegChisq
from latfit.analysis.errorcodes import RelGammaError, ZetaError
from latfit.analysis.errorcodes import FitRangeInconsistency
from latfit.analysis.errorcodes import DOFNonPos, BadChisq, FitFail
from latfit.analysis.errorcodes import DOFNonPosFit, MpiSkip, FinishedSkip
from latfit.analysis.errorcodes import BadJackknifeDist, NoConvergence
from latfit.analysis.errorcodes import EnergySortError, TooManyBadFitsError

# util
from latfit.utilities.postprod.h5jack import ENSEMBLE_DICT

# main
from latfit.mainfunc.cache import reset_cache, reset_main
from latfit.mainfunc.metaclass import FitRangeMetaData
from latfit.mainfunc.postloop import dump_fit_range
from latfit.mainfunc.fitwin import winsize_check, update_fitwin
from latfit.mainfunc.fitwin import xmin_err, xmax_err
from latfit.mainfunc.cuts import cutresult
from latfit.mainfunc.postloop import post_loop

from latfit.extract.proc_folder import proc_folder
from latfit.checks.consistency import fit_range_consistency_check
from latfit.finalout.printerr import printerr

# dynamic
import latfit.mainfunc.print_res as print_res
import latfit.mainfunc.fit_range_sort as frsort
import latfit.singlefit as sfit
import latfit.finalout.mkplot as mkplot
import latfit.extract.extract as ext
import latfit.config
import latfit.checks.checks_and_statements as sands
import latfit.mathfun.proc_meff as effmass

EXCL_ORIG = np.copy(list(EXCL_ORIG_IMPORT))

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False


# for subsequent fits
ACCEPT_ERRORS = (NegChisq, RelGammaError, NoConvergence, OverflowError,
                 EnergySortError, TooManyBadFitsError, BadJackknifeDist,
                 DOFNonPosFit, BadChisq, ZetaError, XmaxError)
if INCLUDE:
    ACCEPT_ERRORS = ()

def tloop():
    """main"""
    mintol = latfit.config.MINTOL # store tolerance preferences
    if TLOOP: # don't pause (show anything) since we are looping over plots
        mkplot.NOSHOW = True
    # outer loop is over matrix subtraction delta_t
    # (assumes we never do a second subtraction)
    loop_len = TSEP_VEC[0] + 1
    start_count = loop_len*TLOOP_START[0]+ TLOOP_START[1]
    tdis_max = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max']
    assert int(tdis_max) == tdis_max, tdis_max

    for i in range(loop_len): # not set up for xstep
        assert np.all(TSEP_VEC[0] == np.asarray(TSEP_VEC)), str(TSEP_VEC)
        if i:
            if MATRIX_SUBTRACTION and TLOOP:
                incr_dt()
            else:
                break
        for j in range(loop_len): # loop over t-t0
            # get rid of all gevp/eff mass processing
            if j:
                if TLOOP:
                    incr_t0()
                else:
                    break
            if j+i*loop_len < start_count:
                continue
            reset_cache()
            if i or j:
                latfit.config.TITLE_PREFIX = latfit.config.title_prefix(
                    tzero=latfit.config.T0,
                    dtm=latfit.config.DELTA_T_MATRIX_SUBTRACTION)
            for tsub in range(int(tdis_max)): # this is the tmax loop

                if tsub % MPISIZE != MPIRANK and MPISIZE > 1:
                    continue
                if not TLOOP and tsub:
                    break
                if VERBOSE:
                    print("starting tsub:", tsub, "rank:", MPIRANK)

                tadd = 0
                flag = 1

                while flag and tadd <= int(
                        tdis_max): # this is the tmin/tadd loop
                    if not TLOOP and tadd:
                        break

                    reset_main(mintol) # reset the fitter for next fit

                   # parallelize loop
                    #if (1000*j+100*i+10*tsub+tadd) % MPISIZE != MPIRANK\
                    #   and MPISIZE > 1:
                        #tadd += 1
                        #continue
                    print("t indices, matdt, t-t0, mpi rank:",
                          i, j, latfit.config.DELTA_T_MATRIX_SUBTRACTION,
                          latfit.config.T0, MPIRANK)
                    print("tadd, tsub, mpi rank:", tadd, tsub, MPIRANK)
                    try:
                        test = fit(tadd=tadd, tsub=tsub)
                        flag = 0 # flag stays 0 if fit succeeds
                        if not test:
                            break
                    except (FitRangeInconsistency, FitFail, MpiSkip):
                        if VERBOSE:
                            print("starting a new main()",
                                  "(inconsistent/fitfail/mpi skip).  rank:",
                                  MPIRANK)
                        flag = 1
                        tadd += 1 # add this to tmin

                    except FinishedSkip:
                        if VERBOSE:
                            print("starting a new main() with new tsub rank:",
                                  MPIRANK)
                        flag = 0 # flag stays 0 if fit succeeds
                        break
                    except DOFNonPos:
                        # exit the loop; we're totally out of dof
                        break
    if TLOOP:
        print("End of t loop.  latfit exiting, rank:", MPIRANK)
    else:
        print("End of selected parameter fit.  latfit exiting, rank:",
              MPIRANK)

def tsep_based_incr(dtee):
    """Increment dt modulo TSEP+2 (+2 is arbitrary, but seems to work)"""
    ret = dtee + 1
    ret = ret  % (TSEP_VEC[0]+2)
    if not ret:
        ret += 1
    return ret

def incr_t0():
    """Increment t0, the RHS GEVP time assuming t-t0=const.
    (increment this const.)"""
    dtee = int(latfit.config.T0[6:])
    dtee = tsep_based_incr(dtee)
    latfit.config.T0 = 'TMINUS'+str(dtee)
    latfit.fit_funcs.DELTAT = -1 if GEVP_DERIV else dtee
    sands.bin_time_statements(BINNUM, USE_LATE_TIMES, latfit.config.T0,
                              BIASED_SPEEDUP)
    latfit.config.FITS.select_and_update(ADD_CONST)
    if VERBOSE:
        print("current delta t matsub =",
              latfit.config.DELTA_T_MATRIX_SUBTRACTION)
        print("new GEVP t-t0 =", latfit.config.T0)

def incr_dt():
    """Increment the matrix subtraction time separation in
    D(t):= C(t)-C(t-dt), where D is the subtracted GEVP matrix"""
    dtee = latfit.config.DELTA_T_MATRIX_SUBTRACTION
    dtee = tsep_based_incr(dtee)
    latfit.config.DELTA_T_MATRIX_SUBTRACTION = dtee
    effmass.EFF_MASS_TOMIN = effmass.create_funcs()
    # check this!!!
    latfit.fit_funcs.TSTEP = TSTEP if not GEVP or GEVP_DEBUG else dtee
    latfit.config.FITS.select_and_update(ADD_CONST)
    if VERBOSE:
        print("new delta t matsub =", latfit.config.DELTA_T_MATRIX_SUBTRACTION)
        print("current GEVP t-t0 =", latfit.config.T0)

@PROFILE
def fit(tadd=0, tsub=0):
    """Main for latfit"""
    # set up 1ab
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])
    test = not FIT

    meta = FitRangeMetaData()
    trials, plotdata, dump_fit_range.fn1 = meta.setup(plotdata)

    ## error processing, parameter extractions

    # check to be sure we have zero'd out the old eff mass blocks
    # if we, e.g., change the GEVP params
    if not tadd and not tsub:
        assert not ext.extract.reuse, list(ext.extract.reuse)

    if trials == -1 and winsize_check(meta, tadd, tsub):
        # try an initial plot, shrink the xmax if it's too big
        update_fitwin(meta, tadd, tsub)
        start = time.perf_counter()
        meta, plotdata, test_success, retsingle_save = dofit_initial(
            meta, plotdata)
        if VERBOSE:
            print("Total elapsed time =",
                  time.perf_counter()-start, "seconds. rank:", MPIRANK)

        # update the known exclusion information with plot points
        # which are nan (not a number) or
        # which have error bars which are too large
        # also, update with deliberate exclusions as part of TLOOP mode
        augment_excl.excl_orig = np.copy(list(latfit.config.FIT_EXCL))

        if FIT:

            test = True
            # print loop info
            if (tadd or tsub) and VERBOSE:
                print("tadd =", tadd, "tsub =",
                      tsub, "rank =", MPIRANK)

            ## allocate results storage, do second initial test fit
            ## (if necessary)
            start = time.perf_counter()
            min_arr, overfit_arr, retsingle_save, fit_range_init = \
                dofit_second_initial(meta, retsingle_save, test_success)
            if VERBOSE:
                print("Total elapsed time =",
                      time.perf_counter()-start, "seconds. rank:", MPIRANK)

            ### Setup for fit range loop

            plotdata = store_init(plotdata)
            prod, sorted_fit_ranges = frsort.fit_range_combos(meta,
                                                              plotdata)

            # store checked fit ranges
            checked = set()

            # assume that manual spec. overrides brute force search
            meta.skip_loop()
            if VERBOSE:
                print("starting loop of max length:"+str(
                    meta.lenprod), "random fit:", meta.random_fit)

            for idx in range(meta.lenprod):

                # exit the fit loop?
                if frsort.exitp(meta, min_arr, overfit_arr, idx):
                    break

                # get one fit range, check it
                excl, checked = frsort.get_one_fit_range(
                    meta, prod, idx, sorted_fit_ranges, checked)
                if excl is not None:
                    excl = list(excl)
                if excl is None:
                    continue
                if sfit.toosmallp(meta, excl):
                    continue

                # update global info about excluded points
                latfit.config.FIT_EXCL = list(excl)

                # do fit
                start = time.perf_counter()
                retsingle, plotdata = dofit(meta,
                                            (idx, excl, fit_range_init),
                                            (min_arr, overfit_arr,
                                             retsingle_save),
                                            plotdata)
                if VERBOSE:
                    print("Total elapsed time =",
                          time.perf_counter()-start,
                          "seconds. rank:", MPIRANK)
                if retsingle[0]: # skip processing
                    continue

                # process and store fit result
                min_arr, overfit_arr, retsingle_save = process_fit_result(
                    retsingle, excl, min_arr, overfit_arr)

                if CALC_PHASE_SHIFT:
                    fit_range_consistency_check(meta, min_arr,
                                                'phase_shift', mod_180=True)
                fit_range_consistency_check(meta, min_arr, 'energy')

            if not meta.skip_loop:

                min_arr, overfit_arr = mpi_gather(min_arr, overfit_arr)

            test = post_loop(meta, (min_arr, overfit_arr),
                             plotdata, retsingle_save, test_success)

        elif not FIT:
            nofit_plot(meta, plotdata, retsingle_save)
    else:
        old_fit_style(meta, trials, plotdata)
    if VERBOSE:
        print("END FIT, rank:", MPIRANK)
    return test

def dofit_initial(meta, plotdata):
    """Do an initial test fit"""

    # acceptable errors for initial fit
    accept_errors = (NegChisq, RelGammaError, NoConvergence,
                     OverflowError, EnergySortError, TooManyBadFitsError,
                     np.linalg.linalg.LinAlgError, BadJackknifeDist,
                     DOFNonPosFit, BadChisq, ZetaError)
    if INCLUDE:
        accept_errors = ()

    test_success = False
    retsingle_save = None
    flag = True
    while flag:
        try:
            if VERBOSE:
                print("Trying initial fit with excluded times:",
                      list(latfit.config.FIT_EXCL),
                      "fit window:", meta.fitwindow,
                      'rank:', MPIRANK)
            retsingle_save = sfit.singlefit(meta, meta.input_f)
            test_success = True if len(retsingle_save) > 2 else test_success
            flag = False
            if FIT and test_success and VERBOSE:
                print("Test fit succeeded. rank:", MPIRANK)
        except XmaxError as err:
            test_success = False
            try:
                meta = xmax_err(meta, err)
            except XminError as err2:
                meta = xmax_err(meta, err2)
            plotdata.fitcoord = meta.fit_coord()
        except XminError as err:
            test_success = False
            try:
                meta = xmin_err(meta, err)
            except XmaxError as err2:
                meta = xmax_err(meta, err2)
            plotdata.fitcoord = meta.fit_coord()
        except accept_errors as err:
            flag = False
            if VERBOSE:
                print("fit failed (acceptably) with error:",
                      err.__class__.__name__)

    # results need for return
    # plotdata, meta, test_success, fit_range_init
    return (meta, plotdata, test_success, retsingle_save)

def dofit_second_initial(meta, retsingle_save, test_success):
    """Do second initial test fit and cut on error size"""

    # store different excluded, and the avg chisq/dof (t^2/dof)
    min_arr = []
    overfit_arr = [] # allow overfits if no usual fits succeed

    # cut late time points from the fit range
    # did we make any cuts?
    samerange = sfit.cut_on_errsize(meta)
    samerange = sfit.cut_on_growing_exp(meta) and samerange
    assert samerange

    fit_range_init = str(latfit.config.FIT_EXCL)
    try:
        if not samerange and FIT:
            if VERBOSE:
                print("Trying second initial fit with excluded times:",
                      list(latfit.config.FIT_EXCL))
            retsingle_save = sfit.singlefit(meta, meta.input_f)
            test_success = True if len(retsingle_save) > 2 else test_success
            if test_success and VERBOSE:
                print("(second) Test fit succeeded. rank:", MPIRANK)
            test_success = True
    except AssertionError:
        print(meta.input_f, meta.fitwindow, meta.options.xmin,
              meta.options.xmax, meta.options.xstep)
        raise
    except XmaxError as err:
        test_success = False
        try:
            meta = xmax_err(meta, err)
        except XminError as err2:
            meta = xmax_err(meta, err2)
        # plotdata.fitcoord = meta.fit_coord()
        fit_range_init = None
    except ACCEPT_ERRORS as err:
        print("fit failed (acceptably) with error:",
              err.__class__.__name__)
        fit_range_init = None
    if test_success:
        result_min, param_err, _, _ = retsingle_save
        printerr(result_min.energy.val, param_err)
        if CALC_PHASE_SHIFT and VERBOSE:
            print_res.print_phaseshift(result_min)
        if not cutresult(result_min, min_arr,
                         overfit_arr, param_err):
            result = (result_min, list(param_err),
                      list(latfit.config.FIT_EXCL))
            # don't overfit
            if result_min.chisq.val/result_min.misc.dof >= 1 and\
               SKIP_OVERFIT:
                min_arr.append(result)
            else:
                overfit_arr.append(result)
        else:
            if VERBOSE:
                print("cutting result of test fits")
    assert len(min_arr) + len(overfit_arr) <= 1, len(
        min_arr) + len(overfit_arr)
    return min_arr, overfit_arr, retsingle_save, fit_range_init

def store_init(plotdata):
    """Storage modification;
    act on info from initial fit ranges"""
    assert sfit.singlefit.coords_full is not None
    plotdata.coords, plotdata.cov = sfit.singlefit.coords_full, \
        sfit.singlefit.cov_full
    augment_excl.excl_orig = np.copy(list(latfit.config.FIT_EXCL))
    return plotdata


def dofit(meta, fit_range_data, results_store, plotdata):
    """Do a fit on a particular fit range"""

    # unpack
    min_arr, overfit_arr, retsingle_save = results_store
    idx, excl, fit_range_init = fit_range_data
    excl = list(excl)
    skip = False
    try:
        showint = int(min(np.floor(meta.lenprod/10), (MPISIZE*5)))
        if not showint:
            showint = 1
    except FloatingPointError:
        print("floating point problem or bug")
        print(meta.lenprod)
        print(MPISIZE)
        showint = 1

    if VERBOSE or not idx % showint:
        print("Trying fit with excluded times:",
              latfit.config.FIT_EXCL,
              "fit window:", meta.fitwindow,
              "fit:",
              str(idx+1)+"/"+str(meta.lenprod))
        print("number of results:", len(min_arr),
              "number of overfit", len(overfit_arr),
              "rank:", MPIRANK)
    assert len(latfit.config.FIT_EXCL) == MULT, "bug"
    # retsingle_save needs a cut on error size
    if frsort.keyexcl(list(excl)) == fit_range_init:
        skip = True
    else:
        try:
            retsingle = sfit.singlefit(meta, meta.input_f)
            if retsingle_save is None:
                retsingle_save = retsingle
            if VERBOSE:
                print("fit succeeded for this selection"+\
                      " excluded points=", list(excl))
            if meta.lenprod == 1 or MAX_RESULTS == 1:
                retsingle_save = retsingle
        except ACCEPT_ERRORS as err:
            # skip on any error
            if VERBOSE or not idx % showint:
                print("fit failed for this selection."+\
                      " excluded points=", excl, "with error:",
                      err.__class__.__name__)
            skip = True
    if not skip:
        result_min, param_err, plotdata.coords, plotdata.cov = retsingle
        if VERBOSE:
            printerr(result_min.energy.val, param_err)
        if CALC_PHASE_SHIFT and VERBOSE:
            print_res.print_phaseshift(result_min)
    else:
        retsingle = (None, None, plotdata.coords, plotdata.cov)

    return (skip, retsingle, retsingle_save), plotdata

def process_fit_result(retsingle, excl, min_arr, overfit_arr):
    """ After fitting, process/store the results
    """
    # unpack
    _, retsingle, retsingle_save = retsingle
    result_min, param_err, _, _ = retsingle
    skip = False

    if cutresult(result_min, min_arr, overfit_arr, param_err):
        skip = True

    if not skip:
        result = [result_min, list(param_err), list(excl)]

        if result_min.chisq.val/result_min.misc.dof >= 1: # don't overfit
            min_arr.append(result)
        else:
            overfit_arr.append(result)
    return min_arr, overfit_arr, retsingle_save

@PROFILE
def augment_excl(excli):
    """If the user has specified excluded indices add these to the list."""
    excl = list(excli)
    for num, (i, j) in enumerate(zip(excl, augment_excl.excl_orig)):
        excl[num] = sorted(list(set(j).union(set(i))))
    return excl
augment_excl.excl_orig = list(EXCL_ORIG)
frsort.augment_excl = augment_excl

def old_fit_style(meta, trials, plotdata):
    """Fit using the original fit style
    (very old, likely does not work)
    """
    list_fit_params = []
    for ctime in range(trials):
        ifile = proc_folder(meta.input_f, ctime, "blk")
        ninput = os.path.join(meta.input_f, ifile)
        result_min, _, plotdata.coords, plotdata.cov =\
            sfit.singlefit(meta, ninput)
        list_fit_params.append(result_min.energy.val)
    printerr(*get_fitparams_loc(list_fit_params, trials))

def nofit_plot(meta, plotdata, retsingle_save):
    """No fit scatter plot """
    #if MPIRANK == 0:
    if not latfit.config.MINTOL or METHOD == 'Nelder-Mead':
        retsingle = sfit.singlefit(meta, meta.input_f)
        plotdata.coords, plotdata.cov = retsingle
    else:
        plotdata.coords, plotdata.cov = retsingle_save
    mkplot.mkplot(plotdata, meta.input_f)


@PROFILE
def mpi_gather(min_arr, overfit_arr):
    """Gather mpi results.
    Does not work for some reason
    """
    return min_arr, overfit_arr
    #min_arr_send = np.array(min_arr)
    #MPI.COMM_WORLD.barrier()
    #min_arr = MPI.COMM_WORLD.gather(min_arr_send, 0)
    #MPI.COMM_WORLD.barrier()
    #print("results gather complete.")
    #overfit_arr = MPI.COMM_WORLD.gather(overfit_arr, 0)
    #MPI.COMM_WORLD.barrier()
    #print("overfit gather complete.")
    #return min_arr, overfit_arr

@PROFILE
def get_fitparams_loc(list_fit_params, trials):
    """Not sure what this does, probably wrong"""
    list_fit_params = np.array(list_fit_params).T.tolist()
    avg_fit_params = [sum(list_fit_params[i])/len(list_fit_params[i])
                      for i in range(len(list_fit_params))]
    if JACKKNIFE == "YES":
        prefactor = (trials-1.0)/(1.0*trials)
    elif JACKKNIFE == "NO":
        prefactor = (1.0)/((trials-1.0)*(1.0*trials))
    else:
        print("***ERROR***")
        print("JACKKNIFE value should be a string with value either")
        print("YES or NO")
        print("Please examine the config file.")
        sys.exit(1)
    err_fit_params = [sqrt(sum([(
        list_fit_params[i][j]-avg_fit_params[i])**2 for j in range(
            len(list_fit_params[i]))])*prefactor) for i in range(
                len(list_fit_params))]
    return avg_fit_params, err_fit_params
