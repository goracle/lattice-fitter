"""The main fit code"""
import sys
import os
from math import sqrt
from collections import namedtuple
from multiprocessing import Pool
import time
import numpy as np
import mpi4py
from mpi4py import MPI

from latfit.config import TLOOP, TSEP_VEC, TLOOP_START, LATTICE_ENSEMBLE
from latfit.config import VERBOSE, FIT, MATRIX_SUBTRACTION, GEVP_DERIV
from latfit.config import BINNUM, USE_LATE_TIMES, BIASED_SPEEDUP, ADD_CONST
from latfit.config import MULT, METHOD, JACKKNIFE, GEVP
from latfit.config import TSTEP, CALC_PHASE_SHIFT, NOLOOP
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.config import INCLUDE, ONLY_EXTRACT, ALTERNATIVE_PARALLELIZATION
from latfit.config import LARGE_DELTA_T_MATRIX_SUBTRACTION

# errors
from latfit.analysis.errorcodes import XmaxError, XminError
from latfit.analysis.errorcodes import NegChisq, FitSuccess
from latfit.analysis.errorcodes import RelGammaError, ZetaError
from latfit.analysis.errorcodes import FitRangeInconsistency
from latfit.analysis.errorcodes import FitRangesAlreadyInconsistent
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
from latfit.mainfunc.fitwin import xmin_err, xmax_err, checkpast
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

DOWRITE = ALTERNATIVE_PARALLELIZATION and not MPIRANK\
    or not ALTERNATIVE_PARALLELIZATION

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
            check = False
            for tsub in range(int(tdis_max)): # this is the tmax loop

                if tsub and ONLY_EXTRACT:
                    break

                if not ALTERNATIVE_PARALLELIZATION:
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

                    # reset the fitter for next fit
                    reset_main(mintol, check=check)

                   # parallelize loop
                    #if (1000*j+100*i+10*tsub+tadd) % MPISIZE != MPIRANK\
                    #   and MPISIZE > 1:
                        #tadd += 1
                        #continue
                    if DOWRITE:
                        print("t indices, matdt, t-t0, mpi rank:",
                              i, j, latfit.config.DELTA_T_MATRIX_SUBTRACTION,
                              latfit.config.T0, MPIRANK)
                        print("tadd, tsub, mpi rank:", tadd, tsub, MPIRANK)
                    try:
                        test, check2 = fit(tadd=tadd, tsub=tsub)
                        check = check or check2
                        flag = 0 # flag stays 0 if fit succeeds
                        if not test:
                            break
                    except (FitRangeInconsistency, FitFail):
                        if VERBOSE:
                            print("starting a new main()",
                                  "(inconsistent/fitfail/mpi skip).  rank:",
                                  MPIRANK)
                        flag = 1
                        check = True
                        tadd += 1 # add this to tmin

                    except MpiSkip:
                        if VERBOSE:
                            print("starting a new main()",
                                  "(mpi skip).  rank:",
                                  MPIRANK)
                        flag = 1
                        tadd += 1 # add this to tmin

                    except FitRangesAlreadyInconsistent:
                        if VERBOSE:
                            print("starting a new main()",
                                  "fit ranges already found to be",
                                  "inconsistent.  rank:",
                                  MPIRANK)
                        flag = 1
                        tadd += 1 # add this to tmin

                    except FinishedSkip:
                        if VERBOSE:
                            print("already done fitting this fit window.",
                                  "starting a new main() with new tsub rank:",
                                  MPIRANK)
                        break

                    except DOFNonPos:
                        if VERBOSE:
                            print("skipping fit window;",
                                  "degrees of freedom <= 0")
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
        print("ADD_CONST", ADD_CONST)
        print("current delta t matsub =",
              latfit.config.DELTA_T_MATRIX_SUBTRACTION)
        print("new GEVP t-t0 =", latfit.config.T0)

def incr_dt():
    """Increment the matrix subtraction time separation in
    D(t):= C(t)-C(t-dt), where D is the subtracted GEVP matrix"""
    if LARGE_DELTA_T_MATRIX_SUBTRACTION:
        ens_offset = 5 if LATTICE_ENSEMBLE == '24c' else 0 # using large dt gives empirical statistical improvement (observed by T. Izubuchi).
    else:
        ens_offset = 0
    if latfit.config.DELTA_T_MATRIX_SUBTRACTION == 1:
        latfit.config.DELTA_T_MATRIX_SUBTRACTION += ens_offset
    dtee = latfit.config.DELTA_T_MATRIX_SUBTRACTION-ens_offset
    dtee = tsep_based_incr(dtee)
    dtee += ens_offset
    latfit.config.DELTA_T_MATRIX_SUBTRACTION = dtee
    assert latfit.config.DELTA_T_MATRIX_SUBTRACTION > ens_offset, (
        latfit.config.DELTA_T_MATRIX_SUBTRACTION, ens_offset)
    effmass.EFF_MASS_TOMIN = effmass.create_funcs()
    # check this!!!
    latfit.fit_funcs.TSTEP = TSTEP if not GEVP else dtee
    latfit.config.FITS.select_and_update(ADD_CONST)
    if VERBOSE:
        print("new delta t matsub =",
              latfit.config.DELTA_T_MATRIX_SUBTRACTION)
        print("current GEVP t-t0 =", latfit.config.T0)

@PROFILE
def fit(tadd=0, tsub=0):
    """Main for latfit"""
    # set up 1ab
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])
    test = not FIT or ONLY_EXTRACT
    processed = False

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
        processed = True
        if VERBOSE:
            print("Total elapsed time =",
                  time.perf_counter()-start, "seconds. rank:", MPIRANK)

        # update the known exclusion information with plot points
        # which are nan (not a number) or
        # which have error bars which are too large
        # also, update with deliberate exclusions as part of TLOOP mode
        augment_excl.excl_orig = np.copy(list(latfit.config.FIT_EXCL))

        if FIT and not ONLY_EXTRACT:

            test = True
            # print loop info
            if (tadd or tsub) and VERBOSE:
                print("tadd =", tadd, "tsub =",
                      tsub, "rank =", MPIRANK)

            ## allocate results storage, do second initial test fit
            ## (if necessary)
            start = time.perf_counter()
            min_arr, overfit_arr, retsingle_save, checked = \
                dofit_second_initial(meta, retsingle_save, test_success)
            if VERBOSE:
                print("Total elapsed time =",
                      time.perf_counter()-start, "seconds. rank:", MPIRANK)

            ### Setup for fit range loop

            plotdata = store_init(plotdata)
            combo_data = frsort.fit_range_combos(meta, plotdata)

            # assume that manual spec. overrides brute force search
            meta.skip_loop()
            # subtract one since we already checked one fit range
            print("starting loop of max length:"+str(
                meta.lenprod-1), "random fit:", meta.random_fit)

            idxstart = 0
            chunk_new = -1
            for chunk in range(6):

                if chunk < chunk_new:
                    continue

                # only one check for exhaustive fits
                if chunk and not meta.random_fit:
                    break
                if frsort.exitp(meta, min_arr, overfit_arr, idxstart):
                    break

                # otherwise, get a chunk of fit ranges
                # excls is this chunk;
                # represented as sets of excluded points
                excls, chunk_new, checked = frsort.combo_data_to_fit_ranges(
                    meta, combo_data, chunk, checked=checked)
                print("starting chunk", chunk,
                      "which has", len(excls),
                      "fit ranges; result goal:",
                      frsort.threshold(chunk), "rank:", MPIRANK)
                if not meta.random_fit:
                    # reduce by 1 since we've already checked one
                    # in dofit_second_initial
                    assert len(excls) == meta.lenprod-1, (
                        len(excls), meta.lenprod)
                excls = [i for i in excls if i is not None]

                # parallelize over this chunk
                results = dofit_parallel(
                    meta, idxstart, excls,
                    (len(min_arr), len(overfit_arr)))
                # keep track of where we are in the overall loop
                idxstart += len(excls)

                # store at least one result
                if results and retsingle_save is None:
                    assert results[0] is not None, results
                    retsingle_save, _ = results[0]

                for retsingle, excl in results:
                    # process and store fit result
                    if retsingle is None:
                        continue
                    min_arr, overfit_arr = process_fit_result(
                        retsingle, min_arr, overfit_arr, excl)
                    # perform another consistency check
                    # (after results collected)
                    consis(meta, min_arr)


            if not meta.skip_loop:

                min_arr, overfit_arr = mpi_gather(min_arr, overfit_arr)

            test = post_loop(meta, (min_arr, overfit_arr), retsingle_save)

        elif not FIT:
            nofit_plot(meta, plotdata, retsingle_save)
    else:
        old_fit_style(meta, trials, plotdata)
    if VERBOSE:
        print("END FIT, rank:", MPIRANK)
    return test, processed

def dofit_parallel(meta, idxstart, excls, results_lengths):
    """Use multiprocess to parallelize over fit ranges
    or over configs
    """
    # initial pass
    # fit the chunk (to be parallelized)
    excls = list(excls)
    results_excls = dofit_parallelized_over_fit_ranges(
        meta, idxstart, excls, results_lengths, False)
    # i has retsingle, idx, excl
    excls = [(i[1], i[2]) for i in results_excls if i is not None]

    # now we have a set of fit ranges which made it past the first
    # two configs without error; thus, we should now let these
    # presumably "good" set of fits proceed

    results = []
    if meta.options.procs > len(excls):
        # parallelize over configs (jackknife samples)
        print("number of good fits is small (rank):", excls, MPIRANK)
        print("len(excls) (rank)", len(excls), MPIRANK)
        for idx_excl in excls:
            toadd = dofit(meta, idx_excl, results_lengths, True)
            if toadd is not None:
                results.append(toadd)
        print("number of results from small set of good ranges (rank):",
              len(results), MPIRANK)
    else:
        # number of good fits is large;
        # continue to parallelize over fit ranges
        print("number of good fits is large (rank):", len(excls), MPIRANK)
        results = dofit_parallelized_over_fit_ranges(
            meta, None, excls, results_lengths, True)
        print("number of results from large set of good ranges (rank):",
              len(results), MPIRANK)

    return results

def dofit_parallelized_over_fit_ranges(
        meta, idxstart, excls, results_lengths, fullfit):
    """fit, parallelized over fit ranges using multiprocess"""

    # index start is added to index
    if isinstance(idxstart, (np.integer, np.float, int)):
        argtup = [(meta, (idx+idxstart, excl), results_lengths, fullfit)
                  for idx, excl in enumerate(excls)]

    # index start has already been added to index
    else:
        assert idxstart is None, idxstart
        argtup = [(meta, idx_excl, results_lengths, fullfit)
                  for idx_excl in excls]

    poolsize = min(meta.options.procs, len(excls))
    with Pool(poolsize) as pool:
        results = pool.starmap(dofit, argtup)
    results = [i for i in results if i is not None]
    return results


def consis(meta, min_arr):
    """Check fit results for consistency"""
    if CALC_PHASE_SHIFT:
        fit_range_consistency_check(meta, min_arr,
                                    'phase_shift', mod_180=True)
    fit_range_consistency_check(meta, min_arr, 'energy')


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
    xmin_store = meta.options.xmin
    xmax_store = meta.options.xmax
    latfit.config.FITS.select_and_update(ADD_CONST)
    while flag:
        try:
            if VERBOSE:
                print("Trying initial fit with excluded times:",
                      list(latfit.config.FIT_EXCL),
                      "fit window:", meta.fitwindow,
                      'rank:', MPIRANK)
            if not ext.iscomplete(): # void the cache here for safety
                # it should be voided if it's partially full
                # this prevents bugs where some error starts filling it
                # then breaks us out of the 'while flag' loop
                # (e.g. fit window found to be inconsistent already)
                if xmin_store == meta.options.xmin and xmax_store == meta.options.xmax:
                    reset_cache()
                elif xmin_store == meta.options.xmin and xmax_store != meta.options.xmax:
                    # removing extra time slices at the end is safe,
                    # since processing always goes forward
                    pass
                elif xmin_store != meta.options.xmin:
                    reset_cache()
            xmin_store = meta.options.xmin
            xmax_store = meta.options.xmax
            retsingle_save = sfit.singlefit(meta, meta.input_f)
            test_success = True if len(retsingle_save) > 2 else test_success
            flag = False
            if FIT and test_success and VERBOSE:
                print("Test fit succeeded. rank:", MPIRANK)
        except XmaxError as err:
            test_success = False
            try:
                meta = xmax_err(meta, err, check_past=False)
            except XminError as err2:
                assert None, "not supported"
                meta = xmax_err(meta, err2, check_past=False)
            plotdata.fitcoord = meta.fit_coord()
        except XminError as err:
            test_success = False
            try:
                meta = xmin_err(meta, err)
            except XmaxError as err2:
                meta = xmax_err(meta, err2, check_past=False)
            plotdata.fitcoord = meta.fit_coord()
        except accept_errors as err:
            flag = False
            if VERBOSE:
                print("fit failed (acceptably) with error:",
                      err.__class__.__name__)

    # results need for return
    # plotdata, meta, test_success, fit_range_init
    assert ext.iscomplete()
    if not NOLOOP:
        checkpast(meta)
    return (meta, plotdata, test_success, retsingle_save)

def dofit_second_initial(meta, retsingle_save, test_success):
    """Do second initial test fit and cut on error size"""

    # store different excluded, and the avg chisq/dof (t^2/dof)
    min_arr = []
    overfit_arr = [] # allow overfits if no usual fits succeed

    # cut late time points from the fit range
    # did we make any cuts?
    assert sfit.singlefit.error2 is not None
    samerange = sfit.cut_on_errsize(meta)
    samerange = sfit.cut_on_growing_exp(meta) and samerange
    assert samerange

    fit_range_init = frsort.keyexcl(list(latfit.config.FIT_EXCL))
    excl = list(latfit.config.FIT_EXCL)

    try:
        if not samerange and FIT:
            if VERBOSE:
                print("Trying second initial fit with excluded times:", excl)
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
            meta = xmax_err(meta, err, check_past=True)
        except XminError as err2:
            meta = xmax_err(meta, err2, check_past=True)
        # plotdata.fitcoord = meta.fit_coord()
        fit_range_init = None
    except ACCEPT_ERRORS as err:
        if DOWRITE:
            print("fit failed (acceptably) with error:",
                  err.__class__.__name__)
        fit_range_init = None
    if test_success:

        min_arr, overfit_arr = process_fit_result(
            retsingle_save, min_arr, overfit_arr, excl)

    assert len(min_arr) + len(overfit_arr) <= 1, len(
        min_arr) + len(overfit_arr)

    # store checked fit ranges
    if fit_range_init is not None:
        checked = frsort.setup_checked(fit_range_init)
    else:
        checked = set()

    return min_arr, overfit_arr, retsingle_save, checked

def store_init(plotdata):
    """Storage modification;
    act on info from initial fit ranges"""
    assert sfit.singlefit.coords_full is not None
    plotdata.coords, plotdata.cov = sfit.singlefit.coords_full, \
        sfit.singlefit.cov_full
    augment_excl.excl_orig = np.copy(list(latfit.config.FIT_EXCL))
    return plotdata

def show_int(meta):
    """Find the fit range index
    when we should show progress"""
    # compute when to print results
    proc_metric = max(MPISIZE*5, meta.options.procs)
    showint = 1
    try:
        showint = int(min(np.floor(meta.lenprod/100), proc_metric))
        if not showint:
            showint = 1
    except FloatingPointError:
        print("floating point problem or bug")
        print(meta.lenprod)
        print(MPISIZE)
    return showint


def dofit(meta, idx_excl, results_lengths, fullfit=True):
    """Do a fit on a particular fit range"""

    # set fit range
    idx, excl = idx_excl
    skip = frsort.set_fit_range(meta, excl)
    # make sure we've checked this fit range for skipping
    # if we demand a full fit
    assert not skip or not fullfit, (skip, fullfit)

    retsingle = None
    retex = None
    if not skip:

        # unpack
        lmin, loverfit = results_lengths

        showint = show_int(meta)

        if (VERBOSE or not idx % showint) and DOWRITE:
            print("Trying fit with excluded times:",
                  latfit.config.FIT_EXCL,
                  "fit window:", meta.fitwindow,
                  "fit:", str(idx+1)+"/"+str(meta.lenprod))
            print("number of results:", lmin,
                  "number of overfit", loverfit,
                  "rank:", MPIRANK)
        assert len(latfit.config.FIT_EXCL) == MULT, "bug"
        retex = None

        # fit timer
        start = time.perf_counter()

        try:
            retsingle = sfit.singlefit(meta, meta.input_f, fullfit=fullfit)
            if VERBOSE:
                print("fit succeeded for this selection"+\
                      " excluded points=", list(excl))
        except ACCEPT_ERRORS as err: # fit fail
            # skip on any error
            if (VERBOSE or not idx % showint) and DOWRITE:
                print("fit failed for this selection."+\
                      " excluded points=", excl, "with error:",
                      err.__class__.__name__)
            skip = True
        except FitSuccess: # initial pass
            retex = idx, excl
            if VERBOSE:
                print("marking", excl, "as good.")
            skip = True

        # end fit timer
        if VERBOSE:
            print("Total elapsed time =",
                  time.perf_counter()-start,
                  "seconds. rank:", MPIRANK)

    if not skip:
        result_min, param_err, _, _ = retsingle
        if VERBOSE:
            printerr(result_min.energy.val, param_err)
            if CALC_PHASE_SHIFT:
                print_res.print_phaseshift(result_min)

    ret = None
    if retex is not None:
        ret = retsingle, *retex
    elif retsingle is not None:
        ret = retsingle, excl

    return ret

def process_fit_result(retsingle, min_arr, overfit_arr, excl):
    """ After fitting, process/store the results
    """
    # unpack
    result_min, param_err, _, _ = retsingle
    skip = False
    excl = list(excl)
    #excl = list(latfit.config.FIT_EXCL)

    if cutresult(result_min, min_arr, overfit_arr, param_err):
        skip = True

    if not skip:
        result = [result_min, list(param_err), list(excl)]

        if result_min.chisq.val/result_min.misc.dof >= 1: # don't overfit
            min_arr.append(result)
        else:
            overfit_arr.append(result)
    return min_arr, overfit_arr

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
