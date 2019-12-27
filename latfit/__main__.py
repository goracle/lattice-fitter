#!/usr/bin/env pypy3
"""Fit function to data.
Compute chi^2 (t^2) and errors.
Plot fit with error bars.
Save result to pdf.
usage note: MAKE SURE YOU SET THE Y LIMITS of your plot by hand!
usage note(2): MAKE SURE as well that you correct the other "magic"
parts of the graph routine
"""

# install pip3
# then sudo pip3 install numdifftools

from collections import namedtuple
import os
import re
from math import sqrt
import time
import sys
import signal
import pickle
import numpy as np
import mpi4py
from mpi4py import MPI

from latfit.config import JACKKNIFE, TSEP_VEC, BINNUM, BIASED_SPEEDUP
from latfit.config import FIT, METHOD, TLOOP, ADD_CONST, USE_LATE_TIMES
from latfit.config import ISOSPIN, MOMSTR, UNCORR, GEVP_DEBUG
from latfit.config import PVALUE_MIN, SYS_ENERGY_GUESS, LT
from latfit.config import GEVP, SUPERJACK_CUTOFF, EFF_MASS, VERBOSE
from latfit.config import MAX_RESULTS, GEVP_DERIV, TLOOP_START
from latfit.config import CALC_PHASE_SHIFT, LATTICE_ENSEMBLE
from latfit.config import SKIP_OVERFIT, NOLOOP, MATRIX_SUBTRACTION
from latfit.utilities.postprod.h5jack import TDIS_MAX
from latfit.analysis.superjack import jack_mean_err
from latfit.makemin.mkmin import convert_to_namedtuple

from latfit.extract.errcheck.xlim_err import fitrange_err
from latfit.extract.proc_folder import proc_folder
import latfit.extract.extract as ext
from latfit.finalout.printerr import printerr
from latfit.makemin.mkmin import NegChisq
from latfit.analysis.errorcodes import XmaxError, RelGammaError, ZetaError
from latfit.analysis.errorcodes import XminError, FitRangeInconsistency
from latfit.analysis.errorcodes import DOFNonPos, BadChisq, FitFail
from latfit.analysis.errorcodes import DOFNonPosFit, MpiSkip
from latfit.analysis.errorcodes import BadJackknifeDist, NoConvergence
from latfit.analysis.errorcodes import EnergySortError, TooManyBadFitsError
from latfit.analysis.result_min import Param
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.config import PHASE_SHIFT_ERR_CUT
from latfit.config import MULT, TSTEP, RANGE_LENGTH_MIN
from latfit.checks.consistency import fit_range_consistency_check
from latfit.utilities import exactmean as em
from latfit.mainfunc.metaclass import FitRangeMetaData
# dynamic
import latfit.mainfunc.fit_range_sort as frsort
import latfit.finalout.mkplot as mkplot
import latfit.mainfunc.print_res as print_res
import latfit.checks.checks_and_statements as sands
import latfit.singlefit as sfit
import latfit.config
import latfit.fit_funcs
import latfit.extract.getblock.gevp_linalg as glin

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

EXCL_ORIG = np.copy(EXCL_ORIG_IMPORT)

# acceptable errors for initial fit
ACCEPT_ERRORS_INIT = (NegChisq, RelGammaError, NoConvergence,
                      OverflowError, EnergySortError, TooManyBadFitsError,
                      np.linalg.linalg.LinAlgError, BadJackknifeDist,
                      DOFNonPosFit, BadChisq, ZetaError)


# for subsequent fits
ACCEPT_ERRORS = (NegChisq, RelGammaError, NoConvergence, OverflowError,
                 EnergySortError, TooManyBadFitsError, BadJackknifeDist,
                 DOFNonPosFit, BadChisq, ZetaError, XmaxError)

# for final representative fit.  dof should be deterministic, so should work
# (at least)
ACCEPT_ERRORS_FIN = (NegChisq, RelGammaError, NoConvergence,
                     OverflowError, EnergySortError, TooManyBadFitsError,
                     BadJackknifeDist, BadChisq, ZetaError)

assert int(TDIS_MAX) == TDIS_MAX, TDIS_MAX

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def winsize_check(meta, tadd, tsub):
    """Check proposed new fit window size to be sure
    there are enough time slices"""
    new_fitwin_len = meta.fitwindow[1] - meta.fitwindow[0] + 1 - tadd - tsub
    ret = new_fitwin_len > 0 and RANGE_LENGTH_MIN <= new_fitwin_len
    return ret

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

    # todo:  get rid of "trials" param
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
        augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)

        # a significant level of work is needed if we are entering the
        # fit loop at all; thus parallelize
        if list(meta.generate_combinations()[0]) and not meta.skip_loop():
            fit.count += 1
            if fit.count % MPISIZE != MPIRANK and MPISIZE > 1:
                raise MpiSkip
            if VERBOSE:
                print("fit.count =", fit.count)

        if FIT:

            test = True
            # print loop info
            print(tloop.ijstr)
            if tadd or tsub:
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
                if excl is None:
                    continue
                if sfit.toosmallp(meta, excl):
                    continue

                # update global info about excluded points
                latfit.config.FIT_EXCL = excl

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
                                                'phase_shift')
                fit_range_consistency_check(meta, min_arr, 'energy')

            if not meta.skip_loop:

                min_arr, overfit_arr = mpi_gather(min_arr, overfit_arr)

            test = post_loop(meta, (min_arr, overfit_arr),
                             plotdata, retsingle_save, test_success)

        elif not FIT:
            nofit_plot(meta, plotdata, retsingle_save)
    else:
        old_fit_style(meta, trials, plotdata)
    print("END FIT, rank:", MPIRANK)
    return test
fit.count = -1

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


def post_loop(meta, loop_store, plotdata,
              retsingle_save, test_success):
    """After loop over fit ranges"""
    result_min = {}
    min_arr, overfit_arr = loop_store
    min_arr = loop_result(min_arr, overfit_arr)
    # did anything succeed?
    # test = False if not list(min_arr) and not meta.random_fit else True
    test = list(min_arr) or meta.random_fit
    if not meta.skip_loop():

        result_min = find_mean_and_err(meta, min_arr)
        param_err = result_min['energy'].err

        latfit.config.FIT_EXCL = closest_fit_to_avg(
            result_min['energy'].val, min_arr)
        # do the best fit again, with good stopping condition
        # latfit.config.FIT_EXCL = min_excl(min_arr)
        print("fit excluded points (indices):",
              latfit.config.FIT_EXCL)

    if (not (meta.skip_loop() and latfit.config.MINTOL)\
        and METHOD == 'NaN') or not test_success\
        and (len(min_arr) + len(overfit_arr) > 1):
        if not TLOOP:
            latfit.config.MINTOL = True
        print("fitting for representative fit")
        try:
            retsingle = sfit.singlefit(meta, meta.input_f)
        except ACCEPT_ERRORS_FIN:
            print("reusing first successful fit"+\
                  " since representative fit failed (NoConvergence)")
            retsingle = retsingle_save
            param_err = retsingle_save[1]
    else:
        print("reusing first successful fit result for representative fit")
        retsingle = retsingle_save
        param_err = retsingle_save[1]
    result_min_close, param_err_close, \
        plotdata.coords, plotdata.cov = retsingle

    print_res.print_fit_results(meta, min_arr)
    result_min, param_err = combine_results(
        result_min, result_min_close,
        meta, param_err, param_err_close)

    print("fit window = ", meta.fitwindow)
    # plot the result
    mkplot.mkplot(plotdata, meta.input_f, result_min,
                  param_err, meta.fitwindow)

    return test

@PROFILE
def combine_results(result_min, result_min_close,
                    meta, param_err, param_err_close):
    """use the representative fit's goodness of fit in final print
    """
    if meta.skip_loop():
        result_min, param_err = result_min_close, param_err_close
    else:
        result_min['chisq'].val = result_min_close.chisq.val
        result_min['chisq'].err = result_min_close.chisq.err
        result_min['misc'] = result_min_close.misc
        result_min['pvalue'] = result_min_close.pvalue
        #result_min['pvalue'].err = result_min_close.pvalue.err
        print("closest representative fit result (lattice units):")
        # convert here since we can't set attributes afterwards
        result_min = convert_to_namedtuple(result_min)
        printerr(result_min_close.energy.val, param_err_close)
        print_res.print_phaseshift(result_min_close)
    return result_min, param_err



@PROFILE
def loop_result(min_arr, overfit_arr):
    """Test if fit range loop succeeded"""
    if min_arr:
        print(min_arr[0], np.array(min_arr).shape)
    min_arr = collapse_filter(min_arr)
    if overfit_arr:
        print(overfit_arr[0])
    overfit_arr = collapse_filter(overfit_arr)
    try:
        assert min_arr, "No fits succeeded."+\
            "  Change fit range manually:"+str(min_arr)
    except AssertionError:
        min_arr = overfit_arr
        try:
            assert overfit_arr, "No fits succeeded."+\
                "  Change fit range manually:"+str(min_arr)
        except AssertionError:
            raise FitFail
    return min_arr


@PROFILE
def collapse_filter(arr):
    """collapse the results array and filter out duplicates"""
    # collapse the array structure introduced by mpi
    shape = np.asarray(arr).shape
    if shape:
        if len(shape) > 1:
            if shape[1] != 3:
                arr = [x for b in arr for x in b]

    # filter out duplicated work (find unique results)
    arr = getuniqueres(arr)
    return arr

@PROFILE
def cutresult(result_min, min_arr, overfit_arr, param_err):
    """Check if result of fit to a
    fit range is acceptable or not, return true if not acceptable
    (result should be recorded or not)
    """
    ret = False
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
                print("warning: phase shift errors on "+\
                        "last state very large")
                ret = True if ISOSPIN == 2 and GEVP else ret
            else:
                print("phase shift errors too large")
                ret = True
    return ret

def mean_and_err_loop_continue(name, min_arr):
    """should we continue in the loop
    """
    ret = False
    if 'misc' in name:
        ret = True
    else:
        try:
            val = min_arr[0][0].__dict__[name].val
        except AttributeError:
            print("a Param got overwritten.  name:", name)
            raise
        if val is None:
            ret = True
    return ret

def fill_err_array(min_arr, name, weight_sum):
    """Fill the error array"""
    fill = []
    for i in min_arr:
        for j in min_arr:
            fill.append(jack_mean_err(
                divbychisq(getattr(i[0], name).arr,
                           getattr(i[0], 'pvalue').arr/weight_sum),
                divbychisq(getattr(j[0], name).arr,
                           getattr(j[0], 'pvalue').arr/weight_sum),
                nosqrt=True)[1])
    fill = em.acsum(fill, axis=0)
    return fill

def parametrize_entry(result_min, name):
    """Make into blank Param object"""
    if name not in result_min:
        result_min[name] = Param()
    return result_min

@PROFILE
def find_mean_and_err(meta, min_arr):
    """Find the mean and error from results of fit"""
    result_min = {}
    weight_sum = em.acsum([getattr(
        i[0], "pvalue").arr for i in min_arr], axis=0)
    for name in min_arr[0][0].__dict__:

        if mean_and_err_loop_continue(name, min_arr):
            continue

        # find the name of the array
        print("finding error in", name, "which has shape=",
              np.asarray(min_arr[0][0].__dict__[name].val).shape)

        # compute the jackknife errors as a check
        # (should give same result as error propagation)
        res_mean, err_check = jack_mean_err(em.acsum([
            divbychisq(getattr(i[0], name).arr, getattr(
                i[0], 'pvalue').arr/weight_sum) for i in min_arr], axis=0))

        # dump the results to file
        # if not (ISOSPIN == 0 and GEVP):
        if len(min_arr) > 1 or (meta.lenprod == 1 and len(min_arr) == 1):
            dump_fit_range(meta, min_arr, name, res_mean, err_check)

        # error propagation check
        result_min = parametrize_entry(result_min, name)
        result_min[name].err = fill_err_array(min_arr, name, weight_sum)
        try:
            result_min[name].err = np.sqrt(result_min[name].err)
        except FloatingPointError:
            print("floating point error in", name)
            print(result_min[name].err)
            if hasattr(result_min[name].err, '__iter__'):
                for i, res in enumerate(result_min[name].err):
                    if np.isreal(res) and res < 0:
                        result_min[name].err[i] = np.nan
            else:
                if np.isreal(result_min[name].err):
                    if res < 0:
                        result_min[name].err = np.nan
            result_min[name].err = np.sqrt(result_min[name].err)

        # perform the comparison
        try:
            assert np.allclose(
                err_check, result_min[name].err, rtol=1e-8)
        except AssertionError:
            print("jackknife error propagation"+\
                    " does not agree with jackknife"+\
                    " error.")
            print(result_min[name].err)
            print(err_check)
            if hasattr(err_check, '__iter__'):
                for i, ress in enumerate(zip(
                        result_min[name].err, err_check)):
                    res1, res2 = ress
                    print(res1, res2, np.allclose(res1, res2,
                                                  rtol=1e-8))
                    if not np.allclose(res1, res2, rtol=1e-8):
                        result_min[name][i].err = np.nan
                        err_check[i] = np.nan

    # find the weighted mean
        result_min[name].val = em.acsum(
            [getattr(i[0], name).val*getattr(i[0], 'pvalue').val
             for i in min_arr],
            axis=0)/em.acsum([getattr(i[0], 'pvalue').val
                              for i in min_arr])
    param_err = np.array(result_min['energy'].err)
    assert not any(np.isnan(param_err)), \
        "A parameter error is not a number (nan)"
    return result_min

@PROFILE
def getuniqueres(min_arr):
    """Find unique fit ranges"""
    ret = []
    keys = set()
    for i in min_arr:
        key = str(i[2])
        if key not in keys:
            ret.append(i)
            keys.add(key)
    return ret

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
def compare_eff_mass_to_range(arr, errmin, mindim=None):
    """Compare the error of err to the effective mass errors.
    In other words, find the minimum error of
    the errors on subsets of effective mass points
    and the error on the points themselves.
    """
    arreff, erreff = min_eff_mass_errors(mindim=mindim)
    if errmin == erreff:
        arr = arreff
    else:
        errmin = min(errmin, erreff)
        if errmin == erreff:
            arr = arreff
    # the error is not going to be a simple em.acstd if we do sample AMA
    # so skip the check in this case
    if not SUPERJACK_CUTOFF:
        errcheck = em.acstd(arr)*np.sqrt(len(arr-1))
        try:
            assert np.allclose(errcheck, errmin, rtol=1e-6)
        except AssertionError:
            print("error check failed")
            print(errmin, errcheck)
            sys.exit(1)
    return arr, errmin

@PROFILE
def dump_min_err_jackknife_blocks(meta, min_arr, mindim=None):
    """Dump the jackknife blocks for the energy with minimum errors"""
    fname = "energy_min_"+str(LATTICE_ENSEMBLE)
    if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
        fname = fname + '_'+dump_fit_range.fn1
    name = 'energy'
    err = np.array([getattr(i[0], name).err for i in min_arr])
    dimops = err.shape[1]
    if dimops == 1:
        err = err[:, 0]
        errmin = min(err)
        ind = list(err).index(min(err))
        arr = getattr(min_arr[ind][0], 'energy').arr
    else:
        assert mindim is not None, "needs specification of operator"+\
            " dimension to write min error jackknife blocks (unsupported)."
        print(err.shape)
        errmin = min(err[:, mindim])
        ind = list(err[:, mindim]).index(errmin)
        fname = fname+'_mindim'+str(mindim)
        arr = np.asarray(getattr(min_arr[ind][0], 'energy').arr[:, mindim])
    arr, errmin = compare_eff_mass_to_range(arr, errmin, mindim=mindim)

    fname = filename_plus_config_info(meta, fname)
    print("dumping jackknife energies with error:", errmin,
          "into file:", fname+'.p')
    pickle.dump(arr, open(fname+'.p', "wb"))

def time_slice_list():
    """Get list of time slices from reuse dictionary keys"""
    # time slice list
    times = []
    for key in sfit.singlefit.reuse:
        if not isinstance(key, float) and not isinstance(key, int):
            continue
        if int(key) == key:
            times.append(int(key))
    times = sorted(times)
    return times

def get_dimops_at_time(time1):
    """ get the dimension of the GEVP (should be == MULT)
    (also, dimops should be == first config's energy values at time1)"""
    dimops = len(sfit.singlefit.reuse[time1][0])
    assert dimops == MULT, (dimops, MULT)
    assert dimops == len(sfit.singlefit.reuse[time1][0])


if EFF_MASS:
    @PROFILE
    def min_eff_mass_errors(mindim=None, getavg=False):
        """Append the errors of the effective mass points
        to errarr"""

        # time slice list
        xcoord = list(sfit.singlefit.coords_full[:, 0])
        assert mindim is None or isinstance(mindim, int),\
            "type check failed"

        # build the time slice lists of eff mass, errors
        dimops = None
        errlist = []
        arrlist = []
        for _, time1 in enumerate(time_slice_list()):

            # we may have extra time slices
            # (due to fit window cuts from TLOOP)
            if time1 not in xcoord:
                continue

            # check dimensionality
            dimops = get_dimops_at_time(time1) if dimops is None else dimops
            assert dimops == get_dimops_at_time(time1), (
                get_dimops_at_time(time1), dimops)

            # masses and errors at this time slice
            arr = sfit.singlefit.reuse[time1]
            err = sfit.singlefit.error2[xcoord.index(time1)]

            # reduce to a specific GEVP dimension
            if mindim is not None:
                arr = arr[:, mindim]
                err = err[mindim]
                assert isinstance(err, float), str(err)
            arrlist.append(arr)
            errlist.append(err)

        if getavg and mindim is None:
            arr, err = config_avg_eff_mass(arrlist, errlist)    
        elif not getavg and mindim is not None:
            arr, err = mindim_eff_mass(arrlist, errlist)
        elif not getavg and mindim is None:
            arr, err = np.array(arrlist), np.array(errlist)
        else:
            assert None, ("mode check fail:", getavg, mindim)

        assert isinstance(err, float) or mindim is None, "index bug"
        return arr, err

    def mindim_eff_mass(arrlist, errlist):
        """find the time slice which gives the minimum error
        then get the energy and error at that point
        this only makes sense if we are at a particular GEVP dimension
        """
        assert isinstance(errlist[0], float),\
            (errlist," ",sfit.singlefit.error2[xcoord.index(10)],
                np.asarray(errlist).shape, np.asarray(errlist[0]).shape)
        err = min(errlist)
        arr = arrlist[errlist.index(err)]
        return arr, err

    def config_avg_eff_mass(arrlist, errlist):
        """Get the config average of the effective mass points.
        Add structure to non-GEVP points to make files dumped like a 1x1 GEVP
        """
        err = np.asarray(errlist)
        # average over configs
        arr = em.acmean(np.asarray(arrlist), axis=1)

        # add structure in arr for backwards compatibility
        if isinstance(arr[0], float): # no GEVP
            assert MULT == 1, MULT
            arr = np.asarray([[i] for i in arr])
            assert isinstance(err[0], float),\
                "error array does not have same structure as"+\
                " eff mass array"
            err = np.asarray([[i] for i in err])

        # index checks
        assert len(arr.shape) == 2, (
            arr, "first dim is time, second dim is operator", arr.shape)
        assert len(err.shape) == 2, (
            err, "first dim is time, second dim is operator", err.shape)
        assert len(err) == len(arr), (len(err), len(arr))
        return arr, err

else:
    @PROFILE
    def min_eff_mass_errors(_):
        """blank"""
        return (None, np.inf)

@PROFILE
def pickle_res(name, min_arr):
    """Return the fit range results to be pickled,
    append the effective mass points
    """
    ret = [getattr(i[0], name).arr for i in min_arr]
    origlshape = np.asarray(ret, dtype=object).shape
    print("res shape", origlshape)
    origl = len(ret)
    if 'energy' in name:
        arreff, _ = min_eff_mass_errors()
        ret = [*ret, *arreff]
    ret = np.asarray(ret, dtype=object)
    assert len(origlshape) == len(ret.shape), str(origlshape)+","+str(
        ret.shape)
    flen = len(ret)
    print("original error length (res):", origl,
          "final error length:", flen)
    return ret

@PROFILE
def pickle_res_err(name, min_arr):
    """Append the effective mass errors to the """
    ret = [getattr(i[0], name).err for i in min_arr]
    print("debug:[getattr(i[0], name) for i in min_arr].shape",
          np.asarray(ret).shape)
    print("debug2:", np.asarray(sfit.singlefit.error2).shape)
    origl = len(ret)
    if GEVP and 'systematics' not in name:
        if len(np.asarray(ret).shape) > 1:
            # dimops check
            dimops1 = (np.array(ret).shape)[1]
            if name == 'min_params' and SYS_ENERGY_GUESS:
                dimops1 = int((dimops1-1)/2)
            dimops2 = (np.asarray(sfit.singlefit.error2).shape)[1]
            assert dimops1 == dimops2, (np.array(ret).shape,
                                        np.asarray(sfit.singlefit.error2),
                                        name)
    if 'energy' in name:
        _, erreff = min_eff_mass_errors(getavg=True)
        ret = np.array([*ret, *erreff])
    ret = np.asarray(ret)
    flen = len(ret)
    print("original error length (err):", origl,
          "final error length:", flen)
    return ret

@PROFILE
def pickle_excl(meta, min_arr):
    """Get the fit ranges to be pickled
    append the effective mass points
    """
    ret = [print_res.inverse_excl(meta, i[2]) for i in min_arr]
    print("original number of fit ranges before effective mass append:",
          len(ret))
    if EFF_MASS:
        xcoord = list(sfit.singlefit.coords_full[:, 0])
        xcoordapp = [[i] for i in xcoord]
        ret = [*ret, *xcoordapp]
    ret = np.array(ret, dtype=object)
    print("final fit range amount:", len(ret))
    return ret

@PROFILE
def dump_fit_range(meta, min_arr, name, res_mean, err_check):
    """Pickle the fit range result"""
    print("starting arg:", name)
    if 'energy' in name: # no clobber (only do this once)
        if MULT > 1:
            for i in range(len(res_mean)):
                dump_min_err_jackknife_blocks(meta, min_arr, mindim=i)
        else:
            dump_min_err_jackknife_blocks(meta, min_arr)
    pickl_res = pickle_res(name, min_arr)
    pickl_res_err = pickle_res_err(name, min_arr)
    pickl_excl = pickle_excl(meta, min_arr)
    pickl_res_fill = np.empty(4, object)
    try:
        pickl_res_fill[:] = [res_mean, err_check, pickl_res, pickl_excl]
        pickl_res = pickl_res_fill
    except ValueError:
        print(np.asarray(res_mean).shape)
        print(np.asarray(err_check).shape)
        print(np.asarray(pickl_res).shape)
        print(np.asarray(pickl_excl).shape)
        print(name)
        raise
    assert pickl_res_err.shape == pickl_res[2].shape[0::2], (
        "array mismatch:", pickl_res_err.shape, pickl_res[2].shape)
    assert len(pickl_res) == 4, "bad result length"

    if not GEVP:
        if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
            name = name+'_'+dump_fit_range.fn1
        name = re.sub('.jkdat', '', name)
    filename = filename_plus_config_info(meta, name)
    filename_err = filename_plus_config_info(meta, name+'_err')
    write_pickle_file_verb(filename, pickl_res)
    write_pickle_file_verb(filename_err, pickl_res_err)
dump_fit_range.fn1 = None

def write_pickle_file_verb(filename, arr):
    """Write pickle file; print info"""
    print("writing pickle file", filename)
    pickle.dump(arr, open(filename+'.p', "wb"))

def filename_plus_config_info(meta, filename):
    """Add config info to file name"""
    if GEVP:
        filename += "_"+MOMSTR+'_I'+str(ISOSPIN)
    if meta.random_fit:
        filename += '_randfit'
    if SYS_ENERGY_GUESS:
        filename += "_sys"
    if MATRIX_SUBTRACTION:
        filename += '_dt'+str(latfit.config.DELTA_T_MATRIX_SUBTRACTION)
    filename += meta.window_str()+"_"+latfit.config.T0
    return filename

@PROFILE
def divbychisq(param_arr, pvalue_arr):
    """Divide a parameter by chisq (t^2)"""
    assert not any(np.isnan(pvalue_arr)), "pvalue array contains nan"
    ret = np.array(param_arr)
    assert ret.shape, str(ret)
    if len(ret.shape) > 1:
        assert param_arr[:, 0].shape == pvalue_arr.shape,\
            "Mismatch between pvalue_arr"+\
            " and parameter array (should be the number of configs):"+\
            str(pvalue_arr.shape)+", "+str(param_arr.shape)
        for i in range(len(ret[0])):
            try:
                assert not any(np.isnan(param_arr[:, i])),\
                    "parameter array contains nan"
            except AssertionError:
                print("found nan in dimension :", i)
                for j in param_arr:
                    print(j)
                sys.exit(1)
            ret[:, i] *= pvalue_arr
            assert not any(np.isnan(ret[:, i])),\
                "parameter array contains nan"
    else:
        try:
            assert not np.any(np.isnan(param_arr)),\
                "parameter array contains nan"
        except AssertionError:
            for i in param_arr:
                print(i)
            raise
        except TypeError:
            print("param_arr=", param_arr)
            raise
        try:
            ret *= pvalue_arr
        except ValueError:
            print("could not be broadcast together")
            print("ret=", ret)
            print("pvalue_arr=", pvalue_arr)
            raise
    assert ret.shape == param_arr.shape,\
        "return shape does not match input shape"
    return ret


@PROFILE
def closest_fit_to_avg(result_min_avg, min_arr):
    """Find closest fit to average fit
    (find the most common fit range)
    """
    minmax = np.nan
    ret_excl = []
    for i, fiti in enumerate(min_arr):
        minmax_i = max(abs(fiti[0].energy.val-result_min_avg))
        if i == 0:
            minmax = minmax_i
            ret_excl = fiti[2]
        else:
            minmax = min(minmax_i, minmax)
            if minmax == minmax_i:
                ret_excl = fiti[2]
    return ret_excl


# obsolete, we should simply pick the model with the smallest errors
# and an adequate chi^2 (t^2)
@PROFILE
def min_excl(min_arr):
    """Find the minimum reduced chisq (t^2) from all the fits considered"""
    minres = sorted(min_arr, key=lambda row: row[0])[0]
    if UNCORR:
        print("min chisq/dof=", minres[0])
    else:
        print("min t^2/dof=", minres[0])
    print("best times to exclude:", minres[1])
    return minres[1]

@PROFILE
def augment_excl(excl):
    """If the user has specified excluded indices add these to the list."""
    for num, (i, j) in enumerate(zip(excl, augment_excl.excl_orig)):
        excl[num] = sorted(list(set(j).union(set(i))))
    return excl
augment_excl.excl_orig = EXCL_ORIG
frsort.augment_excl = augment_excl

@PROFILE
def dof_check(lenfit, dimops, excl):
    """Check the degrees of freedom.  If < 1, cause a skip"""
    dof = (lenfit-1)*dimops
    ret = True
    for i in excl:
        for _ in i:
            dof -= 1
    if dof < 1:
        ret = False
    return ret


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

# https://stackoverflow.com/questions/1158076/implement-touch-using-python
@PROFILE
def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    """unix touch"""
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags,
                           mode=mode, dir_fd=dir_fd)) as fn1:
        os.utime(fn1.fileno() if os.utime in os.supports_fd else fname,
                 dir_fd=None if os.supports_fd else dir_fd, **kwargs)

def xmax_err(meta, err):
    """Handle xmax error"""
    if VERBOSE:
        print("Test fit failed; bad xmax. problemx:", err.problemx)
    meta.decr_xmax(err.problemx)
    if VERBOSE:
        print("xmin, new xmax =", meta.options.xmin, meta.options.xmax)
    if meta.fitwindow[1] < meta.options.xmax and FIT:
        print("***ERROR***")
        print("fit window beyond xmax:", meta.fitwindow)
        sys.exit(1)
    #meta.fitwindow = fitrange_err(meta.options, meta.options.xmin,
    #                              meta.options.xmax)
    #print("new fit window = ", meta.fitwindow)
    return meta

def xmin_err(meta, err):
    """Handle xmax error"""
    if VERBOSE:
        print("Test fit failed; bad xmin.")
    # if we're past the halfway point, then this error is likely a late time
    # error, not an early time error (usually from pion ratio)
    if err.problemx > (meta.options.xmin + meta.options.xmax)/2:
        raise XmaxError(problemx=err.problemx)
    meta.incr_xmin(err.problemx)
    if VERBOSE:
        print("new xmin, xmax =", meta.options.xmin, meta.options.xmax)
    if meta.fitwindow[0] > meta.options.xmin and FIT:
        print("***ERROR***")
        print("fit window beyond xmin:", meta.fitwindow)
        sys.exit(1)
    #meta.fitwindow = fitrange_err(meta.options, meta.options.xmin,
    #                              meta.options.xmax)
    #print("new fit window = ", meta.fitwindow)
    return meta

def dofit_initial(meta, plotdata):
    """Do an initial test fit"""

    test_success = False
    retsingle_save = None
    flag = True
    while flag:
        try:
            if VERBOSE:
                print("Trying initial fit with excluded times:",
                      latfit.config.FIT_EXCL, "fit window:", meta.fitwindow,
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
        except ACCEPT_ERRORS_INIT as err:
            flag = False
            print("fit failed (acceptably) with error:",
                  err.__class__.__name__)

    # results need for return
    # plotdata, meta, test_success, fit_range_init
    return (meta, plotdata, test_success, retsingle_save)

def update_fitwin(meta, tadd, tsub):
    """Update fit window"""
    # tadd tsub cut
    if tadd or tsub:
        #print("tadd =", tadd, "tsub =", tsub)
        for _ in range(tadd):
            meta.incr_xmin()
        for _ in range(tsub):
            meta.decr_xmax()
        partial_reset()


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
                      latfit.config.FIT_EXCL)
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
        if CALC_PHASE_SHIFT:
            print_res.print_phaseshift(result_min)
        if not cutresult(result_min, min_arr,
                         overfit_arr, param_err):
            result = [result_min, list(param_err),
                      list(latfit.config.FIT_EXCL)]
            # don't overfit
            if result_min.chisq.val/result_min.misc.dof >= 1 and\
               SKIP_OVERFIT:
                min_arr.append(result)
            else:
                overfit_arr.append(result)
        else:
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
    augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
    return plotdata


def dofit(meta, fit_range_data, results_store, plotdata):
    """Do a fit on a particular fit range"""

    # unpack
    min_arr, overfit_arr, retsingle_save = results_store
    idx, excl, fit_range_init = fit_range_data
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
    if frsort.keyexcl(excl) == fit_range_init:
        skip = True
    else:
        try:
            retsingle = sfit.singlefit(meta, meta.input_f)
            if retsingle_save is None:
                retsingle_save = retsingle
            if VERBOSE:
                print("fit succeeded for this selection"+\
                      " excluded points=", excl)
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
    print("current delta t matsub =",
          latfit.config.DELTA_T_MATRIX_SUBTRACTION)
    print("new GEVP t-t0 =", latfit.config.T0)

def incr_dt():
    """Increment the matrix subtraction time separation in
    D(t):= C(t)-C(t-dt), where D is the subtracted GEVP matrix"""
    dtee = latfit.config.DELTA_T_MATRIX_SUBTRACTION
    dtee = tsep_based_incr(dtee)
    latfit.config.DELTA_T_MATRIX_SUBTRACTION = dtee
    # check this!!!
    latfit.fit_funcs.TSTEP = TSTEP if not GEVP or GEVP_DEBUG else dtee
    latfit.config.FITS.select_and_update(ADD_CONST)
    print("new delta t matsub =", latfit.config.DELTA_T_MATRIX_SUBTRACTION)
    print("current GEVP t-t0 =", latfit.config.T0)

def main():
    try:
        tloop()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, "")

def signal_handler(sig, frame):
        print('Ctrl+C pressed; raising.')
        raise
signal.signal(signal.SIGINT, signal_handler)

def tloop():
    """main"""
    mintol = latfit.config.MINTOL # store tolerance preferences
    if TLOOP: # don't pause (show anything) since we are looping over plots
        mkplot.NOSHOW = True
    # outer loop is over matrix subtraction delta_t
    # (assumes we never do a second subtraction)
    for i in range(TSEP_VEC[0]+1): # not set up for xstep
        assert np.all(TSEP_VEC[0] == np.asarray(TSEP_VEC)), str(TSEP_VEC)
        if i < TLOOP_START[0]:
            continue
        if i:
            if MATRIX_SUBTRACTION and TLOOP:
                incr_dt()
            else:
                break
        for j in range(TSEP_VEC[0]+1): # loop over t-t0
            # get rid of all gevp/eff mass processing
            if j < TLOOP_START[1]:
                continue
            if j:
                if TLOOP:
                    incr_t0()
                else:
                    break
            ext.reset_extract()
            if i or j:
                latfit.config.TITLE_PREFIX = latfit.config.title_prefix(
                    tzero=latfit.config.T0,
                    dtm=latfit.config.DELTA_T_MATRIX_SUBTRACTION)
            for tsub in range(int(TDIS_MAX)): # this is the tmax loop
                tadd = 0
                flag = 1
                if not TLOOP and tsub:
                    break

                while flag and tadd <= int(TDIS_MAX): # this is the tmin/tadd loop
                    if not TLOOP and tadd:
                        break

                    reset_main(mintol) # reset the fitter for next fit

                   # parallelize loop
                    #if (1000*j+100*i+10*tsub+tadd) % MPISIZE != MPIRANK\
                    #   and MPISIZE > 1:
                        #tadd += 1
                        #continue
                    tloop.ijstr = "t indices, mpi rank: "+str(
                        i)+" "+str(j)+" "+str(MPIRANK)
                    try:
                        test = fit(tadd=tadd, tsub=tsub)
                        flag = 0 # flag stays 0 if fit succeeds
                        if not test:
                            break
                    except (FitRangeInconsistency, FitFail, MpiSkip):
                        if VERBOSE:
                            print("starting a new main()",
                                  "(inconsistent/fitfail/mpi skip).  rank:", MPIRANK)
                        flag = 1
                        tadd += 1 # add this to tmin

                    except DOFNonPos:
                        # exit the loop; we're totally out of dof
                        break
    if TLOOP:
        print("End of t loop.  latfit exiting, rank:", MPIRANK)
    else:
        print("End of selected parameter fit.  latfit exiting, rank:", MPIRANK)
tloop.ijstr = ""

def reset_main(mintol):
    """Reset all dynamic variables"""
    latfit.config.MINTOL = mintol
    latfit.config.FIT_EXCL = np.copy(EXCL_ORIG)
    latfit.config.FIT_EXCL = [list(i) for i in latfit.config.FIT_EXCL]
    partial_reset()

def partial_reset():
    """Partial reset during tloop"""
    glin.reset_sortevals()
    sfit.singlefit_reset()

if __name__ == "__main__":
    print("__main__.py should not be called directly")
    print("install first with python3 setup.py install")
    print("then run: latfit <args>")
    sys.exit(1)
