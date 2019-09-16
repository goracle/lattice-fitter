#!/usr/bin/env python

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
import pickle
import numpy as np
from mpi4py import MPI

from latfit.singlefit import singlefit
import latfit.singlefit
from latfit.config import JACKKNIFE
from latfit.config import FIT, METHOD
from latfit.config import ISOSPIN, MOMSTR, UNCORR
from latfit.config import PVALUE_MIN
from latfit.config import GEVP, SUPERJACK_CUTOFF, EFF_MASS
from latfit.config import MAX_RESULTS
from latfit.config import CALC_PHASE_SHIFT, LATTICE_ENSEMBLE
from latfit.config import SKIP_OVERFIT
from latfit.jackknife_fit import jack_mean_err
from latfit.makemin.mkmin import convert_to_namedtuple

from latfit.extract.errcheck.xlim_err import fitrange_err
from latfit.extract.proc_folder import proc_folder
from latfit.finalout.printerr import printerr
from latfit.finalout.mkplot import mkplot
from latfit.makemin.mkmin import NegChisq
from latfit.analysis.errorcodes import XmaxError, RelGammaError, ZetaError
from latfit.analysis.errorcodes import DOFNonPos, BadChisq
from latfit.analysis.errorcodes import BadJackknifeDist, NoConvergence
from latfit.analysis.errorcodes import EnergySortError, TooManyBadFitsError
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.config import PHASE_SHIFT_ERR_CUT
from latfit.config import MULT
import latfit.config
import latfit.jackknife_fit
from latfit.utilities import exactmean as em
from latfit.mainfunc.metaclass import FitRangeMetaData
import latfit.mainfunc.fit_range_sort as sfit
import latfit.mainfunc.print_res as print_res

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

EXCL_ORIG = np.copy(EXCL_ORIG_IMPORT)

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def main():
    """Main for latfit"""
    # set up 1ab
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])

    meta = FitRangeMetaData()
    trials, plotdata, dump_fit_range.fn1 = meta.setup(plotdata)

    # error processing, parameter extractions

    if trials == -1: # get rid of this
        # try an initial plot, shrink the xmax if it's too big
        print("Trying initial test fit.")
        meta, plotdata, test_success, retsingle_save = dofit_initial(
            meta, plotdata)

        # update the known exclusion information with plot points
        # which are nan (not a number) or
        # which have error bars which are too large
        augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
        if FIT:

            ## allocate results storage, do second initial test fit
            ## (if necessary)
            min_arr, overfit_arr, retsingle_save, fit_range_init = \
                dofit_second_initial(meta, retsingle_save, test_success)

            ### Setup for fit range loop

            plotdata = store_init(plotdata)
            prod, sorted_fit_ranges = sfit.fit_range_combos(meta, plotdata)

            # store checked fit ranges
            checked = set()

            # assume that manual spec. overrides brute force search
            meta.skip_loop()
            print("starting loop of max length:"+str(
                meta.lenprod), "random fit:", meta.random_fit)
            for idx in range(meta.lenprod):

                # parallelize loop
                if idx % MPISIZE != MPIRANK:
                    #print("mpi skip")
                    continue

                # exit the fit loop?
                if sfit.exitp(meta, min_arr, overfit_arr, idx):
                    break

                # get one fit range, check it
                excl, checked = sfit.get_one_fit_range(
                    meta, prod, idx, sorted_fit_ranges, checked)
                if excl is None:
                    continue
                if sfit.toosmallp(meta, excl):
                    print('excl:', excl, 'is too small')
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
                print("Total elapsed time =",
                      time.perf_counter()-start, "seconds")
                if retsingle[0]: # skip processing
                    continue

                # process and store fit result
                min_arr, overfit_arr, retsingle_save = process_fit_result(
                    retsingle, excl, min_arr, overfit_arr)

            if not meta.skip_loop:

                min_arr, overfit_arr = mpi_gather(min_arr, overfit_arr)

            if MPIRANK == 0:

                post_loop(meta,
                          (min_arr, overfit_arr),
                          plotdata, retsingle_save, test_success)

        else:
            nofit_plot(meta, plotdata, retsingle_save)
    else:
        old_fit_style(meta, trials, plotdata)
    print("END STDOUT OUTPUT")

def old_fit_style(meta, trials, plotdata):
    """Fit using the original fit style
    (very old, likely does not work)
    """
    list_fit_params = []
    for ctime in range(trials):
        ifile = proc_folder(meta.input_f, ctime, "blk")
        ninput = os.path.join(meta.input_f, ifile)
        result_min, _, plotdata.coords, plotdata.cov =\
            singlefit(ninput, meta.fitwindow,
                      meta.options.xmin, meta.options.xmax, meta.options.xstep)
        list_fit_params.append(result_min.energy.val)
    printerr(*get_fitparams_loc(list_fit_params, trials))

def nofit_plot(meta, plotdata, retsingle_save):
    """No fit scatter plot """
    if MPIRANK == 0:
        if not latfit.config.MINTOL or METHOD == 'Nelder-Mead':
            retsingle = singlefit(meta.input_f, meta.fitwindow,
                                  meta.options.xmin, meta.options.xmax, meta.options.xstep)
            plotdata.coords, plotdata.cov = retsingle
        else:
            plotdata.coords, plotdata.cov = retsingle_save
        mkplot(plotdata, meta.input_f)


def post_loop(meta, loop_store, plotdata,
              retsingle_save, test_success):
    """After loop over fit ranges"""
    result_min = {}
    min_arr, overfit_arr = loop_store
    min_arr = loop_result(min_arr, overfit_arr)
    if not meta.skiploop:

        result_min = find_mean_and_err(meta, min_arr)
        param_err = result_min['energy'].err

        latfit.config.FIT_EXCL = closest_fit_to_avg(
            result_min['energy'].val, min_arr)
        # do the best fit again, with good stopping condition
        # latfit.config.FIT_EXCL = min_excl(min_arr)
        print("fit excluded points (indices):",
              latfit.config.FIT_EXCL)

    latfit.config.BOOTSTRAP = True
    if (not (meta.skiploop and latfit.config.MINTOL)\
        and METHOD == 'NaN') or not test_success\
        and (len(min_arr) + len(overfit_arr) > 1):
        latfit.config.MINTOL = True
        retsingle = singlefit(meta.input_f, meta.fitwindow,
                              meta.options.xmin, meta.options.xmax,
                              meta.options.xstep)
    else:
        retsingle = retsingle_save
        _ = singlefit(meta.input_f, meta.fitwindow,
                      meta.options.xmin, meta.options.xmax,
                      meta.options.xstep)
    result_min_close, param_err_close, \
        plotdata.coords, plotdata.cov = retsingle

    print_res.print_fit_results(meta, min_arr)
    result_min, param_err = combine_results(
        result_min, result_min_close,
        meta, param_err, param_err_close)

    print("fit window = ", meta.fitwindow)
    # plot the result
    mkplot(plotdata, meta.input_f, result_min,
           param_err, meta.fitwindow)

@PROFILE
def combine_results(result_min, result_min_close,
                    meta, param_err, param_err_close):
    """use the representative fit's goodness of fit in final print
    """
    if meta.skip_loop:
        result_min, param_err = result_min_close, param_err_close
    else:
        result_min['chisq'].val = result_min_close.fun
        result_min['chisq'].err = result_min_close.chisq_err
        result_min['dof'] = result_min_close.dof
        result_min['pvalue'] = result_min_close.pvalue
        result_min['pvalue'].err = result_min_close.pvalue_err
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
        assert overfit_arr, "No fits succeeded."+\
            "  Change fit range manually:"+str(min_arr)
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
    print("p-value = ", result_min.pvalue.val)
    # reject model at 10% level
    if result_min.pvalue.val < PVALUE_MIN:
        print("Not storing result because p-value"+\
                " is below rejection threshold. number"+\
                " of non-overfit results so far =", len(min_arr))
        print("number of overfit results =", len(overfit_arr))
        ret = True

    # is this justifiable?
    if not ret and sfit.skip_large_errors(result_min.energy.val, param_err):
        print("Skipping fit range because param errors"+\
                " are greater than 100%")
        ret = True

    # is this justifiable?
    if not ret and CALC_PHASE_SHIFT and MULT > 1:
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
    if min_arr[0][0].__dict__[name] is None:
        print("name=", name, "is None, skipping")
        ret = True
    elif 'misc' in name:
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
                divbychisq(getattr(j[0], name),
                           getattr(j[0], 'pvalue').arr/weight_sum),
                nosqrt=True)[1])
    return fill

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
        dump_fit_range(meta, min_arr, name, res_mean, err_check)

        # error propagation check

        result_min[name].err = fill_err_array(min_arr, name, weight_sum)
        result_min[name] = em.acsum(result_min[name].err, axis=0)
        try:
            result_min[name].err = np.sqrt(result_min[name].err)
        except FloatingPointError:
            print("floating point error")
            if hasattr(result_min[name].err, '__iter__'):
                for i, res in enumerate(result_min[name].err):
                    if np.isreal(res) and res < 0:
                        result_min[name].err[i] = np.nan
            else:
                if np.isreal(result_min[name].err):
                    if res < 0:
                        result_min[name].err = np.nan
            result_min[name].err = np.sqrt(result_min[name])

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
    param_err = np.array(result_min['energy']).err
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
    min_arr_send = np.array(min_arr)
    MPI.COMM_WORLD.barrier()
    min_arr = MPI.COMM_WORLD.gather(min_arr_send, 0)
    MPI.COMM_WORLD.barrier()
    print("results gather complete.")
    overfit_arr = MPI.COMM_WORLD.gather(overfit_arr, 0)
    MPI.COMM_WORLD.barrier()
    print("overfit gather complete.")
    return min_arr, overfit_arr

@PROFILE
def compare_eff_mass_to_range(arr, errmin, mindim=None):
    """Compare the error of err to the effective mass errors.
    In other words, find the minimum error of
    the errors on subsets of effective mass points
    and the error on the points themselves.
    """
    arreff, erreff = min_eff_mass_errors(mindim)
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
def dump_min_err_jackknife_blocks(min_arr, mindim=None):
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
    print("dumping jackknife energies with error:", errmin,
          "into file:", fname+'.p')
    pickle.dump(arr, open(fname+'.p', "wb"))

if EFF_MASS:
    @PROFILE
    def min_eff_mass_errors(mindim=None, getavg=False):
        """Append the errors of the effective mass points
        to errarr"""
        errlist = []
        arrlist = []
        xcoord = list(singlefit.coords_full[:, 0])
        assert mindim is None or isinstance(mindim, int),\
            "type check failed"
        times = []
        dimops = None
        for key in singlefit.reuse:
            if not isinstance(key, float) and not isinstance(key, int):
                continue
            if int(key) == key:
                times.append(key)
        times = sorted(times)
        for _, time1 in enumerate(times):
            if not isinstance(time1, int) and not isinstance(time1, float):
                continue
            if mindim is None:
                arr = singlefit.reuse[time1]
                err = singlefit.error2[xcoord.index(time1)]
            else:
                dimops = len(singlefit.reuse[time1][0])\
                    if dimops is None else dimops
                assert dimops == len(singlefit.reuse[time1][0])
                if not getavg:
                    arr = singlefit.reuse[time1][:, mindim]
                    err = singlefit.error2[xcoord.index(time1)][mindim]
                    assert isinstance(err, float), str(err)
                else:
                    arr = singlefit.reuse[time1]
                    err = singlefit.error2[xcoord.index(time1)]
            arrlist.append(arr)
            errlist.append(err)
        if not getavg:
            assert isinstance(errlist[0], float),\
                str(errlist)+" "+str(singlefit.error2[xcoord.index(10)])
            err = min(errlist)
            arr = arrlist[errlist.index(err)]
        else:
            err = np.asarray(errlist)
            arr = em.acmean(np.asarray(arrlist), axis=1)
            if isinstance(arr[0], float):
                # add structure in arr for backwards compatibility
                arr = np.asarray([[i] for i in arr])
                assert isinstance(err[0], float),\
                    "error array does not have same structure as eff mass array"
                err = np.asarray([[i] for i in err])
            assert len(arr.shape) == 2, "first dim is time, second dim is operator"
            assert len(err.shape) == 2, "first dim is time, second dim is operator"
            assert len(err) == len(arr)
            assert mindim is None
        assert isinstance(err, float) or mindim is None, "index bug"
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
    ret = [getattr(i[0], name).val for i in min_arr]
    origlshape = np.asarray(ret, dtype=object).shape
    print("res shape", origlshape)
    origl = len(ret)
    if 'energy' in name:
        arreff, _ = min_eff_mass_errors(mindim=None, getavg=True)
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
    print("debug2:", np.asarray(singlefit.error2).shape)
    origl = len(ret)
    if GEVP and 'systematics' not in name:
        if len(np.asarray(ret).shape) > 1:
            # dimops check
            assert (np.array(ret).shape)[1] ==\
                (np.asarray(singlefit.error2).shape)[1], str(
                    np.array(ret).shape)+" "+str(
                        np.asarray(singlefit.error2))+" "+str(name)
    if 'energy' in name:
        _, erreff = min_eff_mass_errors(mindim=None, getavg=True)
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
        xcoord = list(singlefit.coords_full[:, 0])
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
                dump_min_err_jackknife_blocks(min_arr, mindim=i)
        else:
            dump_min_err_jackknife_blocks(min_arr)
    pickl_res = pickle_res(name, min_arr)
    pickl_res_err = pickle_res_err(name, min_arr)
    pickl_excl = pickle_excl(meta, min_arr)
    pickl_res = np.array([res_mean, err_check,
                          pickl_res, pickl_excl], dtype=object)
    assert pickl_res_err.shape == pickl_res[2].shape, "array mismatch:"+\
        str(pickl_res_err.shape)+str(pickl_res[2].shape)
    if not GEVP:
        if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
            name = name+'_'+dump_fit_range.fn1
        name = re.sub('.jkdat', '', name)
        filename = name
        filename_err = name+'_err'
    else:
        filename = name+"_"+MOMSTR+'_I'+str(ISOSPIN)
        filename_err = name+'_err'+"_"+MOMSTR+'_I'+str(ISOSPIN)
    print("writing file", filename)
    assert len(pickl_res) == 4, "bad result length"
    pickle.dump(pickl_res, open(filename+'.p', "wb"))
    print("writing file", filename_err)
    pickle.dump(pickl_res_err, open(filename_err+'.p', "wb"))
dump_fit_range.fn1 = None

@PROFILE
def divbychisq(param_arr, pvalue_arr):
    """Divide a parameter by chisq (t^2)"""
    assert not any(np.isnan(pvalue_arr)), "pvalue array contains nan"
    ret = np.array(param_arr)
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
            assert not any(np.isnan(param_arr)),\
                "parameter array contains nan"
        except AssertionError:
            for i in param_arr:
                print(i)
            sys.exit(1)
        ret *= pvalue_arr
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
    for i, fit in enumerate(min_arr):
        minmax_i = max(abs(fit[0].energy.val-result_min_avg))
        if i == 0:
            minmax = minmax_i
            ret_excl = fit[2]
        else:
            minmax = min(minmax_i, minmax)
            if minmax == minmax_i:
                ret_excl = fit[2]
    return ret_excl


@PROFILE
def errerr(param_err_arr):
    """Find the error on the parameter error."""
    err = np.zeros(param_err_arr[0].shape)
    avgerr = np.zeros(param_err_arr[0].shape)
    param_err_arr = np.asarray(param_err_arr)
    for i, _ in enumerate(err):
        err[i] = em.acstd(param_err_arr[:, i], ddof=1)/np.sqrt(
            len(err))/np.sqrt(MPISIZE)
        avgerr[i] = em.acmean(param_err_arr[:, i])
    return err, avgerr

# obsolete, we should simply pick the model with the smallest errors and an adequate chi^2 (t^2)
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
sfit.augment_excl = augment_excl

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

@PROFILE
def sum_files_total(basename, local_total):
    """Find a total by using temp files"""
    fn1 = open(basename+str(MPIRANK)+".temp", 'w')
    fn1.write(str(local_total))
    total = 0
    for i in range(MPISIZE):
        readname = basename+str(i)+".temp"
        touch(readname)
        gn1 = open(readname, "r")
        lines = gn1.readlines()
        if lines:
            total += int(lines[0])
    return total

def dofit_initial(meta, plotdata):
    """Do an initial test fit"""

    test_success = False
    retsingle_save = None
    try:
        retsingle_save = singlefit(meta.input_f, meta.fitwindow,
                                   meta.options.xmin,
                                   meta.options.xmax, meta.options.xstep)
        test_success = True
        if FIT:
            print("Test fit succeeded.")
    except XmaxError as err:
        print("Test fit failed; bad xmax.")
        test_success = False
        meta.options.xmax = err.problemx-meta.options.xstep
        print("xmin, new xmax =", meta.options.xmin, meta.options.xmax)
        if meta.fitwindow[1] > meta.options.xmax and FIT:
            print("***ERROR***")
            print("fit window beyond xmax:", meta.fitwindow)
            sys.exit(1)
        meta.fitwindow = fitrange_err(meta.options, meta.options.xmin,
                                      meta.options.xmax)
        print("new fit window = ", meta.fitwindow)
        plotdata.fitcoord = meta.fit_coord()
    except (NegChisq, RelGammaError, NoConvergence,
            OverflowError, EnergySortError, TooManyBadFitsError,
            np.linalg.linalg.LinAlgError, BadJackknifeDist,
            DOFNonPos, BadChisq, ZetaError) as err:
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
    samerange = sfit.cut_on_errsize(meta) # did we make any cuts?

    fit_range_init = str(latfit.config.FIT_EXCL)
    try:
        if not samerange and FIT:
            print("Trying second test fit.")
            print("fit excl:", fit_range_init)
            retsingle_save = singlefit(meta.input_f, meta.fitwindow,
                                       meta.options.xmin, meta.options.xmax,
                                       meta.options.xstep)
            print("Test fit succeeded.")
            test_success = True
    except (NegChisq, RelGammaError, OverflowError, NoConvergence,
            BadJackknifeDist, DOFNonPos, EnergySortError,
            TooManyBadFitsError,
            BadChisq, ZetaError) as err:
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
            if result_min.fun/result_min.dof >= 1 and SKIP_OVERFIT:
                min_arr.append(result)
            else:
                overfit_arr.append(result)
    return min_arr, overfit_arr, retsingle_save, fit_range_init

def store_init(plotdata):
    """Storage modification;
    act on info from initial fit ranges"""
    plotdata.coords, plotdata.cov = singlefit.coords_full, \
        singlefit.cov_full
    augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
    return plotdata


def dofit(meta, fit_range_data, results_store, plotdata):
    """Do a fit on a particular fit range"""

    # unpack
    min_arr, overfit_arr, retsingle_save = results_store
    idx, excl, fit_range_init = fit_range_data
    skip = False

    print("Trying fit with excluded times:",
          latfit.config.FIT_EXCL, "fit:",
          str(idx+1)+"/"+str(meta.lenprod))
    print("number of results:", len(min_arr),
          "number of overfit", len(overfit_arr),
          "rank:", MPIRANK)
    assert len(latfit.config.FIT_EXCL) == MULT, "bug"
    # retsingle_save needs a cut on error size
    if sfit.keyexcl(excl) == fit_range_init:
        skip = True
    else:
        try:
            retsingle = singlefit(meta.input_f, meta.fitwindow,
                                  meta.options.xmin, meta.options.xmax,
                                  meta.options.xstep)
            if retsingle_save is None:
                retsingle_save = retsingle
            print("fit succeeded for this selection"+\
                    " excluded points=", excl)
            if meta.lenprod == 1 or MAX_RESULTS == 1:
                retsingle_save = retsingle
        except (NegChisq, RelGammaError, OverflowError,
                NoConvergence, EnergySortError,
                TooManyBadFitsError,
                BadJackknifeDist,
                DOFNonPos, BadChisq, ZetaError) as err:
            # skip on any error
            print("fit failed for this selection"+\
                  " excluded points=", excl, "with error:",
                  err.__class__.__name__)
            skip = True
    if not skip:
        result_min, param_err, plotdata.coords, plotdata.cov = retsingle
        printerr(result_min.energy.val, param_err)
        if CALC_PHASE_SHIFT:
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

if __name__ == "__main__":
    main()
