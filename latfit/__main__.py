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
import subprocess as sp
import sys
from itertools import combinations, chain, product
from warnings import warn
import pickle
import numpy as np
import ast
import h5py
from mpi4py import MPI
import gvar

from latfit.singlefit import singlefit
import latfit.singlefit
import latfit.analysis.sortfit as sortfit
from latfit.config import JACKKNIFE, NOLOOP
from latfit.config import FIT, METHOD, T0
from latfit.config import ISOSPIN, MOMSTR, UNCORR
from latfit.config import ERR_CUT, PVALUE_MIN
from latfit.config import MATRIX_SUBTRACTION, DELTA_T_MATRIX_SUBTRACTION
from latfit.config import DELTA_T2_MATRIX_SUBTRACTION, DELTA_E2_AROUND_THE_WORLD
from latfit.config import GEVP, STYPE, SUPERJACK_CUTOFF, EFF_MASS
from latfit.config import MAX_ITER, BIASED_SPEEDUP, MAX_RESULTS
from latfit.config import CALC_PHASE_SHIFT, LATTICE_ENSEMBLE
from latfit.config import SKIP_OVERFIT
from latfit.jackknife_fit import jack_mean_err
from latfit.makemin.mkmin import convert_to_namedtuple
import latfit.extract.getblock

from latfit.procargs import procargs
from latfit.extract.errcheck.xlim_err import xlim_err
from latfit.extract.errcheck.xlim_err import fitrange_err
from latfit.extract.errcheck.xstep_err import xstep_err
from latfit.extract.errcheck.trials_err import trials_err
from latfit.extract.proc_folder import proc_folder
from latfit.finalout.printerr import printerr
from latfit.finalout.mkplot import mkplot
from latfit.makemin.mkmin import NegChisq
from latfit.extract.getblock import XmaxError
from latfit.utilities.zeta.zeta import RelGammaError, ZetaError
from latfit.jackknife_fit import DOFNonPos, BadChisq
from latfit.jackknife_fit import BadJackknifeDist, NoConvergence
from latfit.jackknife_fit import EnergySortError, TooManyBadFitsError
from latfit.config import MULT, RANGE_LENGTH_MIN 
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.config import PHASE_SHIFT_ERR_CUT, SKIP_LARGE_ERRORS
from latfit.config import ONLY_SMALL_FIT_RANGES
import latfit.config
import latfit.jackknife_fit
from latfit.utilities import exactmean as em

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
def filter_sparse(sampler, fitwindow, xstep=1):
    """Find the items in the power set which do not generate
    arithmetic sequences in the fitwindow powerset (sampler)
    """
    frange = np.arange(fitwindow[0], fitwindow[1]+xstep, xstep)
    retsampler = []
    for excl in sampler:
        excl = list(excl)
        fdel = list(filter(lambda a, sk=excl: a not in sk, frange))
        if len(fdel) < RANGE_LENGTH_MIN and not ONLY_SMALL_FIT_RANGES:
            continue
        if len(fdel) >= RANGE_LENGTH_MIN and ONLY_SMALL_FIT_RANGES:
            continue
        # start = fdel[0]
        incr = xstep if len(fdel) < 2 else fdel[1]-fdel[0]
        skip = False
        for i, timet in enumerate(fdel):
            if i == 0:
                continue
            if fdel[i-1] + incr != timet:
                skip = True
        if skip:
            continue
        retsampler.append(excl)
    return retsampler

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
    excl = augment_excl([[i for i in j] for j in excl])

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


class FitRangeMetaData:
    """Meta data about fit range loop"""
    @PROFILE
    def __init__(self):
        """Define meta data container."""
        self.skiploop = False
        self.lenprod = 0
        self.lenfit = 0
        self.xmin = 0
        self.xmax = np.inf
        self.xstep = 1
        self.fitwindow = []
        self.random_fit = True
        self.lenprod = 0

    @PROFILE
    def skip_loop(self):
        """Set the loop condition"""
        self.skiploop = False if self.lenprod > 1 else True
        self.skiploop = True if NOLOOP else self.skiploop
        if not self.random_fit and not self.skiploop:
            for excl in EXCL_ORIG:
                if len(excl) > 1:
                    #assert None
                    #self.skiploop = True
                    self.skiploop = False

    @PROFILE
    def generate_combinations(self):
        """Generate all possible fit ranges"""
        posexcl = powerset(
            np.arange(self.fitwindow[0],
                      self.fitwindow[1]+self.xstep, self.xstep))
        sampler = filter_sparse(posexcl, self.fitwindow, self.xstep)
        sampler = [list(EXCL_ORIG)] if NOLOOP else sampler
        posexcl = [sampler for i in range(len(latfit.config.FIT_EXCL))]
        prod = product(*posexcl)
        return prod, sampler

    @PROFILE
    def xmin_mat_sub(self):
        """Shift xmin to be later in time in the case of
        around the world subtraction of previous time slices"""
        ret = self.xmin
        delta = DELTA_T_MATRIX_SUBTRACTION
        if DELTA_E2_AROUND_THE_WORLD is not None:
            delta += DELTA_T2_MATRIX_SUBTRACTION
        delta = 0 if not MATRIX_SUBTRACTION else delta
        if GEVP:
            if self.xmin < delta + int(T0[6:]):
                ret = (delta + int(T0[6:]) + 1)* self.xstep
        self.xmin = ret

    @PROFILE
    def fit_coord(self):
        """Get xcoord to plot fit function."""
        return np.arange(self.fitwindow[0],
                         self.fitwindow[1]+self.xstep, self.xstep)


    @PROFILE
    def length_fit(self, prod, sampler):
        """Get length of fit window data"""
        self.lenfit = len(np.arange(self.fitwindow[0],
                                    self.fitwindow[1]+self.xstep,
                                    self.xstep))
        assert self.lenfit > 0 or not FIT, "length of fit range not > 0"
        self.lenprod = len(sampler)**(MULT)
        if NOLOOP:
            assert self.lenprod == 1, "Number of fit ranges is too large."
        latfit.config.MINTOL = True if self.lenprod == 0 else\
            latfit.config.MINTOL
        self.random_fit = True
        if self.lenprod < MAX_ITER: # fit range is small, use brute force
            self.random_fit = False
            prod = list(prod)
            prod = [str(i) for i in prod]
            prod = sorted(prod)
            prod = [ast.literal_eval(i) for i in prod]
            assert len(prod) == self.lenprod, "powerset length mismatch"+\
                " vs. expected length."
        return prod

    @PROFILE
    def actual_range(self):
        """Return the actual range spanned by the fit window"""
        ret = np.arange(self.fitwindow[0],
                        self.fitwindow[1]+self.xstep, self.xstep)
        ret = list(ret)
        latfit.jackknife_fit.WINDOW = ret
        return ret



@PROFILE
def main():
    """Main for latfit"""
    # set up 1ab
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep',
                                 'trials', 'fitmin', 'fitmax'])
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])

    meta = FitRangeMetaData()

    # error processing, parameter extractions
    input_f, options = procargs(sys.argv[1:])
    dump_fit_range.fn1 = str(input_f)
    meta.xmin, meta.xmax = xlim_err(options.xmin, options.xmax)
    latfit.extract.getblock.XMAX = meta.xmax
    meta.xstep = xstep_err(options.xstep, input_f)
    meta.xmin_mat_sub()
    meta.fitwindow = fitrange_err(options, meta.xmin, meta.xmax)
    meta.actual_range()
    print("fit window = ", meta.fitwindow)
    latfit.config.TSTEP = meta.xstep
    plotdata.fitcoord = meta.fit_coord()
    trials = trials_err(options.trials)
    if STYPE != 'hdf5':
        update_num_configs(input_f=(input_f if not GEVP else None))

    if trials == -1:
        # try an initial plot, shrink the xmax if it's too big
        print("Trying initial test fit.")
        test_success = False
        retsingle_save = None
        try:
            retsingle_save = singlefit(input_f, meta.fitwindow,
                                       meta.xmin, meta.xmax, meta.xstep)
            test_success = True
            if FIT:
                print("Test fit succeeded.")
            # do the fit range key processing here
            # since the initial fit augments the list
            fit_range_init = str(list(latfit.config.FIT_EXCL))
        except XmaxError as err:
            fit_range_init = None
            print("Test fit failed; bad xmax.")
            test_success = False
            meta.xmax = err.problemx-meta.xstep
            latfit.extract.getblock.XMAX = meta.xmax
            print("xmin, new xmax =", meta.xmin, meta.xmax)
            if meta.fitwindow[1] > meta.xmax and FIT:
                print("***ERROR***")
                print("fit window beyond xmax:", meta.fitwindow)
                sys.exit(1)
            meta.fitwindow = fitrange_err(options, meta.xmin, meta.xmax)
            print("new fit window = ", meta.fitwindow)
            plotdata.fitcoord = meta.fit_coord()
        except (NegChisq, RelGammaError, NoConvergence,
                OverflowError, EnergySortError, TooManyBadFitsError,
                np.linalg.linalg.LinAlgError, BadJackknifeDist,
                DOFNonPos, BadChisq, ZetaError) as err:
            print("fit failed (acceptably) with error:",
                  err.__class__.__name__)
            pass
        # update the known exclusion information with plot points
        # which are nan (not a number) or
        # which have error bars which are too large
        augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
        if FIT:
            # store different excluded, and the avg chisq/dof (t^2/dof)
            min_arr = []
            overfit_arr = [] # allow overfits if no usual fits succeed

            # generate all possible points excluded from fit range
            prod, sampler = meta.generate_combinations()

            # length of possibilities is useful to know,
            # update powerset if brute force solution is possible
            prod = meta.length_fit(prod, sampler)

            # cut late time points from the fit range
            samerange = cut_on_errsize(meta)

            fit_range_init = str(latfit.config.FIT_EXCL)
            try:
                if not samerange and FIT:
                    print("Trying second test fit.")
                    print("fit excl:", fit_range_init)
                    retsingle_save = singlefit(input_f, meta.fitwindow,
                                               meta.xmin, meta.xmax,
                                               meta.xstep)
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
                printerr(result_min.x, param_err)
                if CALC_PHASE_SHIFT:
                    print_phaseshift(result_min)
                if not cutresult(result_min, min_arr,
                                 overfit_arr, param_err):
                    result = [result_min, list(param_err),
                              list(latfit.config.FIT_EXCL)]
                    # don't overfit
                    if result_min.fun/result_min.dof >= 1 and SKIP_OVERFIT: 
                        min_arr.append(result)
                    else:
                        overfit_arr.append(result)


            augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
            plotdata.coords, plotdata.cov = singlefit.coords_full, \
                singlefit.cov_full
            tsorted = get_tsorted(plotdata)
            sorted_fit_ranges = sort_fit_ranges(meta, tsorted, sampler)

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
                if exitp(meta, min_arr, overfit_arr, idx):
                    break

                # get one fit range, check it
                excl, checked = get_one_fit_range(
                    meta, prod, idx, sorted_fit_ranges, checked)
                if excl is None:
                    continue
                if toosmallp(meta, excl):
                    print('excl:', excl, 'is too small')
                    continue

                # update global info about excluded points
                latfit.config.FIT_EXCL = excl

                # do fit
                START = time.perf_counter()
                print("Trying fit with excluded times:",
                      latfit.config.FIT_EXCL, "fit:",
                      str(idx+1)+"/"+str(meta.lenprod))
                print("number of results:", len(min_arr),
                      "number of overfit", len(overfit_arr),
                      "rank:", MPIRANK)
                assert len(latfit.config.FIT_EXCL) == MULT, "bug"
                # retsingle_save needs a cut on error size
                if keyexcl(excl) == fit_range_init:
                    continue
                else:
                    try:
                        retsingle = singlefit(input_f, meta.fitwindow,
                                              meta.xmin, meta.xmax,
                                              meta.xstep)
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
                        continue
                result_min, param_err, plotdata.coords, \
                    plotdata.cov = retsingle
                printerr(result_min.x, param_err)
                if CALC_PHASE_SHIFT:
                    print_phaseshift(result_min)
                    END = time.perf_counter()
                    print("Total elapsed time =", END-START, "seconds")


                if cutresult(result_min, min_arr, overfit_arr, param_err):
                    continue

                result = [result_min, list(param_err), list(excl)]

                if result_min.fun/result_min.dof >= 1: # don't overfit
                    min_arr.append(result)
                else:
                    overfit_arr.append(result)
                    continue

            if not meta.skip_loop:

                min_arr_send = np.array(min_arr)

                COMM_WORLD.barrier()
                min_arr = MPI.COMM_WORLD.gather(min_arr_send, 0)
                COMM_WORLD.barrier()
                print("results gather complete.")
                overfit_arr = MPI.COMM_WORLD.gather(overfit_arr, 0)
                COMM_WORLD.barrier()
                print("overfit gather complete.")

            if MPIRANK == 0:

                result_min = {}
                min_arr = loop_result(min_arr, overfit_arr)
                if not meta.skiploop:

                    result_min = find_mean_and_err(meta, min_arr)

                    # do the best fit again, with good stopping condition
                    # latfit.config.FIT_EXCL = min_excl(min_arr)
                    latfit.config.FIT_EXCL = closest_fit_to_avg(
                        result_min['x'], min_arr)
                    print("fit excluded points (indices):",
                          latfit.config.FIT_EXCL)

                if (not (meta.skiploop and latfit.config.MINTOL)\
                   and METHOD == 'NaN') or not test_success\
                   and (len(min_arr) + len(overfit_arr) > 1):
                        latfit.config.MINTOL = True
                        retsingle = singlefit(input_f, meta.fitwindow,
                                              meta.xmin, meta.xmax,
                                              meta.xstep)
                else:
                    retsingle = retsingle_save
                result_min_close, param_err_close, \
                    plotdata.coords, plotdata.cov = retsingle

                print_fit_results(meta, min_arr)
                result_min, param_err = combine_results(
                    result_min, result_min_close,
                    meta.skiploop, param_err, param_err_close)

                print("fit window = ", meta.fitwindow)
                # plot the result
                mkplot(plotdata, input_f, result_min,
                       param_err, meta.fitwindow)
        else:
            if MPIRANK == 0:
                if not latfit.config.MINTOL or METHOD == 'Nelder-Mead':
                    retsingle = singlefit(input_f, meta.fitwindow,
                                          meta.xmin, meta.xmax, meta.xstep)
                    plotdata.coords, plotdata.cov = retsingle
                else:
                    plotdata.coords, plotdata.cov = retsingle_save
                mkplot(plotdata, input_f)
    else:
        list_fit_params = []
        for ctime in range(trials):
            ifile = proc_folder(input_f, ctime, "blk")
            ninput = os.path.join(input_f, ifile)
            result_min, param_err, plotdata.coords, plotdata.cov =\
                singlefit(ninput, meta.fitwindow,
                          meta.xmin, meta.xmax, meta.xstep)
            list_fit_params.append(result_min.x)
        printerr(*get_fitparams_loc(list_fit_params, trials))
        sys.exit(0)
    print("END STDOUT OUTPUT")

@PROFILE
def combine_results(result_min, result_min_close,
                    skip_loop, param_err, param_err_close):
    """use the representative fit's goodness of fit in final print
    """
    if skip_loop:
        result_min, param_err = result_min_close, param_err_close
    else:
        result_min['fun'] = result_min_close.fun
        result_min['chisq_err'] = result_min_close.chisq_err
        result_min['dof'] = result_min_close.dof
        result_min['pvalue'] = result_min_close.pvalue
        result_min['pvalue_err'] = result_min_close.pvalue_err
        print("closest representative fit result (lattice units):")
        # convert here since we can't set attributes afterwards
        result_min = convert_to_namedtuple(result_min)
        printerr(result_min_close.x, param_err_close)
        print_phaseshift(result_min_close)
    return result_min, param_err


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
            if i == 0 and MPIRANK == 0:
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
        if i == 0 and MPIRANK == 0:
            print("Finding best times ("+\
                    "most likely to give small chi^2 (t^2) contributions)")
        if MULT > 1:
            coords = np.array([j[i] for j in plotdata.coords[:, 1]])
        else:
            coords = np.array([j for j in plotdata.coords[:, 1]])
        times = np.array(list(plotdata.coords[:, 0]))
        if MULT > 1:
            tsorted.append(sortfit.best_times(
                coords, plotdata.cov[:, :, i, i], i, times))
        else:
            tsorted.append(
                sortfit.best_times(coords, plotdata.cov, 0, times))
    return tsorted


@PROFILE
def loop_result(min_arr, overfit_arr):
    """Test if fit range loop succeeded"""
    if(min_arr):
        print(min_arr[0], np.array(min_arr).shape)
    min_arr = collapse_filter(min_arr)
    if(overfit_arr):
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
def excl_inrange(meta, excl):
    """Find the excluded points in the actual fit window"""
    ret = []
    fullrange = meta.actual_range()
    for i, exc in enumerate(excl):
        newexc = []
        for point in exc:
            if point in fullrange:
                newexc.append(point)
        ret.append(newexc)
    return ret

@PROFILE
def toosmallp(meta, excl):
    """Skip a fit range if it has too few points"""
    ret = False
    excl = excl_inrange(meta, excl)
    # each energy should be included
    if max([len(i) for i in excl]) >= meta.fitwindow[
            1]-meta.fitwindow[0]+meta.xstep:
        print("skipped all the data points for a GEVP dim, "+\
                "so continuing.")
        ret = True

    # each fit curve should be to more than one data point
    if not ret and meta.fitwindow[1]-meta.fitwindow[0] in [
            len(i) for i in excl] and\
            not ONLY_SMALL_FIT_RANGES:
        print("only one data point in fit curve, continuing")
        # ret = True
        pass
    if not ret and meta.fitwindow[1]-meta.fitwindow[0]-meta.xstep > 0 and\
        not ONLY_SMALL_FIT_RANGES:
        if meta.fitwindow[1]-meta.fitwindow[0]-1 in [len(i) for i in excl]:
            print("warning: only two data points in fit curve")
            # allow for very noisy excited states in I=0
            if not (ISOSPIN == 0 and GEVP):
                #ret = True
                pass
    #cut on arithmetic sequence
    if not ret and len(filter_sparse(
            excl, meta.fitwindow, xstep=meta.xstep)) != len(excl):
        print("not an arithmetic sequence")
        # ret = True
        pass
    ret = False if ISOSPIN == 0 and GEVP else ret
    return ret

@PROFILE
def print_phaseshift(result_min):
    """Print the phase shift info from a single fit range"""
    for i in range(MULT):
        if CALC_PHASE_SHIFT:
            print("phase shift of state #")
            if np.isreal(result_min.phase_shift[i]):
                  print(i, gvar.gvar(
                          result_min.phase_shift[i],
                          result_min.phase_shift_err[i]))
            else:
                temperr = result_min.phase_shift_err[i]
                try:
                    assert np.isreal(result_min.phase_shift_err[i]),\
                        "phase shift error is not real"
                except AssertionError:
                    temperr = np.imag(result_min.phase_shift_err[i])
                print(result_min.phase_shift[i], "+/-")
                print(result_min.phase_shift_err[i])
                #print(i, gvar.gvar(
                #          np.imag(result_min.phase_shift[i]),
                #          temperr), 'j')

@PROFILE
def cutresult(result_min, min_arr, overfit_arr, param_err):
    """Check if result of fit to a
    fit range is acceptable or not, return true if not acceptable
    (result should be recorded or not)
    """
    ret = False
    print("p-value = ", result_min.pvalue)
    # reject model at 10% level
    if result_min.pvalue < PVALUE_MIN:
        print("Not storing result because p-value"+\
                " is below rejection threshold. number"+\
                " of non-overfit results so far =", len(min_arr))
        print("number of overfit results =", len(overfit_arr))
        ret = True

    # is this justifiable?
    if not ret and skip_large_errors(result_min.x, param_err):
        print("Skipping fit range because param errors"+\
                " are greater than 100%")
        ret = True

    # is this justifiable?
    if not ret and CALC_PHASE_SHIFT and MULT > 1:
        if any(result_min.phase_shift_err > PHASE_SHIFT_ERR_CUT):
            if all(result_min.phase_shift_err[
                    :-1] < PHASE_SHIFT_ERR_CUT):
                print("warning: phase shift errors on "+\
                        "last state very large")
                ret = True if ISOSPIN == 2 and GEVP else ret
            else:
                print("phase shift errors too large")
                ret = True
    return ret

@PROFILE
def find_mean_and_err(meta, min_arr):
    """Find the mean and error from results of fit"""
    result_min = {}
    weight_sum = em.acsum([getattr(
        i[0], "pvalue_arr") for i in min_arr], axis=0)
    for name in min_arr[0][0].__dict__:
        if min_arr[0][0].__dict__[name] is None:
            print("name=", name, "is None, skipping")
            continue
        if '_err' in name:

            # find the name of the array
            avgname = re.sub('_err', '_arr', name)
            print("finding error in", avgname,
                  "which has shape=",
                  min_arr[0][0].__dict__[avgname].shape)
            assert min_arr[0][0].__dict__[avgname] is not None, \
                "Bad name substitution:"+str(avgname)

            # compute the jackknife errors as a check
            # (should give same result as error propagation)
            res_mean, err_check = jack_mean_err(em.acsum([
                divbychisq(getattr(i[0], avgname), getattr(
                    i[0], 'pvalue_arr')/weight_sum)\
                for i in min_arr], axis=0))

            # dump the results to file
            # if not (ISOSPIN == 0 and GEVP):
            dump_fit_range(meta, min_arr, avgname,
                           res_mean, err_check)

            # error propagation
            result_min[name] = em.acsum([
                jack_mean_err(
                    divbychisq(
                        getattr(
                            i[0], avgname),
                        getattr(
                            i[0], 'pvalue_arr')/weight_sum),
                    divbychisq(
                        getattr(
                            j[0],
                            avgname), getattr(
                                j[0],
                                'pvalue_arr')/weight_sum),
                    nosqrt=True)[1]
                for i in min_arr for j in min_arr], axis=0)
            try:
                result_min[name] = np.sqrt(result_min[name])
            except FloatingPointError:
                print("floating point error")
                if hasattr(result_min[name], '__iter__'):
                    for i, res in enumerate(result_min[name]):
                        if np.isreal(res):
                            if res < 0:
                                result_min[name][i] = np.nan
                else:
                    if np.isreal(result_min[name]):
                        if res < 0:
                            result_min[name] = np.nan
                result_min[name] = np.sqrt(result_min[name])

            # perform the comparison
            try:
                assert np.allclose(
                    err_check, result_min[name], rtol=1e-8)
            except AssertionError:
                print("jackknife error propagation"+\
                        " does not agree with jackknife"+\
                        " error.")
                print(result_min[name])
                print(err_check)
                if hasattr(err_check, '__iter__'):
                    for i, ress in enumerate(zip(
                            result_min[name], err_check)):
                        res1, res2 = ress
                        print(res1, res2, np.allclose(res1,res2, rtol=1e-8))
                        if not np.allclose(res1, res2, rtol=1e-8):
                            result_min[name][i] = np.nan
                            err_check[i] = np.nan

        # process this when we find the error name instead
        elif '_arr' in name:
            continue

        # find the weighted mean
        else:
            result_min[name] = em.acsum(
                [getattr(i[0], name)*getattr(i[0], 'pvalue')
                 for i in min_arr],
                axis=0)/em.acsum([
                    getattr(i[0],
                            'pvalue') for i in min_arr])
    # result_min.x = np.mean(
    # [i[0].x for i in min_arr], axis=0)
    # param_err = np.sqrt(np.mean([np.array(
    # i[1])**2 for i in min_arr], axis=0))
    # param_err = em.acstd([
    # getattr(i[0], 'x') for i in min_arr], axis=0, ddof=1)
    param_err = np.array(result_min['x_err'])
    assert not any(np.isnan(param_err)), \
        "A parameter error is not a number (nan)"
    return result_min

@PROFILE
def print_fit_results(meta, min_arr):
    """ Print the fit results
    """
    print("Fit results:  pvalue, energies, err on energies, included fit points")
    res = []
    for i in min_arr:
        res.append((getattr(i[0], "pvalue"), getattr(i[0], 'x'), i[1], inverse_excl(meta, i[2])))
    res = sorted(res, key=lambda x: x[0])
    for i in res:
        print(i)


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
def inverse_excl(meta, excl):
    """Get the included fit points from excluded points"""
    full = meta.actual_range()
    ret = [np.array(full) for _ in range(len(excl))]
    for i, excldim in enumerate(excl):
        try:
            inds = [full.index(i) for i in excldim]
        except ValueError:
            print("excluded point(s) is(are) not in fit range.")
            inds = []
            for j in excldim:
                if j not in full:
                    print("point is not in fit range:", j)
                else:
                    inds.append(int(full.index(j)))
        ret[i] = list(np.delete(ret[i], inds))
    return ret

@PROFILE
def compare_eff_mass_to_range(arr, err, errmin, mindim=None):
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
            err = erreff
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
    fname = "x_min_"+str(LATTICE_ENSEMBLE)
    if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
        fname = fname + '_'+dump_fit_range.fn1
    errname = 'x_err'
    err = np.array([getattr(i[0], errname) for i in min_arr])
    dimops = err.shape[1]
    if dimops == 1:
        err = err[:,0]
        errmin = min(err)
        ind = list(err).index(min(err))
        arr = getattr(min_arr[ind][0], 'x_arr')
    else:
        assert mindim is not None, "needs specification of operator"+\
            " dimension to write min error jackknife blocks (unsupported)."
        print(err.shape)
        errmin = min(err[:, mindim])
        ind = list(err[:, mindim]).index(errmin)
        fname = fname+'_mindim'+str(mindim)
        arr = np.asarray(getattr(min_arr[ind][0], 'x_arr')[:, mindim])
    arr, errmin = compare_eff_mass_to_range(arr, err, errmin, mindim=mindim)
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
        for key in singlefit.reuse.keys():
            if not isinstance(key, float) and not isinstance(key, int):
                continue
            if int(key) == key:
                times.append(key)
        times = sorted(times)
        for i, time in enumerate(times):
            if not isinstance(time, int) and not isinstance(time, float):
                continue
            if mindim is None:
                arr = singlefit.reuse[time]
                err = singlefit.error2[xcoord.index(time)]
            else:
                dimops = len(singlefit.reuse[time][0])\
                    if dimops is None else dimops
                assert dimops == len(singlefit.reuse[time][0])
                if not getavg:
                    arr = singlefit.reuse[time][:, mindim]
                    err = singlefit.error2[xcoord.index(time)][mindim]
                    assert isinstance(err, float), str(err)
                else:
                    arr = singlefit.reuse[time]
                    err = singlefit.error2[xcoord.index(time)]
            arrlist.append(arr)
            errlist.append(err)
        if not getavg:
            assert isinstance(errlist[0], float), str(errlist)+" "+str(singlefit.error2[xcoord.index(10)])
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
    def min_eff_mass_errors(mindim=None):
        """blank"""
        return (None, np.inf)

@PROFILE
def pickle_res(avgname, min_arr):
    """Return the fit range results to be pickled,
    append the effective mass points
    """
    ret = [getattr(i[0], avgname) for i in min_arr]
    origlshape = np.asarray(ret, dtype=object).shape
    print("res shape", origlshape)
    origl = len(ret)
    if 'x' in avgname:
        arreff, _ = min_eff_mass_errors(mindim=None, getavg=True)
        ret = [*ret, *arreff]
    ret = np.asarray(ret, dtype=object)
    assert len(origlshape) == len(ret.shape), str(origlshape)+","+str(ret.shape)
    flen = len(ret)
    print("original error length (res):", origl, "final error length:", flen)
    return ret

@PROFILE
def pickle_res_err(errname, min_arr):
    """Append the effective mass errors to the """
    ret = [getattr(i[0], errname) for i in min_arr]
    print("debug:[getattr(i[0], errname) for i in min_arr].shape", np.asarray(ret).shape)
    print("debug2:", np.asarray(singlefit.error2).shape)
    origl = len(ret)
    if GEVP and 'systematics' not in errname:
        if len(np.asarray(ret).shape) > 1:
            # dimops check
            assert (np.array(ret).shape)[1] ==\
                (np.asarray(singlefit.error2).shape)[1], str(np.array(ret).shape)+" "+str(np.asarray(singlefit.error2))+" "+str(errname)
    if 'x' in errname:
        _, erreff = min_eff_mass_errors(mindim=None, getavg=True)
        ret = np.array([*ret, *erreff])
    ret = np.asarray(ret)
    flen = len(ret)
    print("original error length (err):", origl, "final error length:", flen)
    return ret

@PROFILE
def pickle_excl(meta, min_arr):
    """Get the fit ranges to be pickled
    append the effective mass points
    """
    ret = [inverse_excl(meta, i[2]) for i in min_arr]
    print("original number of fit ranges before effective mass append:", len(ret))
    if EFF_MASS:
        xcoord = list(singlefit.coords_full[:, 0])
        xcoordapp = [[i] for i in xcoord]
        ret = [*ret, *xcoordapp]
    ret = np.array(ret, dtype=object)
    print("final fit range amount:", len(ret))
    return ret

@PROFILE
def dump_fit_range(meta, min_arr, avgname, res_mean, err_check):
    """Pickle the fit range result"""
    print("starting arg:", avgname)
    if 'x_arr' in avgname: # no clobber (only do this once)
        if MULT > 1:
            for i in range(len(res_mean)):
                dump_min_err_jackknife_blocks(min_arr, mindim=i)
        else:
            dump_min_err_jackknife_blocks(min_arr)
    errname = re.sub('_arr', '_err', avgname)
    avgname = re.sub('_arr', '', avgname)
    avgname = 'fun' if avgname == 'chisq' else avgname
    #pickl_res = [getattr(i[0], avgname)*getattr(i[0], 'pvalue')/em.acsum(
    #    [getattr(i[0], 'pvalue') for i in min_arr]) for i in min_arr]
    pickl_res = pickle_res(avgname, min_arr)
    pickl_res_err = pickle_res_err(errname, min_arr)
    pickl_excl = pickle_excl(meta, min_arr)
    pickl_res = np.array([res_mean, err_check,
                          pickl_res, pickl_excl], dtype=object)
    assert pickl_res_err.shape == pickl_res[2].shape, "array mismatch:"+\
        str(pickl_res_err.shape)+str(pickl_res[2].shape)
    avgname = 'chisq' if avgname == 'fun' else avgname
    avgname = 'tsq' if not UNCORR and avgname == 'chisq' else avgname
    if not GEVP:
        if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
            avgname = avgname+'_'+dump_fit_range.fn1
        avgname = re.sub('.jkdat', '', avgname)
        errname = avgname.replace('_', "_err_", 1)
        filename = avgname
        filename_err = errname
    else:
        filename = avgname+"_"+MOMSTR+'_I'+str(ISOSPIN)
        filename_err = errname+"_"+MOMSTR+'_I'+str(ISOSPIN)
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
    if start == str(latfit.config.FIT_EXCL):
        ret = True
    else:
        ret = False
    return ret

@PROFILE
def closest_fit_to_avg(result_min_avg, min_arr):
    """Find closest fit to average fit
    (find the most common fit range)
    """
    minmax = np.nan
    ret_excl = []
    for i, fit in enumerate(min_arr):
        minmax_i = max(abs(fit[0].x-result_min_avg))
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
            ret = abs(i/j) < 1
    return ret if SKIP_LARGE_ERRORS else False

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
def powerset(iterable):
    """powerset([1, 2, 3]) -->
    () (1, ) (2, ) (3, ) (1, 2) (1, 3) (2, 3) (1, 2, 3)"""
    siter = list(iterable)
    return chain.from_iterable(combinations(siter,
                                            r) for r in range(len(siter)+1))

@PROFILE
def update_num_configs(num_configs=None, input_f=None):
    """Update the number of configs in the case that FIT is False.
    """
    num_configs = -1 if num_configs is None else num_configs
    if not FIT and STYPE == 'hdf5' and num_configs == -1:
        infile = input_f if input_f is not None else\
            latfit.config.GEVP_DIRS[0][0]
        fn1 = h5py.File(infile, 'r')
        for i in fn1:
            if GEVP:
                for j in fn1[i]:
                    latfit.finalout.mkplot.NUM_CONFIGS = np.array(
                        fn1[i+'/'+j]).shape[0]
                    break
            else:
                latfit.finalout.mkplot.NUM_CONFIGS = np.array(
                    fn1[i]).shape[0]
            break
    elif num_configs != -1:
        latfit.finalout.mkplot.NUM_CONFIGS = num_configs


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


@PROFILE
def exitp(meta, min_arr, overfit_arr, idx):
    """Test to exit the fit range loop"""
    ret = False
    if meta.skiploop:
        print("skipping loop")
        ret = True

    if not ret and meta.random_fit:
        if len(min_arr) >= MAX_RESULTS/MPISIZE or (
                len(overfit_arr) >= MAX_RESULTS/MPISIZE
                and not min_arr):
            ret = True
            print("a reasonably large set of indices"+\
                " has been checked, exiting fit range loop."+\
                " (number of fit ranges checked:"+str(idx+1)+")")
            print("rank :", MPIRANK, "exiting fit loop")
    return ret

if __name__ == "__main__":
    main()


# obsolete
#errarr.append(param_err)
#curr_err, avg_curr_err = errerr(errarr)
#print("average statistical error on parameters",
#      avg_curr_err)
#stop = max(curr_err)/avg_curr_err[np.argmax(curr_err)]
#if stop < FITSTOP:
#    print("Estimate for parameter error has"+\
#          " stabilized, exiting loop")
#    break
#else:
#    print("Current error on error =", curr_err)

# need better criterion here,
# maybe just have it be user defined patience level?
# how long should the fit run before giving up?
# if result[0] < FITSTOP and random_fit:
# print("Fit is good enough.  Stopping search.")
# break
# else:
# print("min error so far:", result[0])
