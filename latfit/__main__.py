#!/usr/bin/env python

"""Fit function to data.
Compute chi^2 and errors.
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
from math import sqrt
import sys
from itertools import combinations, chain, product
from random import randint
import subprocess as sp
from warnings import warn
import time
import random
import numpy as np
import h5py
import re
import pickle
from mpi4py import MPI
import gvar

from latfit.singlefit import singlefit
import latfit.singlefit
import latfit.analysis.sortfit as sortfit
from latfit.config import JACKKNIFE, FIT_EXCL, NOLOOP
from latfit.config import FIT
from latfit.config import ISOSPIN, MOMSTR
from latfit.config import ERR_CUT, PVALUE_MIN
from latfit.config import MATRIX_SUBTRACTION, DELTA_T_MATRIX_SUBTRACTION
from latfit.config import DELTA_T2_MATRIX_SUBTRACTION, DELTA_E2_AROUND_THE_WORLD
from latfit.config import GEVP, FIT, STYPE
from latfit.config import MAX_ITER, FITSTOP, BIASED_SPEEDUP, MAX_RESULTS
from latfit.config import CALC_PHASE_SHIFT
from latfit.jackknife_fit import ResultMin, jack_mean_err
import latfit.extract.getblock

from latfit.procargs import procargs
from latfit.extract.errcheck.xlim_err import xlim_err
from latfit.extract.errcheck.xlim_err import fitrange_err
from latfit.extract.errcheck.xstep_err import xstep_err
from latfit.extract.errcheck.trials_err import trials_err
from latfit.extract.proc_folder import proc_folder
from latfit.finalout.printerr import printerr, avg_relerr
from latfit.finalout.mkplot import mkplot
from latfit.makemin.mkmin import NegChisq
from latfit.extract.getblock import XmaxError
from latfit.utilities.zeta.zeta import RelGammaError, ZetaError
from latfit.jackknife_fit import DOFNonPos, BadChisqJackknife
from latfit.jackknife_fit import BadJackknifeDist, NoConvergence
from latfit.config import START_PARAMS, GEVP_DIRS, MULT
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.config import PHASE_SHIFT_ERR_CUT, SKIP_LARGE_ERRORS
import latfit.config

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

EXCL_ORIG = np.copy(EXCL_ORIG_IMPORT)

class Logger(object):
    """log output from fit"""
    def __init__(self):
        """initialize logger"""
        self.terminal = sys.stdout
        self.log = open("fit.log", "a")

    def write(self, message):
        """write to log"""
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass


sys.stdout = Logger()
sys.stderr = Logger()

def xmin_mat_sub(xmin, xstep=1):
    ret = xmin
    delta = DELTA_T_MATRIX_SUBTRACTION
    if DELTA_E2_AROUND_THE_WORLD is not None:
        delta += DELTA_T2_MATRIX_SUBTRACTION
    if GEVP and MATRIX_SUBTRACTION:
        if xmin < delta:
            ret = delta + xstep
    return ret


def setup_logger():
    """Setup the logger"""
    print("BEGIN NEW OUTPUT")
    timedate = time.asctime(time.localtime(time.time()))
    print(timedate)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gitlog = sp.check_output(['git', 'rev-parse', 'HEAD'])
    with open(cwd+'/config.log', 'a') as conflog:
        conflog.write("BEGIN NEW OUTPUT-------------\n")
        conflog.write(timedate+'\n')
        conflog.write("current git commit:"+str(gitlog)+'\n')
        filen = open(os.getcwd()+'/config.py', 'r')
        for line in filen:
            conflog.write(line)
        conflog.write("END OUTPUT-------------------\n")
    if len(gitlog.split()) == 1:
        print("current git commit:", gitlog)
    os.chdir(cwd)


def filter_sparse(sampler, fitrange, xstep=1):
    """Find the items in the power set which do not generate
    arithmetic sequences in the fitrange powerset (sampler)
    """
    frange = np.arange(fitrange[0], fitrange[1]+1, xstep)
    retsampler = []
    for excl in sampler:
        excl = list(excl)
        fdel = list(filter(lambda a: a not in excl, frange))
        if len(fdel) < 3:
            continue
        start = fdel[0]
        incr = fdel[1]-fdel[0]
        skip = False
        for i, time in enumerate(fdel):
            if i == 0:
                continue
            if fdel[i-1] + incr != time:
                skip = True
        if skip:
            continue
        retsampler.append(excl)
    return retsampler


def main():
    """Main for latfit"""
    setup_logger()
    # set up 1ab
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep',
                                 'trials', 'fitmin', 'fitmax'])
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])

    # error processing, parameter extractions
    input_f, options = procargs(sys.argv[1:])
    xmin, xmax = xlim_err(options.xmin, options.xmax)
    latfit.extract.getblock.XMAX = xmax
    xstep = xstep_err(options.xstep, input_f)
    xmin = xmin_mat_sub(xmin, xstep=xstep)
    fitrange = fitrange_err(options, xmin, xmax)
    print("fit range = ", fitrange)
    latfit.config.TSTEP = xstep
    plotdata.fitcoord = fit_coord(fitrange, xstep)
    trials = trials_err(options.trials)
    update_num_configs(input_f=(input_f if not GEVP else None))

    if trials == -1:
        # try an initial plot, shrink the xmax if it's too big
        print("Trying initial test fit.")
        test_success = False
        try:
            retsingle_save = singlefit(input_f, fitrange, xmin, xmax, xstep)
            test_success = True
            print("Test fit succeeded.")
            # do the fit range key processing here since the initial fit augments the list
            fit_range_init = str(list(latfit.config.FIT_EXCL))
        except XmaxError as err:
            fit_range_init = None
            print("Test fit failed; bad xmax.")
            test_success = False
            xmax = err.problemx-xstep
            latfit.extract.getblock.XMAX = xmax
            print("xmin, new xmax =", xmin, xmax)
            if fitrange[1] > xmax and FIT:
                print("***ERROR***")
                print("fit range beyond xmax:", fitrange)
                sys.exit(1)
            fitrange = fitrange_err(options, xmin, xmax)
            print("new fit range = ", fitrange)
            plotdata.fitcoord = fit_coord(fitrange, xstep)
        except (NegChisq, RelGammaError, NoConvergence,
                np.linalg.linalg.LinAlgError, BadJackknifeDist,
                DOFNonPos, BadChisqJackknife, ZetaError) as _:
            pass
        # update the known exclusion information with plot points
        # which are nan (not a number) or
        # which have error bars which are too large
        augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
        if FIT:
            # store different excluded, and the avg chisq/dof
            min_arr = []
            overfit_arr = [] # allow overfits if no usual fits succeed

            # generate all possible points excluded from fit range
            posexcl = powerset(
                np.arange(fitrange[0], fitrange[1]+xstep, xstep))
            sampler = filter_sparse(posexcl, fitrange, xstep)
            sampler = [list(EXCL_ORIG)] if NOLOOP else sampler
            posexcl = [sampler for i in range(len(latfit.config.FIT_EXCL))]
            prod = product(*posexcl)

            # length of possibilities is useful to know
            lenfit = len(np.arange(fitrange[0], fitrange[1]+xstep, xstep))
            assert lenfit > 0 or not FIT, "length of fit range not > 0"
            lenprod = len(sampler)**(MULT)
            if NOLOOP:
                assert lenprod == 1, "Number of fit ranges is too large."
            latfit.config.MINTOL =  True if lenprod == 0 else\
                latfit.config.MINTOL
            random_fit = True
            if lenprod < MAX_ITER: # fit range is small, use brute force
                random_fit = False
                prod = list(prod)
                assert len(prod) == lenprod, "powerset length mismatch"+\
                    " vs. expected length."

            # now guess as to which time slices look the worst to fit
            # try the better ones first
            if not test_success:
                fit_range_init = str(latfit.config.FIT_EXCL)
                try:
                    print("Trying test fit with improved xmax.")
                    retsingle_save = singlefit(input_f,
                                            fitrange, xmin, xmax, xstep)
                    print("Test fit succeeded.")
                except (NegChisq, RelGammaError, OverflowError, NoConvergence,
                        np.linalg.linalg.LinAlgError, BadJackknifeDist,
                        DOFNonPos, BadChisqJackknife, ZetaError) as _:
                    print("Test fit failed, but in an acceptable way. Continuing.")
                    fit_range_init = None
            cut_on_errsize()
            augment_excl.excl_orig = np.copy(latfit.config.FIT_EXCL)
            plotdata.coords, plotdata.cov = singlefit.coords_full, singlefit.cov_full
            tsorted = []
            for i in range(MULT):
                #if MULT == 1:
                #    break
                if i == 0 and MPIRANK == 0:
                    print("Finding best times (most likely to give small chi^2 contributions)")
                if MULT > 1:
                    coords = np.array([j[i] for j in plotdata.coords[:,1]])
                else:
                    coords = np.array([j for j in plotdata.coords[:,1]])
                times = np.array(list(plotdata.coords[:,0]))
                if MULT > 1:
                    tsorted.append(sortfit.best_times(coords, plotdata.cov[:,:,i,i], i, times))
                else:
                    tsorted.append(sortfit.best_times(coords, plotdata.cov, 0, times))
            samp_mult = []
            if random_fit:
                # go in a random order if lenprod is small (biased by how likely fit will succeed),
                for i in range(MULT):
                    #if MULT == 1:
                    #    break
                    if i == 0 and MPIRANK == 0 and BIASED_SPEEDUP:
                        print("Setting up biased sorting of (random) fit ranges")
                    probs, sampi = sortfit.sample_norms(
                        sampler, tsorted[i], lenfit)
                    probs = probs if BIASED_SPEEDUP else None
                    samp_mult.append([probs, sampi])
            else:
                for i in range(MULT):
                    if lenprod == 1:
                        break
                    #if MULT == 1 or lenprod == 1:
                    #    break
                    if i == 0 and MPIRANK == 0:
                        print("Setting up sorting of exhaustive list of fit ranges")
                    sampi = sortfit.sortcombinations(
                        sampler, tsorted[i], lenfit)
                    samp_mult.append(sampi)

            # store checked fit ranges
            checked = set()

            # running error on parameter error
            errarr = []

            # assume that manual spec. overrides brute force search
            skip_loop = False if lenprod > 1 else True
            if not random_fit:
                for excl in EXCL_ORIG:
                    if len(excl) > 1:
                        assert False
                        skip_loop = True
            #if MULT == 1:
            #    skip_loop = True

            print("starting loop of max length:"+str(lenprod))
            for idx in range(lenprod):

                if skip_loop:
                    print("skipping loop")
                    break

                if random_fit:

                    if len(min_arr) > MAX_RESULTS/MPISIZE or (
                            len(overfit_arr) > MAX_RESULTS/MPISIZE and len(
                                min_arr) == 0):
                        print("a reasonably large set of indices"+\
                              " has been checked, exiting fit range loop."+\
                              " (number of fit ranges checked:"+str(idx+1)+")")
                        print("rank :", MPIRANK, "exiting fit loop")
                        break


                # parallelize loop
                if idx % MPISIZE != MPIRANK:
                    print("mpi skip")
                    continue

                # small fit range
                key = None
                if not random_fit:
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

                key = str(list(excl))
                if key in checked:
                    print("key checked, continuing")
                    continue
                checked.add(key)

                # each energy should be included
                if max([len(i) for i in excl]) == fitrange[1]-fitrange[0]+1:
                    print("skipped all the data points for a GEVP dim,"+\
                          "so continuing.")
                    continue

                # each fit curve should be to more than one data point
                if fitrange[1]-fitrange[0] in [len(i) for i in excl]:
                    print("only one data point in fit curve, continuing")
                    continue
                if fitrange[1]-fitrange[0]-1 > 0:
                    if fitrange[1]-fitrange[0]-1 in [len(i) for i in excl]:
                        print("warning: only two data points in fit curve")
                        # continue

                # update global info about excluded points
                latfit.config.FIT_EXCL = excl

                # do fit
                print("Trying fit with excluded times:",
                      latfit.config.FIT_EXCL, "fit:",
                      str(idx+1)+"/"+str(lenprod))
                print("number of results:", len(min_arr),
                      "number of overfit", len(overfit_arr))
                assert len(latfit.config.FIT_EXCL) == MULT, "bug"
                if key == fit_range_init:
                    retsingle = retsingle_save
                else:
                    try:
                        retsingle = singlefit(input_f,
                                            fitrange, xmin, xmax, xstep)
                    except (NegChisq, RelGammaError, OverflowError, NoConvergence,
                            np.linalg.linalg.LinAlgError, BadJackknifeDist,
                            DOFNonPos, BadChisqJackknife, ZetaError) as _:
                        # skip on any error
                        print("fit failed for this selection"+\
                              " excluded points=", excl)
                        continue
                result_min, param_err, plotdata.coords, plotdata.cov = retsingle
                printerr(result_min.x, param_err)

                print("p-value = ", result_min.pvalue)

                # reject model at 10% level
                if result_min.pvalue < PVALUE_MIN:
                    print("Not storing result because p-value"+\
                          " is below rejection threshold. number"+\
                          " of non-overfit results so far =", len(min_arr))
                    print("number of overfit results =", len(overfit_arr))
                    continue

                # is this justifiable?
                if skip_large_errors(result_min.x, param_err):
                    print("Skipping fit range because param errors"+\
                          " are greater than 100%")
                    continue

                # is this justifiable?
                if CALC_PHASE_SHIFT and MULT > 1:
                    if any(result_min.phase_shift_err > PHASE_SHIFT_ERR_CUT):
                        print("phase shift errors too large")
                        continue


                # calculate average relative error (to be minimized)
                # result = (avg_relerr(result_min, param_err), excl)
                # print("avg relative error, fit excl:", result,
                # "dof=", result_min.dof)
                result = [result_min, list(param_err), list(excl)]

                # store result
                if result_min.fun/result_min.dof >= 1: # don't overfit
                    min_arr.append(result)
                else:
                    overfit_arr.append(result)
                    continue

                errarr.append(param_err)
                curr_err, avg_curr_err = errerr(errarr)
                print("average statistical error on parameters",
                      avg_curr_err)
                stop = max(curr_err)/avg_curr_err[np.argmax(curr_err)]
                if stop < FITSTOP:
                    print("Estimate for parameter error has"+\
                          " stabilized, exiting loop")
                    break
                else:
                    print("Current error on error =", curr_err)

                # need better criterion here,
                # maybe just have it be user defined patience level?
                # how long should the fit run before giving up?
                # if result[0] < FITSTOP and random_fit:
                    # print("Fit is good enough.  Stopping search.")
                    # break
                # else:
                   # print("min error so far:", result[0])

                
            if not skip_loop:

                min_arr = MPI.COMM_WORLD.gather(min_arr, 0)
                print("results gather complete.")
                overfit_arr = MPI.COMM_WORLD.gather(overfit_arr, 0)
                print("overfit gather complete.")

            if MPIRANK == 0:

                result_min = {}
                if not skip_loop:

                    # collapse the array structure introduced by mpi
                    min_arr = [x for b in min_arr for x in b]
                    overfit_arr = [x for b in overfit_arr for x in b]

                    # filter out duplicated work (find unique results)
                    min_arr = getuniqueres(min_arr)
                    overfit_arr = getuniqueres(overfit_arr)

                    try:
                        assert len(min_arr) > 0, "No fits succeeded."+\
                            "  Change fit range manually:"+str(min_arr)
                    except AssertionError:
                        min_arr = overfit_arr
                        assert len(overfit_arr) > 0, "No fits succeeded."+\
                            "  Change fit range manually:"+str(min_arr)

                    print("Fit results:  red. chisq, excl")
                    for i in min_arr:
                        print(i[1:])

                    weight_sum = np.sum([getattr(i[0], "pvalue_arr") for i in min_arr], axis=0)
                    for name in min_arr[0][0].__dict__:
                        if min_arr[0][0].__dict__[name] is None:
                            print("name=", name, "is None, skipping")
                            continue
                        if '_err' in name:

                            # find the name of the array
                            avgname = re.sub('_err', '_arr', name)
                            print("finding error in", avgname, "which has shape=",
                                  min_arr[0][0].__dict__[avgname].shape)
                            assert min_arr[0][0].__dict__[avgname] is not None,\
                                "Bad name substitution:"+str(avgname)

                            # compute the jackknife errors as a check
                            # (should give same result as error propagation)
                            res_mean, err_check = jack_mean_err(np.sum([divbychisq(
                                getattr(i[0], avgname), getattr(i[0], 'pvalue_arr')/weight_sum)
                                                              for i in min_arr], axis=0))

                            # dump the results to file
                            dump_fit_range(min_arr, weight_sum, avgname, res_mean, err_check)

                            # error propagation
                            result_min[name] = np.sqrt(np.sum([
                                jack_mean_err(
                                    divbychisq(getattr(i[0], avgname), getattr(i[0], 'pvalue_arr')/weight_sum),
                                    divbychisq(getattr(j[0], avgname), getattr(j[0], 'pvalue_arr')/weight_sum),
                                    nosqrt=True)[1]
                                for i in min_arr for j in min_arr], axis=0))

                            # perform the comparison
                            try:
                                assert np.allclose(err_check, result_min[name], rtol=1e-8)
                            except AssertionError:
                                print("jackknife error propagation does not agree with jackknife error.") 
                                print(result_min[name])
                                print(err_check)
                                sys.exit(1)

                        # process this when we find the error name instead
                        elif '_arr' in name:
                            continue

                        # find the weighted mean
                        else:
                            result_min[name] = np.sum([
                                getattr(i[0], name)*getattr(i[0], 'pvalue') for i in min_arr], axis=0)/np.sum(
                                    [getattr(i[0], 'pvalue') for i in min_arr])
                    # result_min.x = np.mean(
                    # [i[0].x for i in min_arr], axis=0)
                    # param_err = np.sqrt(np.mean([np.array(i[1])**2 for i in min_arr], axis=0))
                    # param_err = np.std([getattr(i[0], 'x') for i in min_arr], axis=0, ddof=1)
                    param_err = np.array(result_min['x_err'])
                    assert not any(np.isnan(param_err)), "A parameter error is not a number (nan)"

                    # do the best fit again, with good stopping condition
                    # latfit.config.FIT_EXCL =  min_excl(min_arr)
                    latfit.config.FIT_EXCL = closest_fit_to_avg(
                        result_min['x'], min_arr)
                    print("fit excluded points (indices):", latfit.config.FIT_EXCL)

                if not (skip_loop and latfit.config.MINTOL):
                    latfit.config.MINTOL =  True
                    retsingle = singlefit(input_f, fitrange,
                                          xmin, xmax, xstep)
                else:
                    retsingle = retsingle_save
                result_min_close, param_err_close, plotdata.coords, plotdata.cov = retsingle

                # use the representative fit's goodness of fit in final print

                result_min['fun'] = result_min_close.fun
                result_min['chisq_err'] = result_min_close.chisq_err
                result_min['dof'] = result_min_close.dof
                result_min['pvalue'] = result_min_close.pvalue
                result_min['pvalue_err'] = result_min_close.pvalue_err

                result_min = convert_to_namedtuple(result_min)

                print("closest representative fit result (lattice units):")
                printerr(result_min_close.x, param_err_close)
                for i in range(MULT):
                    if CALC_PHASE_SHIFT:
                        print("phase shift of state #",
                              i, gvar.gvar(result_min_close.phase_shift[i],
                              result_min_close.phase_shift_err[i]))

                if skip_loop:
                    result_min, param_err = result_min_close, param_err_close

                # plot the result
                mkplot(plotdata, input_f, result_min, param_err, fitrange)
        else:
            if MPIRANK == 0:
                retsingle = singlefit(input_f, fitrange, xmin, xmax, xstep)
                plotdata.coords, plotdata.cov = retsingle
                mkplot(plotdata, input_f)
    else:
        list_fit_params = []
        for ctime in range(trials):
            ifile = proc_folder(input_f, ctime, "blk")
            ninput = os.path.join(input_f, ifile)
            result_min, param_err, plotdata.coords, plotdata.cov = singlefit(
                ninput, fitrange, xmin, xmax, xstep)
            list_fit_params.append(result_min.x)
        printerr(*get_fitparams_loc(list_fit_params, trials))
        sys.exit(0)
    print("END STDOUT OUTPUT")
    warn("END STDERR OUTPUT")

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
            

def dump_fit_range(min_arr, weight_sum, avgname, res_mean, err_check):
    """Pickle the fit range result
    """
    name = re.sub('_arr', '_err', avgname)
    avgname = re.sub('_arr', '', avgname)
    avgname = 'fun' if avgname == 'chisq' else avgname
    #pickl_res = [getattr(i[0], avgname)*getattr(i[0], 'pvalue')/np.sum(
    #    [getattr(i[0], 'pvalue') for i in min_arr]) for i in min_arr]
    pickl_res = [getattr(i[0], avgname) for i in min_arr]
    pickl_res = np.array([res_mean, err_check, np.array(pickl_res)],
                         dtype=object)
    pickl_res_err = np.array([getattr(i[0], name) for i in min_arr])
    avgname = 'chisq' if avgname == 'fun' else avgname
    pickle.dump(pickl_res, open(
        avgname+"_"+MOMSTR+'_I'+str(ISOSPIN)+'.p', "wb"))
    pickle.dump(pickl_res_err, open(
        name+"_"+MOMSTR+'_I'+str(ISOSPIN)+'.p', "wb"))

def divbychisq(param_arr, pvalue_arr):
    """Divide a parameter by chisq"""
    assert not any(np.isnan(pvalue_arr)), "pvalue array contains nan"
    ret = np.array(param_arr)
    if len(ret.shape) > 1:
        assert param_arr[:, 0].shape == pvalue_arr.shape, "Mismatch between pvalue_arr"+\
            " and parameter array (should be the number of configs):"+\
            str(pvalue_arr.shape)+","+str(param_arr.shape)
        for i in range(len(ret[0])):
            try:
                assert not any(np.isnan(param_arr[:, i])), "parameter array contains nan"
            except AssertionError:
                print("found nan in dimension :", i)
                for j in param_arr:
                    print(j)
                sys.exit(1)
            ret[:, i] *= pvalue_arr
            assert not any(np.isnan(ret[:, i])), "parameter array contains nan"
    else:
        try:
            assert not any(np.isnan(param_arr)), "parameter array contains nan"
        except AssertionError:
            for i in param_arr:
                print(i)
            sys.exit(1)
        ret *= pvalue_arr
    assert ret.shape == param_arr.shape, "return shape does not match input shape"
    return ret
        
    

def convert_to_namedtuple(dictionary):
    return namedtuple('min', dictionary.keys())(**dictionary)


def cut_on_errsize():
    """Cut on the size of the error bars on individual points"""
    err = singlefit.error2
    coords = singlefit.coords_full
    assert singlefit.error2 is not None, "Bug in the acquiring error bars"
    #assert GEVP, "other versions not supported yet"+str(err.shape)+" "+str(coords.shape)
    for i in range(len(coords)):
        excl_add = coords[i][0]
        if MULT > 1:
            for j in range(len(coords[0][1])):
                if err[i][j]/coords[i][1][j] > ERR_CUT:
                    print("err =", err[i][j], "coords =", coords[i][1][j])
                    print("cutting dimension", j, "for time slice", excl_add)
                    print("err/coords > ERR_CUT =", ERR_CUT)
                    latfit.config.FIT_EXCL[j].append(excl_add)
                    latfit.config.FIT_EXCL[j] = list(set(
                        latfit.config.FIT_EXCL[j]))
        else:
            if err[i]/coords[i][1] > ERR_CUT:
                print("err =", err[i], "coords =", coords[i][1])
                print("cutting dimension", j, "for time slice", excl_add)
                print("err/coords > ERR_CUT =", ERR_CUT)
                latfit.config.FIT_EXCL[0].append(excl_add)
                latfit.config.FIT_EXCL[0] = list(set(
                    latfit.config.FIT_EXCL[0]))
                

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


def errerr(param_err_arr):
    """Find the error on the parameter error."""
    err = np.zeros(param_err_arr[0].shape)
    avgerr = np.zeros(param_err_arr[0].shape)
    param_err_arr = np.asarray(param_err_arr)
    for i in range(len(err)):
        err[i] = np.std(param_err_arr[:, i], ddof=1)/np.sqrt(len(err))/np.sqrt(MPISIZE)
        avgerr[i] = np.mean(param_err_arr[:, i])
    return err, avgerr
    

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
            assert j > 0, "negative error found:"+\
                str(result_param)+" "+str(param_err)
            ret = abs(i/j) < 1
    return ret if SKIP_LARGE_ERRORS else False
            
    

# obsolete, we should simply pick the model with the smallest errors and an adequate chi^2
def min_excl(min_arr):
    """Find the minimum reduced chisq from all the fits considered"""
    minres = sorted(min_arr, key=lambda row: row[0])[0]
    print("min chisq/dof=", minres[0])
    print("best times to exclude:", minres[1])
    return minres[1]

def augment_excl(excl):
    """If the user has specified excluded indices add these to the list."""
    for num, (i, j) in enumerate(zip(excl, augment_excl.excl_orig)):
        excl[num] = sorted(list(set(j).union(set(i))))
    return excl
augment_excl.excl_orig = EXCL_ORIG

def dof_check(lenfit, dimops, excl):
    """Check the degrees of freedom.  If < 1, cause a skip"""
    dof = (lenfit-1)*dimops
    ret = True
    for i in excl:
        for j in i:
            dof -= 1
    if dof < 1:
        ret = False
    return ret

# dof check (broken)
# if not dof_check(lenfit, len(START_PARAMS), excl):
#    print("dof < 1 for excluded times:", excl,
#          "\nSkipping:", str(idx)+"/"+str(lenprod))
#    continue



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def update_num_configs(num_configs=None, input_f=None):
    """Update the number of configs in the case that FIT is False.
    """
    num_configs = -1 if num_configs is None else num_configs
    if not FIT and STYPE == 'hdf5' and num_configs == -1:
        infile = input_f if input_f is not None else latfit.config.GEVP_DIRS[0][0]
        fn1 = h5py.File(infile, 'r')
        for i in fn1:
            for j in fn1[i]:
                latfit.finalout.mkplot.NUM_CONFIGS = np.array(
                    fn1[i+'/'+j]).shape[0]
                break
            break
    elif num_configs != -1:
        latfit.finalout.mkplot.NUM_CONFIGS = num_configs
    
def fit_coord(fitrange, xstep):
    """Get xcoord to plot fit function."""
    return np.arange(fitrange[0], fitrange[1]+xstep, xstep)


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
def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags, mode=mode, dir_fd=dir_fd)) as f:
        os.utime(f.fileno() if os.utime in os.supports_fd else fname,
            dir_fd=None if os.supports_fd else dir_fd, **kwargs)

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
        if len(lines) != 0:
            total += int(lines[0])
    return total


if __name__ == "__main__":
    main()
