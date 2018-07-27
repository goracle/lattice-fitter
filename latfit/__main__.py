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
from mpi4py import MPI

from latfit.singlefit import singlefit
import latfit.singlefit
import latfit.analysis.sortfit as sortfit
from latfit.config import JACKKNIFE, FIT_EXCL
from latfit.config import FIT
from latfit.config import MATRIX_SUBTRACTION, DELTA_T_MATRIX_SUBTRACTION
from latfit.config import GEVP, FIT, STYPE

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
from latfit.config import START_PARAMS, GEVP_DIRS, MULT
from latfit.config import FIT_EXCL as EXCL_ORIG
import latfit.config

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

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
    if GEVP:
        if xmin < DELTA_T_MATRIX_SUBTRACTION:
            ret = DELTA_T_MATRIX_SUBTRACTION + xstep
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
    xstep = xstep_err(options.xstep, input_f)
    xmin = xmin_mat_sub(xmin, xstep=xstep)
    fitrange = fitrange_err(options, xmin, xmax)
    print("fit range = ", fitrange)
    latfit.config.TSTEP = xstep
    plotdata.fitcoord = fit_coord(fitrange, xstep)
    trials = trials_err(options.trials)
    update_num_configs()

    if trials == -1:
        # try an initial plot, shrink the xmax if it's too big
        try:
            _ = singlefit(input_f, fitrange, xmin, xmax, xstep)
        except XmaxError as err:
            xmax = err.problemx-xstep
            print("xmin, new xmax =", xmin, xmax)
            if fitrange[1] > xmax and FIT:
                print("***ERROR***")
                print("fit range beyond xmax:", fitrange)
                sys.exit(1)
            fitrange = fitrange_err(options, xmin, xmax)
            print("new fit range = ", fitrange)
            plotdata.fitcoord = fit_coord(fitrange, xstep)
        except (NegChisq, RelGammaError,
                np.linalg.linalg.LinAlgError,
                DOFNonPos, BadChisqJackknife, ZetaError) as _:
            pass
        if FIT:
            # store different excluded, and the avg chisq/dof
            min_arr = []

            # generate all possible points excluded from fit range
            posexcl = powerset(
                np.arange(fitrange[0], fitrange[1]+xstep, xstep))
            sampler = filter_sparse(posexcl, fitrange, xstep)
            posexcl = [sampler for i in range(len(latfit.config.FIT_EXCL))]
            prod = product(*posexcl)

            # length of possibilities is useful to know
            lenfit = len(np.arange(fitrange[0], fitrange[1]+xstep, xstep))
            lenprod = len(sampler)**(MULT)
            random_fit = True
            if lenprod < 5000: # fit range is small, use brute force
                random_fit = False
                prod = list(prod)
                assert len(prod) == lenprod, "powerset length mismatch"+\
                    " vs. expected length."

            # now guess as to which time slices look the worst to fit
            # try the better ones first
            try:
                retsingle = singlefit(input_f,
                                        fitrange, xmin, xmax, xstep)
            except (NegChisq, RelGammaError, OverflowError,
                    np.linalg.linalg.LinAlgError,
                    DOFNonPos, BadChisqJackknife, ZetaError) as _:
                print("Test fit failed, but in an acceptable way. Continuing.")
            plotdata.coords, plotdata.cov = singlefit.coords_full, singlefit.cov_full
            tsorted = []
            for i in range(MULT):
                coords = np.array([j[i] for j in plotdata.coords[:,1]])
                times = np.array(list(plotdata.coords[:,0]))
                tsorted.append(sortfit.best_times(coords, plotdata.cov[:,:,i,i], i, times))
            samp_mult = []
            if random_fit:
                # go in a random order if lenprod is small (biased by how likely fit will succeed),
                for i in range(MULT):
                    probs, sampi = sortfit.sample_norms(sampler, tsorted[i])
                    samp_mult.append([probs, sampi])
            else:
                for i in range(MULT):
                    sampi = sortfit.sortcombinations(sampler, tsorted[i])
                    samp_mult.append(sampi)

            # store checked indicies
            checked = set()
            idx = -1

            # assume that manual spec. overrides brute force search
            skip_loop = False
            if not random_fit:
                for excl in FIT_EXCL:
                    if len(excl) > 0:
                        skip_loop = True

            for idx in range(lenprod):

                if skip_loop:
                    break

                if len(checked) == lenprod or idx == 5000:
                    print("a reasonably large set of indices"+\
                          " has been checked, exiting."+\
                          " (number of fit ranges checked:"+str(idx+1)+")")
                    break

                # parallelize loop
                if idx % MPISIZE != MPIRANK:
                    continue

                # small fit range
                key = None
                if not random_fit:
                    excl = prod[idx]
                    key = idx
                else: # large fit range, try to get lucky
                    if idx == 0:
                        excl = latfit.config.FIT_EXCL
                    else:
                        excl = [np.random.choice(samp_mult[i][0], p=samp_mult[i][1])
                                for i in range(len(latfit.config.FIT_EXCL))]
                    key = str(excl)
                if key in checked:
                    continue
                checked.add(key)

                # add user info
                excl = augment_excl([[i for i in j] for j in excl])

                # each fit curve should be to more than one data point
                if fitrange[1]-fitrange[0] in [len(i) for i in excl]:
                    continue

                # each energy should be included
                if max([len(i) for i in excl]) == fitrange[1]-fitrange[0]+1:
                    continue

                # dof check
                if not dof_check(lenfit, len(START_PARAMS), excl):
                    print("dof < 1 for excluded times:", excl,
                          "\nSkipping:", str(idx)+"/"+str(lenprod))
                    continue

                # update global info about excluded points
                latfit.config.FIT_EXCL = excl

                # do fit
                print("Trying fit with excluded times:",
                      latfit.config.FIT_EXCL, "fit:",
                      str(idx)+"/"+str(lenprod))
                try:
                    retsingle = singlefit(input_f,
                                          fitrange, xmin, xmax, xstep)
                except (NegChisq, RelGammaError, OverflowError,
                        np.linalg.linalg.LinAlgError,
                        DOFNonPos, BadChisqJackknife, ZetaError) as _:
                    # skip on any error
                    print("fit failed for this selection excluded points=",
                          excl)
                    continue
                result_min, param_err, plotdata.coords, plotdata.cov = retsingle
                printerr(result_min.x, param_err)

                # reject model at 10% level
                if result_min.pvalue < .1:
                    continue

                # calculate average relative error (to be minimized)
                result = (avg_relerr(result_min, param_err), excl)
                print("avg relative error, fit excl:", result, "dof=", result_min.dof)

                # store result
                if result_min.fun/result_min.dof >= 1: # don't overfit
                    min_arr.append(result)
                else:
                    continue

                # need better criterion here, maybe just have it be user defined patience level?
                # how long should the fit run before giving up?
                if result[0] < 0.0035 and random_fit:
                    print("Fit is good enough.  Stopping search.")
                    break
                else:
                    print("min error so far:", result[0])
                print("p-value = ", result_min.pvalue)
                
            if not skip_loop:

                min_arr = MPI.COMM_WORLD.gather(min_arr, 0)

            if MPIRANK == 0:
                if not skip_loop:

                    min_arr = [x for b in min_arr for x in b]
                    assert min_arr, "No fits succeeded."+\
                        "  Change fit range manually:"+str(min_arr)

                    print("Fit results:  red. chisq, excl")
                    for i in min_arr:
                        print(i)

                    # do the best fit again, with good stopping condition
                    latfit.config.FIT_EXCL =  min_excl(min_arr)
                latfit.config.MINTOL =  True
                retsingle = singlefit(input_f, fitrange, xmin, xmax, xstep)
                result_min, param_err, plotdata.coords, plotdata.cov = retsingle
                printerr(result_min.x, param_err)

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


# obsolete, we should simply pick the model with the smallest errors and an adequate chi^2
def min_excl(min_arr):
    """Find the minimum reduced chisq from all the fits considered"""
    minres = sorted(min_arr, key=lambda row: row[0])[0]
    print("min chisq/dof=", minres[0])
    print("best times to exclude:", minres[1])
    return minres[1]

def augment_excl(excl):
    """If the user has specified excluded indices add these to the list."""
    for num, (i, j) in enumerate(zip(excl, EXCL_ORIG)):
        excl[num] = sorted(list(set(j).union(set(i))))
    return excl

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

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def update_num_configs():
    """Update the number of configs in the case that FIT is False.
    """
    if not FIT and GEVP and STYPE == 'hdf5':
        fn1 = h5py.File(latfit.config.GEVP_DIRS[0][0], 'r')
        for i in fn1:
            for j in fn1[i]:
                latfit.finalout.mkplot.NUM_CONFIGS = np.array(
                    fn1[i+'/'+j]).shape[0]
                break
            break

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


if __name__ == "__main__":
    main()
