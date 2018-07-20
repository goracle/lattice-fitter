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

from latfit.singlefit import singlefit
from latfit.config import JACKKNIFE
from latfit.config import FIT
from latfit.config import MATRIX_SUBTRACTION, DELTA_T_MATRIX_SUBTRACTION
from latfit.config import GEVP, FIT, STYPE

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
from latfit.jackknife_fit import DOFNonPos, BadChisqJackknife
from latfit.config import GEVP_DIRS
from latfit.config import FIT_EXCL as EXCL_ORIG
import latfit.config


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
            fitrange = fitrange_err(options, xmin, xmax)
            print("new fit range = ", fitrange)
            plotdata.fitcoord = fit_coord(fitrange, xstep)
        except (NegChisq, RelGammaError,
                np.linalg.linalg.LinAlgError,
                DOFNonPos, BadChisqJackknife, ZetaError) as _:
            pass
        if FIT:
            chisq_arr = []
            if not skip:
                result_min, param_err, plotdata.coords, plotdata.cov = retsingle
                chisq_arr = [(result_min.fun, latfit.config.FIT_EXCL)]
            posexcl = [powerset(np.arange(fitrange[0], fitrange[1]+xstep, xstep)) for i in range(len(latfit.config.FIT_EXCL))]
            prod = product(*posexcl)
            lenfit = len(np.arange(fitrange[0], fitrange[1]+xstep, xstep))
            lenprod = 2**(len(GEVP_DIRS)*lenfit)
            for i, excl in enumerate(prod):
                excl = augment_excl([[i for i in j] for j in excl])
                if not dof_check(lenfit, len(GEVP_DIRS), excl):
                    print("dof < 1 for excluded times:", excl, "Skipping:", str(i)+"/"+str(lenprod))
                    continue
                latfit.config.FIT_EXCL = excl
                print("Trying fit with excluded times:",
                      latfit.config.FIT_EXCL, "fit:", str(i)+"/"+str(lenprod))
                try:
                    retsingle = singlefit(input_f, fitrange, xmin, xmax, xstep)
                except (NegChisq, RelGammaError, np.linalg.linalg.LinAlgError,
                        DOFNonPos, BadChisqJackknife, ZetaError) as _:
                    print("fit failed for this selection excluded points=", excl)
                    continue
                result_min, param_err, plotdata.coords, plotdata.cov = retsingle
                printerr(result_min.x, param_err)
                try:
                    result = (result_min.fun/result_min.dof, excl)
                except ZeroDivisionError:
                    print("infinite chisq/dof. fit excl:", excl)
                    continue
                print("chisq/dof, fit excl:", result, "dof=", result_min.dof)
                if result[0] >= 1: # don't overfit
                    chisq_arr.append(result)
            assert chisq_arr, "No fits succeeded.  Change fit range manually."
            for i in chisq_arr:
                print(i)
            latfit.config.FIT_EXCL =  min_excl(chisq_arr)
            latfit.config.MINTOL =  True
            retsingle = singlefit(input_f, fitrange, xmin, xmax, xstep)
            result_min, param_err, plotdata.coords, plotdata.cov = retsingle
            printerr(result_min.x, param_err)
            mkplot(plotdata, input_f, result_min, param_err, fitrange)
        else:
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

def min_excl(chisq_arr):
    minres = sorted(chisq_arr, key=lambda row: row[0])[0]
    print("min chisq/dof=", minres[0])
    print("best times to exclude:", minres[1])
    return minres[1]

def augment_excl(excl):
    for num, (i, j) in enumerate(zip(excl, EXCL_ORIG)):
        excl[num] = sorted(list(set(j).union(set(i))))
    return excl

def dof_check(lenfit, dimops, excl):
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
