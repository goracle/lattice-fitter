"""Standard fit branch"""
import sys
from collections import namedtuple
from numpy import sqrt
import numpy as np

#package modules
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.makemin.dof_errchk import dof_errchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr
from latfit.mathfun.covinv_avg import covinv_avg
from latfit.jackknife_fit import jackknife_fit
from latfit.analysis.get_fit_params import get_fit_params

#import global variables
from latfit.config import FIT
from latfit.config import JACKKNIFE_FIT
from latfit.config import JACKKNIFE
from latfit.config import PRINT_CORR
from latfit.config import GEVP

def singlefit(input_f, fitrange, xmin, xmax, xstep):
    """Get data to fit
    and minimized params for the fit function (if we're fitting)
    """
    #test to see if file/folder exists
    inputexists(input_f)

    ####process the file(s)
    coords_full, cov_full, reuse = extract(
        input_f, xmin, xmax, xstep)
    #do this so reuse goes from reuse[time][config] to more convenient reuse[config][time]

    ##Now that we have the data to fit, do pre-proccess it
    params = namedtuple('fit_params', ['dimops', 'num_configs', 'prefactor'])
    params = get_fit_params(cov_full, reuse, xmin)

    #correct covariance matrix for jackknife factor
    cov_full *= params.prefactor

    #debug branch
    if PRINT_CORR:
        print(coords)
        if GEVP:
            print([sqrt(np.diag(cov[i][i])) for i in range(len(cov))])
        else:
            print([sqrt(cov[i][i]) for i in range(len(cov))])
        sys.exit(0)

    #select part of data to fit
    start_index = int((fitrange[0]-xmin)/xstep)
    stop_index = int(len(coords_full)-1-(xmax-fitrange[1])/xstep)
    time_range = np.arange(fitrange[0], fitrange[1]+1, xstep)

    coords = coords_full[start_index:stop_index+1]
    cov = cov_full[start_index:stop_index+1, start_index:stop_index+1]
    #reuse = swap(reuse, 0, 1), turn it into an array
    reuse = np.array([[reuse[time][config]
                       for time in time_range]
                      for config in range(params.num_configs)])

    #error handling for Degrees of Freedom <= 0 (it should be > 0).
    #number of points plotted = len(cov).
    #DOF = len(cov) - START_PARAMS
    dof_errchk(len(cov), params.dimops)

    ###we have data 6ab
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    if FIT:
        covinv = covinv_avg(cov, params.dimops)

    if GEVP:
        print("(Rough) scale of errors in data points = ",
              np.diag(sqrt(cov[0][0])))
    else:
        print("(Rough) scale of errors in data points = ", sqrt(cov[0][0]))

    if FIT:
        if JACKKNIFE_FIT and JACKKNIFE == 'YES':
            result_min, param_err = jackknife_fit(params, reuse, coords, time_range, covinv)
        else:
            result_min = mkmin(covinv, coords)
            ####compute errors 8ab, print results (not needed for plot part)
            param_err = geterr(result_min, covinv, coords)

        #use a consistent error bar scheme;
        #if fitrange isn't max use conventional,
        #otherwise use the new double jackknife estimate
        if xmin != fitrange[0] or xmax != fitrange[1]:
            result_min.error_bars = None
        return result_min, param_err, coords_full, cov_full
    else:
        return coords, cov
