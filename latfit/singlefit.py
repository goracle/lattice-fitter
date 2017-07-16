"""Standard fit branch"""
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

def singlefit(input_f, xmin, xmax, xstep):
    """Get data to fit
    and minimized params for the fit function (if we're fitting)
    """
    #test to see if file/folder exists
    inputexists(input_f)

    ####process the file(s)
    coords, cov, reuse = extract(input_f, xmin, xmax, xstep)
    #do this so reuse goes from reuse[time][config] to more convenient reuse[config][time]

    ##Now that we have the data to fit, do pre-proccess it
    params = namedtuple('fit_params', ['dimops', 'num_configs', 'prefactor'])
    params = get_fit_params(cov, reuse, xmin)

    time_range = np.arange(xmin, xmax+1, xstep)
    #reuse = swap(reuse, 0, 1)
    reuse = np.array([[reuse[time][config]
                       for time in time_range]
                      for config in range(params.num_configs)])
    cov *= params.prefactor

    #error handling for Degrees of Freedom <= 0 (it should be > 0).
    #number of points plotted = len(cov).
    #DOF = len(cov) - START_PARAMS
    dof_errchk(len(cov), params.dimops)

    ###we have data 6ab
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    if FIT:
        covinv = covinv_avg(cov, params.dimops)

    print("(Rough) scale of errors in data points = ", sqrt(cov[0][0]))

    if FIT:
        #comment out options{...}, bounds for L-BFGS-B
        ###start minimizer
        #result_min = namedtuple('min', ['x', 'fun', 'status', 'err_in_chisq'])
        if JACKKNIFE_FIT and JACKKNIFE == 'YES':
            result_min, param_err = jackknife_fit(params)
        else:
            result_min = mkmin(covinv, coords)
            ####compute errors 8ab, print results (not needed for plot part)
            param_err = geterr(result_min, covinv, coords)
        return result_min, param_err, coords, cov
    else:
        return coords, cov
