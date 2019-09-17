"""Minimizes chi^2 (t^2)"""
import sys
from collections import namedtuple
from scipy.optimize import minimize
# from scipy.optimize import curve_fit
# from iminuit import Minuit
from iminuit import minimize as minit
import numpy as np

from latfit.config import METHOD
from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS
from latfit.config import BINDS
from latfit.config import JACKKNIFE_FIT
# from latfit.config import MINTOL
from latfit.config import SYSTEMATIC_EST
import latfit.config

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def mkmin(covinv, coords, method=METHOD):
    """Minimization of chi^2 (t^2) section of fitter.
    Return minimized result.
    """
    start_params = [*START_PARAMS, *START_PARAMS,
                    *START_PARAMS] if SYSTEMATIC_EST else START_PARAMS
    if method not in set(['L-BFGS-B', 'minuit']):
        if latfit.config.MINTOL:
            options = {'maxiter': 10000, 'maxfev': 10000,
                       'xatol': 0.00000001, 'fatol': 0.00000001}
        else:
            options = {}
        res_min = minimize(chi_sq, start_params, (covinv, coords),
                           method=method,
                           options=options)
        #else:
        #    res_min = minimize(chi_sq, start_params, (covinv, coords),
        #                       method=METHOD)
        # options={'disp': True})
        #'maxiter': 10000,
        #'maxfev': 10000})
        # method = 'BFGS'
        # method = 'L-BFGS-B'
        # bounds = BINDS
        # options = {'disp': True}
    elif method in set(['L-BFGS-B']):
        if latfit.config.MINTOL:
            options = {}
        else:
            options = {}
        try:
            res_min = minimize(chi_sq, start_params, (covinv, coords),
                               method=method, bounds=BINDS,
                               options=options)
        except FloatingPointError:
            print('floating point error')
            print('covinv:')
            print(covinv)
            print('coords')
            print(coords)
            sys.exit(1)

    # print "minimized params = ", res_min.x
    elif 'minuit' in method:
        options = {}
        try:
            res_min = minit(chi_sq, start_params, (covinv, coords),
                            method=method, bounds=BINDS,
                            options=options)
            status = res_min.minuit.get_fmin().is_valid
            status = 0 if res_min.success else 1
            res_min.status = status
        except RuntimeError:
            status = 1
            res_min = {}
            res_min['status'] = 1
        res_min = convert_to_namedtuple(res_min)
        # insert string here
    if not JACKKNIFE_FIT:
        print("number of iterations = ", res_min.nit)
        print("successfully minimized = ", res_min.success)
        print("status of optimizer = ", res_min.status)
        print("message of optimizer = ", res_min.message)
    # print "chi^2 minimized = ", res_min.fun
    # print "chi^2 minimized check = ", chi_sq(res_min.x, covinv, coords)
    # print covinv
    if res_min.status and latfit.config.BOOTSTRAP:
        print("boostrap debug")
        print("covinv =", covinv)
        print("coords =", coords)
        print("start_params =", start_params)
        print("chisq =", chi_sq(start_params, covinv, coords))
    if not res_min.status:
        if res_min.fun < 0:
            print("negative chi^2 found:", res_min.fun)
            print("result =", res_min.x)
            print("chi^2 (t^2; check) =",
                  chi_sq(res_min.x, covinv, coords))
            print("covinv:", covinv)
            sys.exit(1)
            raise NegChisq
    # print "degrees of freedom = ", dimcov-len(start_params)
    # print "chi^2 reduced = ", res_min.fun/(dimcov-len(start_params))
    return prune_res_min(res_min)

@PROFILE
def convert_to_namedtuple(dictionary):
    """Convert dictionary to named tuple"""
    return namedtuple('min', dictionary.keys())(**dictionary)



if SYSTEMATIC_EST:
    @PROFILE
    def prune_res_min(res_min):
        """Get rid of systematic error information"""
        print([res_min.x[len(START_PARAMS):][2*i+1] for i in range(len(START_PARAMS))])
        res_min.x = np.array(res_min.x)[:len(START_PARAMS)]
        return res_min
else:
    @PROFILE
    def prune_res_min(res_min):
        """pass"""
        return res_min

class NegChisq(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    @PROFILE
    def __init__(self, problemx=None, message=''):
        print("***ERROR***")
        print("Chi^2 (t^2) minimizer failed.",
              "Chi^2 (t^2) found to be less than zero.")
        super(NegChisq, self).__init__(message)
        self.problemx = problemx
        self.message = message
