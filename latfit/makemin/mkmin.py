"""Minimizes chi^2"""
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from iminuit import Minuit
from iminuit import minimize as minit
import numpy as np

from latfit.config import METHOD
from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS
from latfit.config import BINDS, NOLOOP
from latfit.config import AUTO_FIT
from latfit.config import ASSISTED_FIT
from latfit.config import fit_func
from latfit.config import JACKKNIFE_FIT
from collections import namedtuple
# from latfit.config import MINTOL
from latfit.config import GEVP, SYSTEMATIC_EST
import latfit.config

def mkmin(covinv, coords, method=METHOD):
    """Minimization of chi^2 section of fitter.
    Return minimized result.
    """
    if AUTO_FIT:
        assert None, "unsupported"
        if GEVP:
            print("untested, unsupported.")
            sys.exit(0)
        xcoords = [coords[i][0] for i in range(len(coords))]
        ycoords = [coords[i][1] for i in range(len(coords))]
        if ASSISTED_FIT:
            guess = START_PARAMS
        else:
            guess = ((i[0]+i[1])/2 for i in BINDS)
        # lparams = len(BINDS)

        def func(ctime, *tp):
            """puts fit function in right form for curve_fit"""
            return fit_func(ctime, tp)
        try:
            popt, _ = curve_fit(xcoords, ycoords, func, p0=guess)
            start_params = popt
        except(ValueError, RuntimeError):
            print("Automatic guess of starting params failed.")
            print("Attempting to continue with manual entry.")
            start_params = START_PARAMS
    else:
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
    if method in set(['L-BFGS-B']):
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
    if 'minuit' in method:
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
        if False:
            def func(trial_params):
                """minimize this."""
                return chi_sq(trial_params, covinv, coords)
            fparams = [str(i) for i in range(len(START_PARAMS))]
            printl = 1 if NOLOOP else 0
            minimizer = Minuit(func,
                            use_array_call=True,
                            print_level=printl,
                            pedantic=True if NOLOOP else False,
                            forced_parameters=fparams)
            res_min = {}
            res_min['x'] = np.nan*np.asarray(START_PARAMS)
            res_min['status'] = 0
            res_min['fun'] = np.inf
            try:
                minimizer.migrad()
                res_min['status'] = 0 if minimizer.get_fmin().is_valid else 1
                vals = np.asarray(list(dict(minimizer.values).values()))
                res_min['x'] = vals
                res_min['fun'] = func(vals)
            except RuntimeError:
                res_min['status'] = 1
            res_min = convert_to_namedtuple(res_min)
        
    if not JACKKNIFE_FIT:
        print("number of iterations = ", res_min.nit)
        print("successfully minimized = ", res_min.success)
        print("status of optimizer = ", res_min.status)
        print("message of optimizer = ", res_min.message)
    # print "chi^2 minimized = ", res_min.fun
    # print "chi^2 minimized check = ", chi_sq(res_min.x, covinv, coords)
    # print covinv
    if not res_min.status:
        if res_min.fun < 0:
            raise NegChisq
    # print "degrees of freedom = ", dimcov-len(start_params)
    # print "chi^2 reduced = ", res_min.fun/(dimcov-len(start_params))
    return prune_res_min(res_min)

def convert_to_namedtuple(dictionary):
    """Convert dictionary to named tuple"""
    return namedtuple('min', dictionary.keys())(**dictionary)



if SYSTEMATIC_EST:
    def prune_res_min(res_min):
        """Get rid of systematic error information"""
        print([res_min.x[len(START_PARAMS):][2*i+1] for i in range(len(START_PARAMS))])
        res_min.x = np.array(res_min.x)[:len(START_PARAMS)]
        return res_min
else:
    def prune_res_min(res_min):
        """pass"""
        return res_min

class NegChisq(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, problemx=None, message=''):
        print("***ERROR***")
        print("Chi^2 minimizer failed. Chi^2 found to be less than zero.")
        super(NegChisq, self).__init__(message)
        self.problemx = problemx
        self.message = message
