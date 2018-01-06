"""Minimizes chi^2"""
import sys
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from latfit.config import METHOD
from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS
from latfit.config import BINDS
from latfit.config import AUTO_FIT
from latfit.config import ASSISTED_FIT
from latfit.config import fit_func
from latfit.config import JACKKNIFE_FIT

def mkmin(covinv, coords):
    """Minimization of chi^2 section of fitter.
    Return minimized result.
    """
    if AUTO_FIT:
        if GEVP:
            print("untested, unsupported.")
            sys.exit(0)
        xcoords = [coords[i][0] for i in range(len(coords))]
        ycoords = [coords[i][1] for i in range(len(coords))]
        if ASSISTED_FIT:
            guess = START_PARAMS
        else:
            guess = ((i[0]+i[1])/2 for i in BINDS)
        #lparams = len(BINDS)
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
        start_params = START_PARAMS
    if not METHOD in set(['L-BFGS-B']):
        res_min = minimize(chi_sq, start_params, (covinv, coords),
                           method=METHOD, options={'disp': True, 'maxiter':10000, 'maxfev':10000,
                                                   'xatol':0.00000001, 'fatol':0.00000001})
        #method = 'BFGS'
        #method = 'L-BFGS-B'
        #bounds = BINDS
        #options = {'disp': True}
    if METHOD in set(['L-BFGS-B']):
        res_min = minimize(chi_sq, start_params, (covinv, coords),
                           method=METHOD, bounds=BINDS,
                           options={'disp': True})
    #print "minimized params = ", res_min.x
    if not JACKKNIFE_FIT:
        print("number of iterations = ", res_min.nit)
        print("successfully minimized = ", res_min.success)
        print("status of optimizer = ", res_min.status)
        print("message of optimizer = ", res_min.message)
    #print "chi^2 minimized = ", res_min.fun
    #print "chi^2 minimized check = ", chi_sq(res_min.x, covinv, coords)
    #print covinv
    if res_min.fun < 0:
        print("***ERROR***")
        print("Chi^2 minimizer failed. Chi^2 found to be less than zero.")
        sys.exit(1)
    #print "degrees of freedom = ", dimcov-len(start_params)
    #print "chi^2 reduced = ", res_min.fun/(dimcov-len(start_params))
    return res_min
