from __future__ import division
from scipy.optimize import minimize

from latfit.config import METHOD
from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS
from latfit.config import BINDS

def mkmin(covinv, coords, dimcov):
    """Minimization of chi^2 section of fitter.
    Return minimized result.
    """
    if not METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, START_PARAMS, (covinv, coords),
                              method=METHOD)
                          #method='BFGS')
                          #method='L-BFGS-B',
                          #bounds=BINDS,
                          #options={'disp': True})
    if METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, START_PARAMS, (covinv, coords),
                              method=METHOD, bounds=BINDS,
                              options={'disp': True})
        print "number of iterations = ", RESULT_MIN.nit
    print "minimized params = ", RESULT_MIN.x
    print "successfully minimized = ", RESULT_MIN.success
    print "status of optimizer = ", RESULT_MIN.status
    print "message of optimizer = ", RESULT_MIN.message
    print "chi^2 minimized = ", RESULT_MIN.fun
    if RESULT_MIN.fun < 0:
        print "***ERROR***"
        print "Chi^2 minimizer failed. Chi^2 found to be less than zero."
    print "chi^2 reduced = ", RESULT_MIN.fun/(dimcov-len(START_PARAMS))
    print "degrees of freedom = ", dimcov-len(START_PARAMS)
    return RESULT_MIN
