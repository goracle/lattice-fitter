from scipy.optimize import minimize

from latfit.globs import METHOD
from latfit.mathfun.chi_sq import chi_sq

def mkmin(start_params, covinv, coords, switch, binds, dimcov):
    """Minimization of chi^2 section of fitter.
    Return minimized result.
    """
    if not METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (covinv, coords, switch),
                              method=METHOD)
                          #method='BFGS')
                          #method='L-BFGS-B',
                          #bounds=binds,
                          #options={'disp': True})
    if METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (covinv, coords, switch),
                              method=METHOD, bounds=binds,
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
    print "chi^2 reduced = ", RESULT_MIN.fun/(dimcov-len(start_params))
    print "degrees of freedom = ", dimcov-len(start_params)
    return RESULT_MIN
