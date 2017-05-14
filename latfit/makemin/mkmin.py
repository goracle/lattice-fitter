from __future__ import division
from scipy.optimize import minimize
from scipy.optimize import curve_fit

from latfit.config import METHOD
from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS
from latfit.config import BINDS
from latfit.config import AUTO_FIT
from latfit.config import ASSISTED_FIT
from latfit.config import fit_func

def mkmin(covinv, coords):
    """Minimization of chi^2 section of fitter.
    Return minimized result.
    """
    x=[coords[i][0] for i in range(len(coords))]
    y=[coords[i][1] for i in range(len(coords))]
    if AUTO_FIT:
        if ASSISTED_FIT:
            guess = START_PARAMS
        else:
            guess=((i[0]+i[1])/2 for i in BINDS)
        l = len(BINDS)
        def f(t,*tp):
            return fit_func(t,tp)
        try:
            popt,pcov=curve_fit(x,y,f,p0=guess)
            start_params=popt
        except:
            print "Automatic guess of starting params failed."
            print "Attempting to continue with manual entry."
            start_params=START_PARAMS
    else:
            start_params=START_PARAMS
    if not METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (covinv, coords),
                              method=METHOD,options={'disp': True})
                          #method='BFGS')
                          #method='L-BFGS-B',
                          #bounds=BINDS,
                          #options={'disp': True})
    if METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (covinv, coords),
                              method=METHOD, bounds=BINDS,
                              options={'disp': True})
        print "number of iterations = ", RESULT_MIN.nit
    #print "minimized params = ", RESULT_MIN.x
    print "successfully minimized = ", RESULT_MIN.success
    print "status of optimizer = ", RESULT_MIN.status
    print "message of optimizer = ", RESULT_MIN.message
    #print "chi^2 minimized = ", RESULT_MIN.fun
    #print "chi^2 minimized check = ",chi_sq(RESULT_MIN.x,covinv,coords)
    #print covinv
    if RESULT_MIN.fun < 0:
        print "***ERROR***"
        print "Chi^2 minimizer failed. Chi^2 found to be less than zero."
    #print "degrees of freedom = ", dimcov-len(start_params)
    #print "chi^2 reduced = ", RESULT_MIN.fun/(dimcov-len(start_params))
    return RESULT_MIN
