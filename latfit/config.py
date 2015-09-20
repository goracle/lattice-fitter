##the boundary for when the fitter warns you if the eigenvalues
##of your covariance are very small
EIGCUT = 10**(-10)

##starting values for fit parameters
##START_PARAMS = [-.18, 0.09405524, 0, .1]
START_PARAMS = [-.18, 0, .1]

##bounds for fit parameters
BINDS = ((None, None), (None, None), (0.0779, None))

##method used by the scipy.optimize.minimize
##other internals will need to be edited if you change this
##it's probably not a good idea
METHOD = 'L-BFGS-B'

##jackknife correction? YES or NO
##correction only happens if multiple files are processed
JACKCKNIFE = YES

from math import fsum
from numpy import arange

def fit_func(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]+ctime*(trial_params[1]/(
            trial_params[2]+ctime)+fsum([trial_params[ci]/(
                trial_params[ci+1]+ctime) for ci in arange(
                    3, len(trial_params), 2)]))
