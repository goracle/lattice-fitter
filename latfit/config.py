##the boundary for when the fitter warns you if the eigenvalues
##of your covariance are very small
EIGCUT = 10**(-20)

##starting values for fit parameters
##START_PARAMS = [-.18, 0.09405524, 0, .1]
##START_PARAMS = [-.18, 0, .1]
START_PARAMS = [4e7,.257,.083e6]

##bounds for fit parameters
BINDS = ((7e6,1e8), (0, 1),(7e5,3e6))

##Uncorrelated fit? True or False
UNCORR = True
#UNCORR = False

##scale to plot (divisor)
FINE = 1.0
##method used by the scipy.optimize.minimize
##other internals will need to be edited if you change this
##it's probably not a good idea
METHOD = 'L-BFGS-B'

##jackknife correction? "YES" or "NO"
##correction only happens if multiple files are processed
JACKKNIFE = 'YES'

###PLOT PARAMETERS
TITLE = 'Pion Corr'
XLABEL = 'Time'
YLABEL = 'Func'

from math import fsum
from numpy import arange, exp

#setup is for simple exponential fit, but one can easily modify it.
def fit_func(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]*exp(-trial_params[1]*ctime)+trial_params[2]
    #other test function
    #return trial_params[0]+ctime*(trial_params[1]/(
    #        trial_params[2]+ctime)+fsum([trial_params[ci]/(
    #            trial_params[ci+1]+ctime) for ci in arange(
    #                3, len(trial_params), 2)]))
