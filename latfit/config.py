##the boundary for when the fitter warns you if the eigenvalues
##of your covariance are very small
EIGCUT = 10**(-20)
##boundary scale for zero'ing out a failing inverse Hessian
##(neg diagaonal entrie(s)).  Below 1/(CUTOFF*SCALE), an entry is set to 0
CUTOFF = 10**(7)

##Do a fit at all?
#FIT = False
FIT = True
##starting values for fit parameters
##START_PARAMS = [-.18, 0.09405524, 0, .1]
##START_PARAMS = [-.18, 0, .1]
START_PARAMS = [6.67889076e+10,   5e-01,   1.64009479e+08]
#if set to true, AUTO_FIT uses curve_fit from scipy.optimize to bootstrap START_PARAMS.  Bounds must still be set manually.
#Bounds are used to find very rough start parameters: taken as the midpoints
#Probably, you should set FIT to False to first find some reasonable bounds.
#If ASSISTED_FIT is also True, use start_params to find the guess for the AUTO fitter
AUTO_FIT=True
ASSISTED_FIT=False

##bounds for fit parameters
#optional, scale parameter to set binds
#scale=1e11
SCALE=1e10
BINDS = ((SCALE*.1,10*SCALE), (0,1),(.01*SCALE,.02*SCALE))
#BINDS = ((scale*.01,30*scale), (0, .8),(.01*scale*0,scale))

##Uncorrelated fit? True or False
#UNCORR = True
UNCORR = False

##Plot Effective Mass? True or False
EFF_MASS = False
#EFF_MASS = True

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
#TITLE = 'FigureT_vec_pol_snk_1_sep4_momsrc000_momsnk000'
#no title takes the current working directory as the title
TITLE = ''
XLABEL = 'Time (lattice units)'
if EFF_MASS:
    YLABEL = 'Effective Mass (lattice units)'
else:
    YLABEL = 'Corr Func'

from math import fsum
from numpy import arange, exp

#setup is for simple exponential fit, but one can easily modify it.
def fit_func_exp(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    #return trial_params[0]*exp(-trial_params[1]*ctime)
    return trial_params[0]*exp(-trial_params[1]*ctime)+trial_params[2]
    #other test function
    #return trial_params[0]+ctime*(trial_params[1]/(
    #        trial_params[2]+ctime)+fsum([trial_params[ci]/(
    #            trial_params[ci+1]+ctime) for ci in arange(
    #                3, len(trial_params), 2)]))


def fit_func(ctime,trial_params):
    #return trial_params[0]
    return fit_func_exp(ctime,trial_params)
#effective mass fits not supported
if EFF_MASS:
    FIT = False
