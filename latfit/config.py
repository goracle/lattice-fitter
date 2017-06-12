from math import fsum,log
import numpy as np
from numpy import arange,exp
from sympy import exp as exps
import sys

##the boundary for when the fitter warns you if the eigenvalues
##of your covariance are very small
EIGCUT = 10**(-20)

##Do a fit at all?
#FIT = False
FIT = True
##Uncorrelated fit? True or False
UNCORR = True
#UNCORR = False

##Plot Effective Mass? True or False
EFF_MASS = False
#EFF_MASS = True

#EFF_MASS_METHOD 1: analytic for arg to acosh (good for when additive const = 0)
#EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
#EFF_MASS_METHOD 3: one param fit (bad when additive constant = 0)
EFF_MASS_METHOD = 3

##starting values for fit parameters

START_PARAMS = [1.02356707e+11,   4.47338103e-01,   1.52757540e+09]
#START_PARAMS = [6.68203895e+05,   2.46978036e-01]
###-------BEGIN POSSIBLY OBSOLETE------###

#if set to true, AUTO_FIT uses curve_fit from scipy.optimize to bootstrap START_PARAMS.  Bounds must still be set manually.
#Bounds are used to find very rough start parameters: taken as the midpoints
#Probably, you should set FIT to False to first find some reasonable bounds.
#If ASSISTED_FIT is also True, use start_params to find the guess for the AUTO fitter
#(for use with L-BFGS-B)

AUTO_FIT=False
#AUTO_FIT=False
#ASSISTED_FIT=True
ASSISTED_FIT=False

##bounds for fit parameters
#optional, scale parameter to set binds
#scale=1e11
SCALE=1e11
BINDS = ((SCALE*.1,10*SCALE), (.4,.6),(.01*SCALE,.03*SCALE))
#BINDS = ((scale*.01,30*scale), (0, .8),(.01*scale*0,scale))

##boundary scale for zero'ing out a failing inverse Hessian
##(neg diagaonal entrie(s)).  Below 1/(CUTOFF*SCALE), an entry is set to 0
CUTOFF = 10**(7)

###-------END POSSIBLY OBSOLETE------###
##fineness of scale to plot (higher is more fine)
FINE = 1000.0
##method used by the scipy.optimize.minimize
##other internals will need to be edited if you change this
##it's probably not a good idea
METHOD = 'Nelder-Mead'
#METHOD = 'L-BFGS-B'

##jackknife correction? "YES" or "NO"
##correction only happens if multiple files are processed
JACKKNIFE = 'YES'

###DISPLAY PARAMETERS
#TITLE = 'FigureT_vec_pol_snk_1_sep4_momsrc000_momsnk000'
#no title takes the current working directory as the title
TITLE_PREFIX = ''
TITLE = ''
XLABEL = r'$t/a$'
if EFF_MASS:
    YLABEL = r'$am_{res}^{eff}}(t)$'
else:
    YLABEL = 'C(t)'

###setup is for cosh fit, but one can easily modify it.

#setup is for simple exponential fit, but one can easily modify it.
def fit_func_exp(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    #return trial_params[0]*(exp(-trial_params[1]*ctime)+exp(-trial_params[1]*(32-ctime)))
    return trial_params[0]*(exp(-trial_params[1]*ctime)+exp(-trial_params[1]*(24-ctime)))+trial_params[2]
    #other test function
    #return trial_params[0]+ctime*(trial_params[1]/(
    #        trial_params[2]+ctime)+fsum([trial_params[ci]/(
    #            trial_params[ci+1]+ctime) for ci in arange(
    #                3, len(trial_params), 2)]))

def fit_func(ctime,trial_params):
    #return trial_params[0]
    return np.array([fit_func_exp(ctime,trial_params)])

C=0*5.05447626030778e8 #additive constant added to effective mass functions
if EFF_MASS:
    if EFF_MASS_METHOD < 3:
        C=SCALE*.01563
        START_PARAMS = [.5]
        def fit_func(ctime,trial_params):
            return np.array([trial_params[0]])
    if EFF_MASS_METHOD == 2:
        C=0
        pass
    if EFF_MASS_METHOD == 3:
        FIT = True
        START_PARAMS = [.5]
        def fit_func(ctime,trial_params):
            #return trial_params[0]
            return np.array([fit_func_1p(ctime,trial_params)])

#for EFF_MASS_METHOD = 2
def fit_func_3pt_sym(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    #return trial_params[0]*(exp(-trial_params[1]*ctime)+exp(-trial_params[1]*(32-ctime)))
    return trial_params[0]*(exps(-trial_params[1]*ctime)+exps(-trial_params[1]*(24-ctime)))+trial_params[2]
    #other test function
    #return trial_params[0]+ctime*(trial_params[1]/(
    #        trial_params[2]+ctime)+fsum([trial_params[ci]/(
    #            trial_params[ci+1]+ctime) for ci in arange(
    #                3, len(trial_params), 2)]))

#one parameter fit, probably not good idea to edit (for effective mass plots)
def fit_func_1p(ctime,trial_params):
    C1 = exp(-trial_params[0]*ctime)+exp(-trial_params[0]*(24-ctime))
    C2 = exp(-trial_params[0]*(ctime+1))+exp(-trial_params[0]*(24-(ctime+1)))
    C3 = exp(-trial_params[0]*(ctime+2))+exp(-trial_params[0]*(24-(ctime+2)))
    arg = (C2-C1)/(C3-C2)
    if arg <= 0:
        print "imaginary effective mass."
        print "problematic time slices:",ctime,ctime+1,ctime+2
        print "C1=",C1
        print "C2=",C2
        print "C3=",C3
        sys.exit(1)
    return log((C2-C1)/(C3-C2))
