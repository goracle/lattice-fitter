"""Config for lattice fitter."""
import sys
from math import log
from copy import copy
import numpy as np
from numpy import exp
from sympy import exp as exps

###TYPE OF FIT
##plot anything at all?
NO_PLOT = False
#NO_PLOT = True
##Do a fit at all?
#FIT = False
FIT = True
##Jackknife fit?
#JACKKNIFE_FIT = ''
JACKKNIFE_FIT = 'DOUBLE'
#JACKKNIFE_FIT = 'FROZEN'
##Uncorrelated fit? True or False
#UNCORR = True
UNCORR = False
##Plot Effective Mass? True or False
EFF_MASS = False
#EFF_MASS = True
#solve the generalized eigenvalue problem (GEVP)
GEVP = True
#GEVP = False
if GEVP:
    EFF_MASS = True # full fits not currently supported
##GENERALIZED PENCIL OF FUNCTION (see arXiv:1010.0202, for use with GEVP)
#if non-zero, set to 1
#(only do one pencil, more than one is supported,
#but probably not a good idea - see ref above)
NUM_PENCILS = 0
#paper set this to 4
PENCIL_SHIFT = 1

###METHODS/PARAMS

#EFF_MASS_METHOD 1: analytic for arg to acosh
#(good for when additive const = 0)
#numeric solve system of three transcendental equations
#(bad for all cases; DO NOT USE.  It doesn't converge very often.)
#EFF_MASS_METHOD 2:
#EFF_MASS_METHOD 3: one param fit (bad when additive constant = 0)
EFF_MASS_METHOD = 3

#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#'sep4/pipisigma_momsrc000_momsnk000'],
#['sep4/sigmapipi_momsrc000_momsnk000', 'sigmasigma_mom000']]
GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
              'sep4/pipisigma_momsrc000_momsnk000', 'S_pipipipi_A_1plus'],
             ['sep4/sigmapipi_momsrc000_momsnk000',
              'sigmasigma_mom000', 'sigmaS_pipi_A_1plus'],
             ['pipiS_pipi_A_1plus', 'pipisigma_A_1plus', 'pipi_A_1plus']]
###DISPLAY PARAMETERS
#no title given takes the current working directory as the title
TITLE_PREFIX = r'$\pi\pi, \sigma$, momtotal000 '
#TITLE_PREFIX = r'3x3 GEVP, $\pi\pi, \sigma$, momtotal000 '
#TITLE_PREFIX = 'TEST '
TITLE = ''
XLABEL = r'$t/a$'
if EFF_MASS:
    YLABEL = r'$am_{res}^{eff}}(t)$'
else:
    YLABEL = 'C(t)'

###starting values for fit parameters
if GEVP:
    START_PARAMS = [.5]*len(GEVP_DIRS)
else:
    START_PARAMS = [7.02356707e+11, 4.47338103e-01, 1.52757540e+11]
    #START_PARAMS = [6.68203895e+05, 2.46978036e-01]

C = 0
if EFF_MASS:
    if EFF_MASS_METHOD < 3:
        #additive constant added to effective mass functions
        SCALE = 1e11
        C = 1.935*SCALE*0
        #C = SCALE*0.01563
        START_PARAMS = [.5] if not GEVP else START_PARAMS
    elif EFF_MASS_METHOD == 3:
        START_PARAMS = [.5] if not GEVP else START_PARAMS

###setup is for cosh fit, but one can easily modify it.

##library of functions to fit.  define them in the usual way
#setup is for simple exponential fit, but one can easily modify it.
def fit_func_exp(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]*(
        exp(-trial_params[1]*ctime)+exp(-trial_params[1]*(
            24-ctime)))+trial_params[2]
def fit_func_exp_long(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]*(exp(-trial_params[1]*ctime)+exp(-trial_params[1]*(32-ctime)))

##select which of the above functions to use
if GEVP:
    def prefit_func(ctime, trial_params):
        """Gevp fit function"""
        if ctime:
            pass
        return np.array(trial_params)
elif not EFF_MASS:
    def prefit_func(ctime, trial_params):
        """Non gevp fit function"""
        return np.array([fit_func_exp(ctime, trial_params)])
        #return np.array([fit_func_exp_long(ctime, trial_params)])

elif EFF_MASS and not GEVP:
    if EFF_MASS_METHOD < 3:
        def prefit_func(ctime, trial_params):
            """eff_mass_meth<3; fit_func"""
            if ctime:
                pass
            return np.array([trial_params[0]]) if not GEVP else np.array(trial_params)
    elif EFF_MASS_METHOD == 3:
        FIT = True
        def prefit_func(ctime, trial_params):
            """eff_mass_meth=3; fit_func"""
            return (np.array([fit_func_1p(ctime, trial_params)])
                    if not GEVP else np.array(trial_params))
    if EFF_MASS_METHOD == 2:
        pass

##RARELY EDIT BELOW
##bounds for fit parameters
#optional, scale parameter to set binds
#scale = 1e11
SCALE = 1e11
##for use with L-BFGS-B
BINDS = ((SCALE*.1, 10*SCALE), (.4, .6), (.01*SCALE, .03*SCALE))
#BINDS = ((scale*.01, 30*scale), (0, .8), (.01*scale*0, scale))

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
##eliminate problematic configs.
#Set this to a list of ints indexing the configs,
#e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs
ELIM_JKCONF_LIST = []
###-------BEGIN POSSIBLY OBSOLETE------###

##the boundary for when the fitter warns you if the eigenvalues
##of your covariance are very small
EIGCUT = 10**(-23)

#if set to true, AUTO_FIT uses curve_fit from scipy.optimize
#to bootstrap START_PARAMS.  Bounds must still be set manually.
#Bounds are used to find very rough start parameters: taken as the midpoints
#Probably, you should set FIT to False to first find some reasonable bounds.
#If ASSISTED_FIT is also True,
#use start_params to find the guess for the AUTO fitter
#(for use with L-BFGS-B)

AUTO_FIT = False
#AUTO_FIT = False
#ASSISTED_FIT = True
ASSISTED_FIT = False

##boundary scale for zero'ing out a failing inverse Hessian
##(neg diagaonal entrie(s)).  Below 1/(CUTOFF*SCALE), an entry is set to 0
CUTOFF = 10**(7)

###-------END POSSIBLY OBSOLETE------###

##DO NOT EDIT BELOW

#for general pencil of function
FIT_FUNC_COPY = copy(prefit_func)
def fit_func(ctime, trial_params):
    """Fit function."""
    return np.hstack(
        [FIT_FUNC_COPY(ctime, trial_params[i*len(START_PARAMS):(i+1)*len(
            START_PARAMS)]) for i in range(2**NUM_PENCILS)])
START_PARAMS = list(START_PARAMS)*2**NUM_PENCILS

#for EFF_MASS_METHOD = 2
def fit_func_3pt_sym(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    #return trial_params[0]*(exps(-trial_params[1]*ctime)+exps(-trial_params[1]*(32-ctime)))
    return trial_params[0]*(exps(
        -trial_params[1]*ctime)+exps(
            -trial_params[1]*(24-ctime)))+trial_params[2]

#for EFF_MASS_METHOD = 3
def fit_func_1p(ctime, trial_params):
    """Fit function for effective mass method 3 (one parameter fit)"""
    corr1 = exp(-trial_params[0]*ctime)+exp(-trial_params[0]*(24-ctime))
    corr2 = exp(-trial_params[0]*(ctime+1))+exp(-trial_params[0]*(24-(
        ctime+1)))
    corr3 = exp(-trial_params[0]*(ctime+2))+exp(-trial_params[0]*(
        24-(ctime+2)))
    arg = (corr2-corr1)/(corr3-corr2)
    if arg <= 0:
        print("imaginary effective mass.")
        print("problematic time slices:", ctime, ctime+1, ctime+2)
        print("corr1 = ", corr1)
        print("corr2 = ", corr2)
        print("corr3 = ", corr3)
        sys.exit(1)
    return log((corr2-corr1)/(corr3-corr2))
