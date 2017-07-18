"""Config for lattice fitter."""
import sys
from copy import copy
from math import log
import numpy as np
from numpy import exp
from sympy import exp as exps

from latfit.mathfun.proc_meff import test_arg

###TYPE OF FIT

##plot anything at all?

NO_PLOT = False
#NO_PLOT = True

##Do a fit at all?

#FIT = False
FIT = True

##Jackknife fit?

JACKKNIFE_FIT = 'DOUBLE'
#JACKKNIFE_FIT = ''
#JACKKNIFE_FIT = 'FROZEN'

##Uncorrelated fit? True or False

#UNCORR = True
UNCORR = False

##Plot Effective Mass? True or False

#EFF_MASS = False
EFF_MASS = True

#solve the generalized eigenvalue problem (GEVP)

GEVP = True
#GEVP = False

#print correlation function, and sqrt(diag(cov)) and exit

PRINT_CORR = False
#PRINT_CORR = True

###METHODS/PARAMS

#time extent

LT = 24

#rhs time separation (t0) of GEVP matrix
#(used for non eff mass fits)

TRHS = 3

#additive constant

ADD_CONST = True

#EFF_MASS_METHOD 1: analytic for arg to acosh
#(good for when additive const = 0)
#EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
#(bad for all cases; DO NOT USE.  It doesn't converge very often.)
#EFF_MASS_METHOD 3: one param fit

EFF_MASS_METHOD = 3

#Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

#LOG=True
LOG=False

#####2x2 I=0
GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
'sep4/pipisigma_momsrc000_momsnk000'],
['sep4/sigmapipi_momsrc000_momsnk000', 'sigmasigma_mom000']]

#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#'S_pipipipi_A_1plus'], ['pipiS_pipi_A_1plus', 'pipi_A_1plus']]

#####3x3, I0
#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#              'sep4/pipisigma_momsrc000_momsnk000',
#              'S_pipipipi_A_1plus'],
#             ['sep4/sigmapipi_momsrc000_momsnk000',
#              'sigmasigma_mom000', 'sigmaS_pipi_A_1plus'],
#             ['pipiS_pipi_A_1plus', 'pipisigma_A_1plus', 'pipi_A_1plus']]

#####3x3, I2, pipi, 000, 100, 110
#GEVP_DIRS = [['S_pipiS_pipi_A_1plus', 'S_pipipipi_A_1plus',
#'S_pipiUUpipi_A_1plus'],
#['pipiS_pipi_A_1plus', 'pipi_A_1plus', 'pipiUUpipi_A_1plus'],
#['UUpipiS_pipi_A_1plus', 'UUpipipipi_A_1plus', 'UUpipiUUpipi_A_1plus']]


###DISPLAY PARAMETERS
#no title given takes the current working directory as the title

#title prefix
if GEVP:
    if len(GEVP_DIRS) == 2:
        TITLE_PREFIX = r'$\pi\pi, \sigma$, momtotal000 '
    elif len(GEVP_DIRS) == 3:
        #TITLE_PREFIX = r'3x3 GEVP, $\pi\pi, \sigma$, momtotal000 '
        TITLE_PREFIX = r'3x3 GEVP, $\pi\pi$, momtotal000 '
else:
    TITLE_PREFIX = 'TEST2 '

#title

TITLE = ''

#axes labels

XLABEL = r'$t/a$'

if EFF_MASS:
    if EFF_MASS_METHOD == 3:
        if LOG:
            YLABEL = r'one param fit'
        else:
            YLABEL = r'log(one param fit)'
    else:
        YLABEL = r'$am_{eff}(t)$'
else:
    YLABEL = 'C(t)'

###setup is for cosh fit, but one can easily modify it.

###starting values for fit parameters
if GEVP:
    MULT = len(GEVP_DIRS)
else:
    MULT = 1

C = 0
if EFF_MASS:
    C = 0
    START_PARAMS = [.5]*MULT
    if EFF_MASS_METHOD < 3:
        #additive constant added to effective mass functions
        SCALE = 1e11
        C = 1.935*SCALE*0
        #C = SCALE*0.01563
else:
    if ADD_CONST:
        START_PARAMS = [6.02356707e+11, 4.47338103e-01, 1.5270e+11]*MULT
    else:
        START_PARAMS = [1.68203895e+02, 6.46978036e-01]*MULT


##library of functions to fit.  define them in the usual way
#setup is for simple exponential fit, but one can easily modify it.
def fit_func_exp(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]*(exp(
        -trial_params[1]*ctime)+exp(-trial_params[1]*(LT-ctime)))

def fit_func_exp_add(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    return trial_params[0]*(exp(
        -trial_params[1]*ctime)+exp(
            -trial_params[1]*(LT-ctime)))+trial_params[2]

def fit_func_exp_gevp(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv)) GEVP, cosh+const
    """
    return ((exp(-trial_params[0]*ctime)+exp(
        -trial_params[1]*(LT-ctime)))+trial_params[1])/(
            (exp(-trial_params[0]*(TRHS))+exp(
                -trial_params[1]*(LT-(TRHS))))+trial_params[1])


##select which of the above functions to use

ORIGL = int(len(START_PARAMS)/MULT)
if EFF_MASS:

    ###check len of start params
    if ORIGL != 1:
        print("***ERROR***")
        print("dimension of GEVP matrix and start params do not match")
        print("(or for non-gevp fits, the start_param len>1)")
        sys.exit(1)

    ###select fit function
    if EFF_MASS_METHOD == 1 or EFF_MASS_METHOD == 2:
        def prefit_func(ctime, trial_params):
            """eff mass method 1, fit func, single const fit
            """
            if ctime:
                pass
            return np.array(trial_params)

    elif EFF_MASS_METHOD == 3:
        FIT = True #meaningless (without log) otherwise
        def prefit_func(ctime, trial_params):
            """eff mass 3, fit func
            """
            return np.array([fit_func_1p(
                ctime, trial_params[j:j+1]) for j in range(MULT)])
    else:
        print("***ERROR***")
        print("check config file fit func selection.")
        sys.exit(1)

else:
    if GEVP:
        ###check len of start params
        if ORIGL != 2:
            print("***ERROR***")
            print("flag 1 length of start_params invalid")
            sys.exit(1)
        ###select fit function
        def prefit_func(ctime, trial_params):
            """gevp fit func, non eff mass"""
            return np.array([
                fit_func_exp_gevp(ctime, trial_params[j*ORIGL:(j+1)*ORIGL])
                for j in range(MULT)])
    else:
        if ADD_CONST:
            ###check len of start params
            if ORIGL != 3:
                print("***ERROR***")
                print("flag 2 length of start_params invalid")
                sys.exit(1)
            ###select fit function
            def prefit_func(ctime, trial_params):
                """fit func non gevp, additive const
                """
                return np.array([fit_func_exp_add(ctime, trial_params)])
        else:
            ###check len of start params
            if ORIGL != 2:
                print("***ERROR***")
                print("flag 3 length of start_params invalid")
                sys.exit(1)
            ###select fit function
            def prefit_func(ctime, trial_params):
                """fit func non gevp, no additive const
                """
                return np.array([fit_func_exp(ctime, trial_params)])

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
#Simply set this to a list of ints indexing the configs,
#e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

ELIM_JKCONF_LIST = range(14)
#ELIM_JKCONF_LIST = []

###-------BEGIN POSSIBLY OBSOLETE------###

#multiply both sides of the gevp matrix by norms

#NORMS = [[1.0/(16**6), 1.0/(16**3)], [1.0/(16**3), 1]]
NORMS = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

##GENERALIZED PENCIL OF FUNCTION (see arXiv:1010.0202, for use with GEVP)
#if non-zero, set to 1 (only do one pencil,
#more than one is supported, but probably not a good idea - see ref above)

NUM_PENCILS = 0
PENCIL_SHIFT = 1 #paper set shift to 4

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

def fit_func_3pt_sym(ctime, trial_params):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    for EFF_MASS_METHOD = 2
    """
    return trial_params[0]*(exps(
        -trial_params[1]*ctime)+exps(
            -trial_params[1]*(LT-ctime)))+trial_params[2]

def fit_func_1p(ctime, trial_params):
    """one parameter eff. mass fit function
    for EFF_MASS_METHOD = 3
    """
    corr1 = exp(-trial_params[0]*ctime)+exp(
        -trial_params[0]*(LT-ctime))
    corr2 = exp(-trial_params[0]*(ctime+1))+exp(
        -trial_params[0]*(LT-(ctime+1)))
    corr3 = exp(-trial_params[0]*(ctime+2))+exp(
        -trial_params[0]*(LT-(ctime+2)))
    if corr3 == corr2:
        print("imaginary effective mass.")
        print("problematic time slices:", ctime, ctime+1, ctime+2)
        print("corr1 = ", corr1)
        print("corr2 = ", corr2)
        print("corr3 = ", corr3)
        sys.exit(1)
    sol = (corr2-corr1)/(corr3-corr2)
    if LOG:
        if not test_arg(sol, config.sent):
            print("problematic time slices:", ctime, ctime+1, ctime+2)
            print("corr1 = ", corr1)
            print("corr2 = ", corr2)
            print("corr3 = ", corr3)
            sys.exit(1)
    else:
        pass
    return sol
config.sent = object()
