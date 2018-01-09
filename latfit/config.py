"""Config for lattice fitter."""
import sys
from copy import copy
from math import log
import numpy as np
from numpy import exp
from sympy import exp as exps

from latfit.analysis.test_arg import test_arg
SENT = object()

###TYPE OF FIT

##plot anything at all?

NO_PLOT = True
NO_PLOT = False

##Do a fit at all?

FIT = False
FIT = True

##Jackknife fit?

JACKKNIFE_FIT = 'FROZEN'
JACKKNIFE_FIT = 'SINGLE'
JACKKNIFE_FIT = 'DOUBLE'

##Uncorrelated fit? True or False

UNCORR = True
UNCORR = False

##Plot Effective Mass? True or False

EFF_MASS = False
EFF_MASS = True

#solve the generalized eigenvalue problem (GEVP)

#GEVP = True
GEVP = False

#print correlation function, and sqrt(diag(cov)) and exit

PRINT_CORR = False
#PRINT_CORR = True

###METHODS/PARAMS

#time extent (1/2 is time slice where the mirroring occurs in periodic bc's)

TSEP = 3
LT = 64-2*TSEP

#rhs time separation (t0) of GEVP matrix
#(used for non eff mass fits)

TRHS = 3

#additive constant

ADD_CONST = False
ADD_CONST = True

#EFF_MASS_METHOD 1: analytic for arg to acosh
#(good for when additive const = 0)
#EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
#(bad for all cases; DO NOT USE.  It doesn't converge very often.)
#EFF_MASS_METHOD 3: one param fit

EFF_MASS_METHOD = 3

#Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

LOG=False
LOG=True

#do inverse via a correlation matrix (for higher numerical stability)
CORRMATRIX=True

##eliminate problematic configs.
#Simply set this to a list of ints indexing the configs,
#e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

#ELIM_JKCONF_LIST = range(48)
ELIM_JKCONF_LIST = []

##dynamic binning of configs.  BINNUM is number of configs per bin.
BINNUM = 1

#rescale the fit function by factor RESCALE
RESCALE = 1e5
RESCALE = 1.0

#####2x2 I=0
#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#'sep4/pipisigma_momsrc000_momsnk000'],
#['sep4/sigmapipi_momsrc000_momsnk000', 'sigmasigma_mom000']]

#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#'S_pipipipi_A_1PLUS'], ['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS']]

GEVP_DIRS = [['I0/S_pipiS_pipi_A_1PLUS.jkdat',
'I0/S_pipisigma_A_1PLUS.jkdat'],
['I0/sigmaS_pipi_A_1PLUS.jkdat', 'I0/sigmasigma_A_1PLUS.jkdat']]

#####3x3, I0
#GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
#              'sep4/pipisigma_momsrc000_momsnk000',
#              'S_pipipipi_A_1PLUS'],
#             ['sep4/sigmapipi_momsrc000_momsnk000',
#              'sigmasigma_mom000', 'sigmaS_pipi_A_1PLUS'],
#             ['pipiS_pipi_A_1PLUS', 'pipisigma_A_1PLUS', 'pipi_A_1PLUS']]

#####3x3, I2, pipi, 000, 100, 110
#GEVP_DIRS = [['S_pipiS_pipi_A_1PLUS', 'S_pipipipi_A_1PLUS',
#'S_pipiUUpipi_A_1PLUS'],
#['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS', 'pipiUUpipi_A_1PLUS'],
#['UUpipiS_pipi_A_1PLUS', 'UUpipipipi_A_1PLUS', 'UUpipiUUpipi_A_1PLUS']]


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
    TITLE_PREFIX = '24c '

#title

TITLE = ''

#axes labels

XLABEL = r'$t/a$'

if EFF_MASS:
    if EFF_MASS_METHOD == 3:
        if LOG:
            YLABEL = r'log(one param global fit)'
        else:
            YLABEL = r'one param global fit'
    else:
        YLABEL = r'$am_{eff}(t)$'
else:
    YLABEL = 'C(t)'

#precision to display, number of decimal places

PREC_DISP = 4

#stringent tolerance for minimizer?  true=stringent
MINTOL = True
MINTOL = False

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
        START_PARAMS = [-1.74580294, 2.8e-01,  3.0120e-02]*MULT
        #START_PARAMS = [1.54580294e+12, 3.61658103e-01, -8.7120e+08]*MULT
        #START_PARAMS = [.154580294, 3.61658103e-01, -8.7120e-5]*MULT
    else:
        START_PARAMS = [-1.18203895e+01, 4.46978036e-01]*MULT


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
        if ORIGL != 2 and FIT:
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
        if ADD_CONST and FIT:
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
        elif FIT:
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

#File format.  are the jackkknife blocks in ascii or hdf5?
STYPE = 'ascii'
STYPE = 'hdf5'
HDF5_PREFIX = 'I2'


#optional, scale parameter to set binds
#scale = 1e11
SCALE = 1e13

##bounds for fit parameters
##for use with L-BFGS-B
BINDS = ((SCALE*.1, 10*SCALE), (.4, .6), (.01*SCALE, .03*SCALE))
BINDS_LSQ = ([-np.inf,-np.inf,-9e08], [np.inf,np.inf,-6e08])
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

if FIT:
    FIT_FUNC_COPY = copy(prefit_func)

def fit_func(ctime, trial_params):
    """Fit function."""
    return np.hstack(
        [RESCALE*FIT_FUNC_COPY(ctime, trial_params[i*len(START_PARAMS):(i+1)*len(
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
    if ADD_CONST:
        corr3 = exp(-trial_params[0]*(ctime+2))+exp(
            -trial_params[0]*(LT-(ctime+2)))
        if np.array_equal(corr3, corr2):
            print("imaginary effective mass.")
            print("problematic time slices:", ctime, ctime+1, ctime+2)
            print("trial_param =", trial_params[0])
            print("START_PARAMS =", START_PARAMS)
            print("corr1 = ", corr1)
            print("corr2 = ", corr2)
            print("corr3 = ", corr3)
            sys.exit(1)
        sol = (corr2-corr1)/(corr3-corr2)
    else:
        sol = (corr1)/(corr2)
    if LOG:
        if not test_arg(sol, SENT):
            print("problematic time slices:", ctime, ctime+1, ctime+2)
            print("corr1 = ", corr1)
            print("corr2 = ", corr2)
            print("corr3 = ", corr3)
            sys.exit(1)
        sol = log(sol)
    else:
        pass
    return sol
