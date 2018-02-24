"""Config for lattice fitter."""
import sys
from copy import copy
from math import log
from numbers import Number
import numpy as np
from numpy import exp
from sympy import exp as exps
from latfit.analysis.test_arg import zero_p, testsol

# TYPE OF FIT

# plot anything at all?

NO_PLOT = True
NO_PLOT = False

# Do a fit at all?

FIT = False
FIT = True

# Jackknife fit?

JACKKNIFE_FIT = 'FROZEN'
JACKKNIFE_FIT = 'SINGLE'
JACKKNIFE_FIT = 'DOUBLE'

# Uncorrelated fit? True or False

UNCORR = True
UNCORR = False

# Plot Effective Mass? True or False

EFF_MASS = True
EFF_MASS = False

# EFF_MASS_METHOD 1: analytic for arg to acosh
# (good for when additive const = 0)
# EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
# (bad for all cases; DO NOT USE.  It doesn't converge very often.)
# EFF_MASS_METHOD 3: one param fit

EFF_MASS_METHOD = 3

# solve the generalized eigenvalue problem (GEVP)

GEVP = True
GEVP = False

# METHODS/PARAMS

# time extent (1/2 is time slice where the mirroring occurs in periodic bc's)

TSEP = 3
LT = 64-2*TSEP

# additive constant

ADD_CONST = False
ADD_CONST = True

#isospin value (convenience switch)
ISOSPIN = 2

# calculate the I=0 phase shift?

L_BOX = 24
PION_MASS = 0.13908
CALC_PHASE_SHIFT = False
CALC_PHASE_SHIFT = True

# Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

LOG = False
LOG = True

# stringent tolerance for minimizer?  true = stringent
MINTOL = False
MINTOL = True

# rescale the fit function by factor RESCALE
RESCALE = 1.0
RESCALE = 1e12

# starting values for fit parameters
if EFF_MASS and EFF_MASS_METHOD != 2:
    START_PARAMS = [.5]
else:
    if ADD_CONST:
        START_PARAMS = [1.4580294, 5.0e-01, 3.0120e-02]
    else:
        START_PARAMS = [-1.18203895e+01, 4.46978036e-01]

# 2x2 I = 0
# GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
# 'sep4/pipisigma_momsrc000_momsnk000'],
# ['sep4/sigmapipi_momsrc000_momsnk000', 'sigmasigma_mom000']]

# GEVP_DIRS = [['sep4/pipi_mom1src000_mom2src000_mom1snk000',
# 'S_pipipipi_A_1PLUS'], ['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS']]

if ISOSPIN == 0:
    # 3x3, I0
    GEVP_DIRS = [
        ['I0/S_pipiS_pipi_A_1PLUS.jkdat',
        'I0/S_pipisigma_A_1PLUS.jkdat',
        'I0/S_pipipipi_A_1PLUS.jkdat'],
        ['I0/sigmaS_pipi_A_1PLUS.jkdat',
        'I0/sigmasigma_A_1PLUS.jkdat',
        'I0/sigmapipi_A_1PLUS.jkdat'],
        ['I0/pipiS_pipi_A_1PLUS.jkdat',
        'I0/pipisigma_A_1PLUS.jkdat',
        'I0/pipi_A_1PLUS.jkdat']
    ]

    # sigma
    GEVP_DIRS = [
        ['I0/S_pipiS_pipi_A_1PLUS.jkdat', 'I0/S_pipisigma_A_1PLUS.jkdat'],
        ['I0/sigmaS_pipi_A_1PLUS.jkdat', 'I0/sigmasigma_A_1PLUS.jkdat']
    ]

    # no sigma
    GEVP_DIRS = [
        ['I0/S_pipiS_pipi_A_1PLUS.jkdat', 'I0/S_pipipipi_A_1PLUS.jkdat'],
        ['I0/pipiS_pipi_A_1PLUS.jkdat', 'I0/pipi_A_1PLUS.jkdat']
    ]
elif ISOSPIN == 2:
    # pipi with one unit of momentum
    GEVP_DIRS = [
        ['I2/S_pipiS_pipi_A_1PLUS.jkdat', 'I2/S_pipipipi_A_1PLUS.jkdat'],
        ['I2/pipiS_pipi_A_1PLUS.jkdat', 'I2/pipi_A_1PLUS.jkdat']
    ]



# 3x3, I2, pipi, 000, 100, 110
# GEVP_DIRS = [['S_pipiS_pipi_A_1PLUS', 'S_pipipipi_A_1PLUS',
# 'S_pipiUUpipi_A_1PLUS'],
# ['pipiS_pipi_A_1PLUS', 'pipi_A_1PLUS', 'pipiUUpipi_A_1PLUS'],
# ['UUpipiS_pipi_A_1PLUS', 'UUpipipipi_A_1PLUS', 'UUpipiUUpipi_A_1PLUS']]

# modify the configs used and bin

# eliminate problematic configs.
# Simply set this to a list of ints indexing the configs,
# e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

ELIM_JKCONF_LIST = [18, 24, 11, 21, 28, 32, 12,
                    45, 26, 28, 33, 35, 40, 41, 43, 50]
ELIM_JKCONF_LIST = []

# dynamic binning of configs.  BINNUM is number of configs per bin.
BINNUM = 1



# DISPLAY PARAMETERS
# no title given takes the current working directory as the title

# title prefix
if GEVP:
    if len(GEVP_DIRS) == 2:
        if ISOSPIN == 0:
            TITLE_PREFIX = r'$\pi\pi, \sigma$, momtotal000 '
        elif ISOSPIN == 2:
            TITLE_PREFIX = r'$\pi\pi, I2, momtotal000 '
    elif len(GEVP_DIRS) == 3:
        # TITLE_PREFIX = r'3x3 GEVP, $\pi\pi, \sigma$, momtotal000 '
        TITLE_PREFIX = r'3x3 GEVP, $\pi\pi$, momtotal000 '
else:
    TITLE_PREFIX = '24c '

# title

TITLE = ''

# axes labels

XLABEL = r'$t/a$'

if EFF_MASS:
    if EFF_MASS_METHOD == 3:
        if LOG and ADD_CONST:
            YLABEL = r'log(one param global fit)'
        elif LOG and not ADD_CONST:
            YLABEL = r'Effective Mass (lattice units)'
        else:
            YLABEL = r'one param global fit'
    else:
        YLABEL = r'$am_{eff}(t)$'
else:
    YLABEL = 'C(t)'

# box plot (for effective mass tolerance display)?
BOX_PLOT = False
BOX_PLOT = True

# precision to display, number of decimal places

PREC_DISP = 4

# RARELY EDIT BELOW

# how many time slices to skip at a time
TSTEP = 1

# File format.  are the jackkknife blocks in ascii or hdf5?
STYPE = 'ascii'
STYPE = 'hdf5'

# prefix for hdf5 dataset location;
# ALTS will be tried if HDF5_PREFIX doesn't work
GROUP_LIST = ['I1', 'I0', 'I2']

# optional, scale parameter to set binds
SCALE = 1e13
if EFF_MASS_METHOD < 3:
    # additive constant added to effective mass functions
    SCALE = 1e11

# bounds for fit parameters
# for use with L-BFGS-B
BINDS = ((SCALE*.1, 10*SCALE), (.4, .6), (.01*SCALE, .03*SCALE))
BINDS_LSQ = ([-np.inf, -np.inf, -9e08], [np.inf, np.inf, -6e08])
# BINDS = ((scale*.01, 30*scale), (0, .8), (.01*scale*0, scale))

# fineness of scale to plot (higher is more fine)

FINE = 1000.0

# do inverse via a correlation matrix (for higher numerical stability)

CORRMATRIX = False
CORRMATRIX = True

# use experimental average of jackknife error bars for error bars
# this switch appears to have a negligible impact on the result, and
# should be identical to using the avg. cov. in the infinite statistics limit

ERROR_BAR_METHOD = 'jk'
ERROR_BAR_METHOD = 'avgcov'

# method used by the scipy.optimize.minimize
# other internals will need to be edited if you change this
# it's probably not a good idea

METHOD = 'Nelder-Mead'
# METHOD = 'L-BFGS-B'

# jackknife correction? "YES" or "NO"
# correction only happens if multiple files are processed

JACKKNIFE = 'YES'

# print correlation function, and sqrt(diag(cov)) and exit

PRINT_CORR = True
PRINT_CORR = False

# -------BEGIN POSSIBLY OBSOLETE------#

# multiply both sides of the gevp matrix by norms

# NORMS = [[1.0/(16**6), 1.0/(16**3)], [1.0/(16**3), 1]]
NORMS = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]

# GENERALIZED PENCIL OF FUNCTION (see arXiv:1010.0202, for use with GEVP)
# if non-zero, set to 1 (only do one pencil,
# more than one is supported, but probably not a good idea - see ref above)

NUM_PENCILS = 0
PENCIL_SHIFT = 1  # paper set shift to 4

# the boundary for when the fitter warns you if the eigenvalues
# of your covariance are very small

EIGCUT = 10**(-23)

# if set to true, AUTO_FIT uses curve_fit from scipy.optimize
# to bootstrap START_PARAMS.  Bounds must still be set manually.
# Bounds are used to find very rough start parameters: taken as the midpoints
# Probably, you should set FIT to False to first find some reasonable bounds.
# If ASSISTED_FIT is also True,
# use start_params to find the guess for the AUTO fitter
# (for use with L-BFGS-B)

AUTO_FIT = False
# AUTO_FIT = False

# ASSISTED_FIT = True
ASSISTED_FIT = False

# boundary scale for zero'ing out a failing inverse Hessian
# (neg diagaonal entrie(s)).  Below 1/(CUTOFF*SCALE), an entry is set to 0
CUTOFF = 10**(7)

# additive constant subtracted by hand from exact effective mass functions
# questionable, since this is an extra, badly optimized, fit parameter
C = 1.935*SCALE*0 if (ADD_CONST and EFF_MASS_METHOD == 1 and EFF_MASS) else 0

# rhs time separation (t0) of GEVP matrix
# (used for non eff mass fits), probably obsolete

TRHS = 6

# -------END POSSIBLY OBSOLETE------#

# FIT FUNCTION/PROCESSING FUNCTION SELECTION


def fit_func_1p(ctime, trial_params):
    """one parameter eff. mass fit function
    for EFF_MASS_METHOD = 3
    """
    corrs = [exp(-trial_params[0]*(ctime+i*TSTEP)) +
             exp(-trial_params[0]*(LT-(ctime+i*TSTEP)))
             for i in range(RANGE1P)]
    return ratio(corrs, ctime, nocheck=True)

# library of functions to fit.  define them in the usual way
if ADD_CONST:
    def fit_func_exp(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) + exp(
            -trial_params[1]*(LT-ctime))) + trial_params[2]

    def ratio(corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (and take log)"""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if not nocheck:
            zero_p(corrs[1], corrs[2], times)
        sol = (corrs[1]-corrs[0])/(corrs[2]-corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        sol = log(sol) if LOG else sol
        return sol

    def acosh_ratio(corrs, times=None, nocheck=False):
        """Process data into effective mass ratio,
        for an exact call to acosh."""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if not nocheck:
            zero_p(corrs[1]-C, times[1:])
        sol = (corrs[0]-corrs[1]+corrs[2]-corrs[3])/2.0/(corrs[1]-corrs[2])
        if not nocheck:
            testsol(sol, corrs, times)
        return sol

    def fit_func_sym(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv))
        for EFF_MASS_METHOD = 2
        """
        return trial_params[0]*(
            exps(-trial_params[1]*ctime) +
            exps(-trial_params[1]*(LT-ctime)))+trial_params[2]

    def fit_func_exp_gevp(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        return ((exp(-trial_params[0]*ctime) +
                 exp(-trial_params[1]*(LT-ctime))) + trial_params[2])/(
                     (exp(-trial_params[0]*(TRHS)) +
                      exp(-trial_params[1]*(LT-(TRHS)))) + trial_params[2])

else:
    def fit_func_exp(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) +
                                exp(-trial_params[1]*(LT-ctime)))

    def ratio(corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio
        (and take log), no additive constant
        """
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if not nocheck:
            zero_p(corrs[1], times[1])
        sol = (corrs[0])/(corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        sol = log(sol) if LOG else sol
        return sol

    def acosh_ratio(corrs, times=None, nocheck=False):
        """Process data into effective mass ratio,
        for an exact call to acosh (no additive constant)."""
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if not nocheck:
            zero_p(corrs[1]-C, times[1])
        sol = (corrs[0]+corrs[2]-2*C)/2/(corrs[1]-C)
        if not nocheck:
            testsol(sol, corrs, times)
        return sol

    def fit_func_sym(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv))
        for EFF_MASS_METHOD = 2
        """
        return trial_params[0]*(
            exps(-trial_params[1]*ctime) +
            exps(-trial_params[1]*(LT-ctime)))

    def fit_func_exp_gevp(ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        return (exp(-trial_params[0]*ctime) +
                exp(-trial_params[1]*(LT-ctime)))/(
                    (exp(-trial_params[0]*(TRHS)) +
                     exp(-trial_params[1]*(LT-(TRHS)))))


# select which of the above library functions to use

ORIGL = len(START_PARAMS)
if EFF_MASS:

    # check len of start params
    if ORIGL != 1 and EFF_MASS_METHOD != 2:
        print("***ERROR***")
        print("dimension of GEVP matrix and start params do not match")
        print("(or for non-gevp fits, the start_param len>1)")
        sys.exit(1)

    # select fit function
    if EFF_MASS_METHOD == 1 or EFF_MASS_METHOD == 2 or EFF_MASS_METHOD == 4:
        if RESCALE != 1.0:
            def prefit_func(_, trial_params):
                """eff mass method 1, fit func, single const fit
                """
                return trial_params
        else:
            if len(START_PARAMS) == 1:
                def prefit_func(_, trial_params):
                    """eff mass method 1, fit func, single const fit
                    """
                    return RESCALE*trial_params
            else:
                def prefit_func(_, trial_params):
                    """eff mass method 1, fit func, single const fit
                    """
                    return [RESCALE*trial_param for
                            trial_param in trial_params]

    elif EFF_MASS_METHOD == 3:
        if RESCALE != 1.0:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [RESCALE * fit_func_1p(ctime, trial_params[j:j+1])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [fit_func_1p(ctime, trial_params[j:j+1])
                        for j in range(MULT)]
    else:
        print("***ERROR***")
        print("check config file fit func selection.")
        sys.exit(1)

else:
    if GEVP:
        # check len of start params
        if ORIGL != 2 and FIT:
            print("***ERROR***")
            print("flag 1 length of start_params invalid")
            sys.exit(1)
        # select fit function
        if RESCALE != 1.0:
            def prefit_func(ctime, trial_params):
                """gevp fit func, non eff mass"""
                return [RESCALE*fit_func_exp_gevp(
                    ctime, trial_params[j*ORIGL:(j+1)*ORIGL])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """gevp fit func, non eff mass"""
                return [fit_func_exp_gevp(
                    ctime, trial_params[j*ORIGL:(j+1)*ORIGL])
                        for j in range(MULT)]
    else:
        if FIT:
            # check len of start params
            if ORIGL != (3 if ADD_CONST else 2):
                print("***ERROR***")
                print("flag 2 length of start_params invalid")
                sys.exit(1)
            # select fit function
            if RESCALE != 1.0:
                def prefit_func(ctime, trial_params):
                    """Rescaled exp fit function."""
                    return RESCALE*fit_func_exp(ctime, trial_params)
            else:
                def prefit_func(ctime, trial_params):
                    """Prefit function, copy of exponential fit function."""
                    return fit_func_exp(ctime, trial_params)
        else:
            def prefit_func(__, _):
                """fit function doesn't do anything because FIT = False"""
                pass

# DO NOT EDIT BELOW THIS LINE
# for general pencil of function

if FIT:
    FIT_FUNC_COPY = copy(prefit_func)

if NUM_PENCILS > 0:
    def fit_func(ctime, trial_params):
        """Fit function (num_pencils > 0)."""
        return np.hstack(
            [RESCALE*FIT_FUNC_COPY(
                ctime, trial_params[i*len(START_PARAMS):(i+1)*len(
                    START_PARAMS)]) for i in range(2**NUM_PENCILS)])
else:
    def fit_func(ctime, trial_params):
        """Fit function."""
        return prefit_func(ctime, trial_params)


MULT = len(GEVP_DIRS) if GEVP else 1
START_PARAMS = (list(START_PARAMS)*MULT)*2**NUM_PENCILS
RANGE1P = 3 if ADD_CONST else 2
if EFF_MASS:
    if EFF_MASS_METHOD in [1, 3, 4]:
        print("rescale set to 1.0")
        RESCALE = 1.0
if TSEP == 0:
    CALC_PHASE_SHIFT = False
