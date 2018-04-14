"""Config for lattice fitter."""
import sys
from collections import namedtuple
from copy import copy
import numpy as np
import latfit.analysis.misc as misc
from latfit.analysis.gevp_dirs import gevp_dirs
from latfit.fit_funcs import FitFunctions
import latfit.fit_funcs
from math import sqrt, pi

# TYPE OF FIT

# plot anything at all?

NO_PLOT = True
NO_PLOT = False

# Do a fit at all?

FIT = True
FIT = False

# Jackknife fit?

JACKKNIFE_FIT = 'FROZEN'
JACKKNIFE_FIT = 'SINGLE'
JACKKNIFE_FIT = 'DOUBLE'

# Plot Effective Mass? True or False

EFF_MASS = False
EFF_MASS = True

# EFF_MASS_METHOD 1: analytic for arg to acosh
# (good for when additive const = 0, but noiser than 3 and 4)
# EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
# (bad for all cases; DO NOT USE.  It doesn't converge very often.)
# EFF_MASS_METHOD 3: one param fit
# EFF_MASS_METHOD 4: same as 2, but equations have one free parameter (
# traditional effective mass method), typically a fast version of 3

EFF_MASS_METHOD = 4

# solve the generalized eigenvalue problem (GEVP)

GEVP = True
GEVP = False

# METHODS/PARAMS

# Uncorrelated fit? True or False

UNCORR = True
UNCORR = False

# Pion ratio?  Put single pion correlators in the denominator
# of the eff mass equation to get better statistics.
PIONRATIO = True
PIONRATIO = False

# use fixed pion mass in ratio fits?
USE_FIXED_MASS = False
USE_FIXED_MASS = True

# time extent (1/2 is time slice where the mirroring occurs in periodic bc's)

TSEP_VEC = [3, 3]
TSEP_VEC = [0]
LT = 64

# exclude from fit range these time slices.  shape = (GEVP dim, tslice elim)

FIT_EXCL = [[],[2,5,6, 7,8  ]]
FIT_EXCL = [[],[6, 7, 8]]
FIT_EXCL = [[]]

# additive constant
ADD_CONST_VEC = [True, True]
ADD_CONST_VEC = [False]
ADD_CONST = ADD_CONST_VEC[0]  # no need to modify

# isospin value (convenience switch)
ISOSPIN = 0
DIM = 2
# don't include the sigma in the gevp fits
SIGMA = True
SIGMA = False
# non-zero center of mass
MOMSTR = 'perm momtotal001'
MOMSTR = 'momtotal001'
MOMSTR = 'momtotal000'
# group irrep
IRREP = 'T_1_1MINUS'
IRREP = 'T_1_2MINUS'
IRREP = 'T_1_3MINUS'
IRREP = 'T_1_MINUS'
IRREP = 'A1_avg'
IRREP = 'A_1PLUS'

# calculate the I=0 phase shift?

L_BOX = 24
AINVERSE = 1.015
PION_MASS = 0.13975*AINVERSE
CALC_PHASE_SHIFT = False
CALC_PHASE_SHIFT = True
misc.BOX_LENGTH = L_BOX
misc.MASS = PION_MASS/AINVERSE

# dispersive lines
PLOT_DISPERSIVE = False
PLOT_DISPERSIVE = True
DISP_ADD = (2*pi/L_BOX)**2*1
DISP_ENERGIES = [misc.dispersive([0,0,0])+ misc.dispersive([0,0,1]), sqrt((2*misc.dispersive([0,0,1]))**2+DISP_ADD)]
DISP_ENERGIES = [2*misc.dispersive([0,0,1])]
DISP_ENERGIES = [misc.dispersive([0,0,1])]

# pickle, unpickle

PICKLE = 'clean'
PICKLE = 'unpickle'
PICKLE = 'pickle'
PICKLE = None

# Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

LOG = False
LOG = True
LOG = False if PIONRATIO else LOG

# stringent tolerance for minimizer?  true = stringent
MINTOL = False
MINTOL = True

# rescale the fit function by factor RESCALE
RESCALE = 1e12
RESCALE = 1.0

# starting values for fit parameters
if EFF_MASS and EFF_MASS_METHOD != 2:
    START_PARAMS = [.5]
    if PIONRATIO:
        START_PARAMS = [.05,0.0005]
else:
    if ADD_CONST:
        START_PARAMS = [0.0580294, -0.003, 0.13920]
    else:
        START_PARAMS = [1.18203895, 0.046978036e-01]



# modify the configs used and bin

# eliminate problematic configs.
# Simply set this to a list of ints indexing the configs,
# e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

ELIM_JKCONF_LIST = [18, 24, 11, 21, 28, 32, 12,
                    45, 26, 28, 33, 35, 40, 41, 43, 50]
ELIM_JKCONF_LIST = [2, 3]
ELIM_JKCONF_LIST = [4, 5, 6, 7]
ELIM_JKCONF_LIST = [6, 7, 8, 9, 10, 11]
ELIM_JKCONF_LIST = []


# dynamic binning of configs.  BINNUM is number of configs per bin.
BINNUM = 1

# DISPLAY PARAMETERS
# no title given takes the current working directory as the title

# title prefix
if GEVP:
    if DIM == 2:
        if ISOSPIN == 0:
            if SIGMA:
                TITLE_PREFIX = r'$\pi\pi, \sigma$, I0, ' + MOMSTR + ' '
            else:
                TITLE_PREFIX = r'$\pi\pi$, I0, ' + MOMSTR + ' '
        elif ISOSPIN == 2:
            TITLE_PREFIX = r'$\pi\pi$, I2, ' + MOMSTR + ' '
        elif ISOSPIN == 1:
            TITLE_PREFIX = r'$\pi\pi, \rho$ I1, ' + MOMSTR + ' '
    elif DIM == 3:
        if SIGMA and ISOSPIN == 0:
            TITLE_PREFIX = r'3x3 GEVP, $\pi\pi, \sigma$, ' + MOMSTR + ' '
        else:
            TITLE_PREFIX = r'3x3 GEVP, $\pi\pi$, ' + MOMSTR + ' '
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

# not correct, do not modify, should be 0
PTOTSQ = 0

# DO NOT MODIFY
LT_VEC = []
for tsep in TSEP_VEC:
    LT_VEC.append(LT-2*tsep)
LT = LT_VEC[0]
ADD_CONST_VEC = list(map(int, ADD_CONST_VEC))

# library of functions to fit.  define them in the usual way

FITS = FitFunctions()

UP = namedtuple('update', ['add_const', 'log', 'lt', 'c', 'tstep', 'pionmass', 'pionratio'])
UP.add_const = ADD_CONST
UP.log = LOG
UP.c = C
UP.tstep = TSTEP
UP.pionmass = misc.MASS
UP.pionratio = PIONRATIO
UP.lt = LT
FITS.select(UP)

# END DO NOT MODIFY

# -------END POSSIBLY OBSOLETE------#

# FIT FUNCTION/PROCESSING FUNCTION SELECTION


# select which of the above library functions to use

ORIGL = len(START_PARAMS)
if EFF_MASS:

    # check len of start params
    if ORIGL != 1 and EFF_MASS_METHOD != 2 and not PIONRATIO:
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
                return [RESCALE * FITS.f['fit_func_1p'][ADD_CONST_VEC[j]](ctime, trial_params[j:j+1*ORIGL], LT_VEC[j])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [FITS.f['fit_func_1p'][ADD_CONST_VEC[j]](ctime, trial_params[j:j+1*ORIGL], LT_VEC[j])
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
                return [RESCALE*FITS.f['fit_func_exp_gevp'][ADD_CONST_VEC[j]](
                    ctime, trial_params[j*ORIGL:(j+1)*ORIGL], LT_VEC[j])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """gevp fit func, non eff mass"""
                return [FITS.f['fit_func_exp_gevp'][ADD_CONST_VEC[j]](
                    ctime, trial_params[j*ORIGL:(j+1)*ORIGL], LT_VEC[j])
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
                if PIONRATIO:
                    def prefit_func(ctime, trial_params):
                        """Pion ratio"""
                        return RESCALE*FITS[
                            'pion_ratio'](ctime, trial_params)
                else:
                    def prefit_func(ctime, trial_params):
                        """Rescaled exp fit function."""
                        return RESCALE*FITS[
                            'fit_func_exp'](ctime, trial_params)
            else:
                if PIONRATIO:
                    def prefit_func(ctime, trial_params):
                        """Prefit function, copy of
                        exponential fit function."""
                        return FITS['pion_ratio'](ctime, trial_params)
                else:
                    def prefit_func(ctime, trial_params):
                        """Prefit function, copy of
                        exponential fit function."""
                        return FITS['fit_func_exp'](ctime, trial_params)
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

GEVP_DIRS = gevp_dirs(ISOSPIN, MOMSTR, IRREP, DIM, SIGMA)
print(GEVP_DIRS)
MULT = len(GEVP_DIRS) if GEVP else 1
if GEVP:
    assert DIM == MULT, "Error in GEVP_DIRS length."
assert not(LOG and PIONRATIO), "Taking a log is improper when doing a pion ratio fit."
assert len(LT_VEC) == MULT, "Must set time separation separately for each diagonal element of GEVP matrix"
assert len(ADD_CONST_VEC) == MULT, "Must separately set, whether or not to use an additive constant in the fit function, for each diagonal element of GEVP matrix"
assert PIONRATIO or not EFF_MASS_METHOD == 1, "No exact inverse function exists for pion ratio method."
assert PIONRATIO or not EFF_MASS_METHOD == 2, "Symbolic solve not supported for pion ratio method."
START_PARAMS = (list(START_PARAMS)*MULT)*2**NUM_PENCILS
latfit.fit_funcs.USE_FIXED_MASS = USE_FIXED_MASS
if PIONRATIO:
    FITS.test()
if EFF_MASS:
    if EFF_MASS_METHOD in [1, 3, 4]:
        print("rescale set to 1.0")
        RESCALE = 1.0
