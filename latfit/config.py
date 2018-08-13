"""Config for lattice fitter."""
import sys
import re
from math import sqrt, pi
from collections import namedtuple
from copy import copy
import numpy as np
import latfit.analysis.misc as misc
from latfit.analysis.gevp_dirs import gevp_dirs
from latfit.fit_funcs import FitFunctions
import latfit.fit_funcs
from latfit.utilities import read_file as rf
from latfit.utilities import op_compose as opc

# TYPE OF FIT

# Do a fit at all?

FIT = False
FIT = True

# solve the generalized eigenvalue problem (GEVP)

GEVP = False
GEVP = True

# Plot Effective Mass? True or False

EFF_MASS = False
EFF_MASS = True
EFF_MASS = True if GEVP else EFF_MASS

# EFF_MASS_METHOD 1: analytic for arg to acosh
# (good for when additive const = 0, but noiser than 3 and 4)
# EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
# (bad for all cases; DO NOT USE.  It doesn't converge very often.)
# EFF_MASS_METHOD 3: one param fit
# EFF_MASS_METHOD 4: same as 2, but equations have one free parameter (
# traditional effective mass method), typically a fast version of 3

EFF_MASS_METHOD = 4

# METHODS/PARAMS

# super jackknife cutoff:  first n configs have variance in exact, n to N=total length:
# variance in sloppy.  if n= 0 then don't do superjackknife (sloppy only)
SUPERJACK_CUTOFF = 10
SUPERJACK_CUTOFF = 0

# isospin value, (0,1,2 supported)
ISOSPIN = 2

# group irrep
IRREP = 'T_1_2MINUS'
IRREP = 'T_1_MINUS'
IRREP = 'T_1_3MINUS'
IRREP = 'T_1_MINUS'
IRREP = 'A1x_mom011'
IRREP = 'A_1PLUS_mom000'
IRREP = 'A1_mom11'
# non-zero center of mass
MOMSTR = opc.get_comp_str(IRREP)

# how many loop iterations until we start using random samples
MAX_ITER = 100
# average relative error on the parameter errors to attempt to achieve
# if achieved, exit the fit loop
FITSTOP = 0.01
# If set to True, speed up the fit loop by looking at models
# which resemble non-interacting (dispersive) energies first
# this biases the results, so turn off if doing a final fit
BIASED_SPEEDUP = True
BIASED_SPEEDUP = False
# MAX_RESULTS is the max number of usable fit ranges to average over
# (useful for random fitting; the fitter will otherwise take a long time)
# set this to np.inf to turn off
MAX_RESULTS = np.inf
MAX_RESULTS = 15

# automatically generate free energies, no need to modify if GEVP
# (einstein dispersion relation sqrt(m^2+p^2))
L_BOX = 24
AINVERSE = 1.015
PION_MASS = 0.13975*AINVERSE
misc.BOX_LENGTH = L_BOX
misc.MASS = PION_MASS/AINVERSE
DISP_ENERGIES = opc.free_energies(IRREP, misc.MASS, L_BOX) if GEVP else []
# manual, e.g.
# DISP_ENERGIES = [2*misc.dispersive([0, 0, 1])]

# don't include the sigma in the gevp fits
SIGMA = True if ISOSPIN == 0 else False
DIM = len(DISP_ENERGIES) + (1 if SIGMA else 0) # no need to change

# time extent (1/2 is time slice where the mirroring occurs in periodic bc's)

TSEP_VEC = [0]
TSEP_VEC = [3, 3]
TSEP_VEC = [3, 0, 3]
TSEP_VEC = [3]*DIM if GEVP else [0]
LT = 64

# print raw gevp info (for debugging source construction)

GEVP_DEBUG = True
GEVP_DEBUG = False


# additive constant, due to around-the-world effect
# do the subtraction at the level of the GEVP matrix
MATRIX_SUBTRACTION = False
MATRIX_SUBTRACTION = True
MATRIX_SUBTRACTION = False if GEVP_DEBUG else MATRIX_SUBTRACTION
DELTA_T_MATRIX_SUBTRACTION = 3 if not GEVP_DEBUG else 0
DELTA_T2_MATRIX_SUBTRACTION = 3 if not GEVP_DEBUG else 0
# do the subtraction at the level of the eigenvalues
ADD_CONST_VEC = [True]*DIM if GEVP else [False]
ADD_CONST_VEC = [False]*DIM if GEVP_DEBUG else ADD_CONST_VEC
ADD_CONST = ADD_CONST_VEC[0] or (MATRIX_SUBTRACTION and GEVP)  # no need to modify
# second order around the world delta energy (E(k_max)-E(k_min)),
# set to None if only subtracting for first order or if all orders are constant
DELTA_E2_AROUND_THE_WORLD = misc.dispersive([1,1,1])-misc.dispersive([1,0,0])
DELTA_E2_AROUND_THE_WORLD = 0
DELTA_E2_AROUND_THE_WORLD = None
# DELTA_E2_AROUND_THE_WORLD -= DELTA_E_AROUND_THE_WORLD # (below)
DELTA_E2_AROUND_THE_WORLD = None if not GEVP else DELTA_E2_AROUND_THE_WORLD

# exclude from fit range these time slices.  shape = (GEVP dim, tslice elim)

FIT_EXCL = [[], [2, 5, 6, 7, 8]]
FIT_EXCL = [[], [], []]
FIT_EXCL = [[5], [5, 6], [5, 6], []]
FIT_EXCL = [[], [5, 10, 11, 12, 13, 14, 15, 16, 17],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
FIT_EXCL = [[] for _ in range(DIM)] if GEVP else [[]]

# eliminate problematic configs.
# Simply set this to a list of ints indexing the configs,
# e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

ELIM_JKCONF_LIST = []

# dynamic binning of configs.  BINNUM is number of configs per bin.
BINNUM = 1

# stringent tolerance for minimizer?  true = stringent
MINTOL = True
MINTOL = False

# rescale the fit function by factor RESCALE
RESCALE = 1e12
RESCALE = 1.0

# T0 behavior for GEVP (t/2 or t-1)

T0 = 'TMINUS1' if ISOSPIN == 2 else 'ROUND'
T0 = 'ROUND' # ceil(t/2)
T0 = 'TMINUS1' # t-1

# Pion ratio?  Put single pion correlators in the denominator
# of the eff mass equation to get better statistics.
PIONRATIO = True
PIONRATIO = False

# use fixed pion mass in ratio fits?
USE_FIXED_MASS = False
USE_FIXED_MASS = True

# starting values for fit parameters
if EFF_MASS and EFF_MASS_METHOD != 2:
    START_PARAMS = [.5]
    if PIONRATIO:
        START_PARAMS = [.05, 0.0005]
else:
    if ADD_CONST:
        START_PARAMS = [0.0580294, -0.003, 0.13920]
    else:
        START_PARAMS = [1.18203895, 0.046978036e-01]



# modify the configs used and bin

# Uncorrelated fit? True or False

UNCORR = True
UNCORR = False

# Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

LOG = False
LOG = True
LOG = False if PIONRATIO else LOG

# Jackknife fit? (keep double for correctness, others not supported)

JACKKNIFE_FIT = 'FROZEN'
JACKKNIFE_FIT = 'SINGLE'
JACKKNIFE_FIT = 'DOUBLE'

# pickle, unpickle

PICKLE = 'clean'
PICKLE = 'unpickle'
PICKLE = 'pickle'
PICKLE = None

# DISPLAY PARAMETERS
# no title given takes the current working directory as the title

# title prefix

# p_cm = 001, no need to modify 
PSTR_TITLE = r"$\vec{p}_{CM}=$"+rf.ptostr(rf.procmom(MOMSTR))

if GEVP:
    if SIGMA and ISOSPIN == 0:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, $\pi\pi, \sigma$, ' + PSTR_TITLE + ' '
    elif ISOSPIN == 2:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, I2, $\pi\pi$, ' + PSTR_TITLE + ' '
    elif ISOSPIN == 1:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, I1, $\pi\pi,~\rho$ ' + PSTR_TITLE + ' '
    else:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, $\pi\pi$, ' + PSTR_TITLE + ' '

else:
    TITLE_PREFIX = '24c '

if SUPERJACK_CUTOFF:
    pass
    # TITLE_PREFIX = TITLE_PREFIX + 'exact '
else:
    TITLE_PREFIX = TITLE_PREFIX + '(zmobius) '
if MATRIX_SUBTRACTION and DELTA_E2_AROUND_THE_WORLD is not None:
    TITLE_PREFIX = TITLE_PREFIX + 'matdt'+\
        str(DELTA_T_MATRIX_SUBTRACTION)+','+\
        str(DELTA_T2_MATRIX_SUBTRACTION)+' '
elif MATRIX_SUBTRACTION:
    TITLE_PREFIX = TITLE_PREFIX + 'matdt'+\
        str(DELTA_T_MATRIX_SUBTRACTION)+' '
elif True in ADD_CONST_VEC:
    TITLE_PREFIX = TITLE_PREFIX + 'eigdt1 '

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

# calculate the I=0 phase shift?
CALC_PHASE_SHIFT = False
CALC_PHASE_SHIFT = True

# box plot (for effective mass tolerance display)?
BOX_PLOT = False
BOX_PLOT = True

# dispersive lines
PLOT_DISPERSIVE = True
PLOT_DISPERSIVE = False if not GEVP else True

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

# plot anything at all?

NO_PLOT = True
NO_PLOT = False

# -------BEGIN POSSIBLY OBSOLETE------#

# multiply both sides of the gevp matrix by norms

# NORMS = [[1.0/(16**6), 1.0/(16**3)], [1.0/(16**3), 1]]
NORMS = [[1.0/10**6, 1.0/10**3/10**(2.5),
          1.0/10**3/10**5.5],
         [1.0/10**3/10**2.5, 1.0/10**5,
          1.0/10**2.5/10**5.5],
         [1.0/10**3/10**5.5,
          1.0/10**2.5/10**5.5, 1.0/10**11]]
NORMS = [[1.0]*DIM]*DIM

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
if not GEVP_DEBUG:
    assert MATRIX_SUBTRACTION or not GEVP,\
        "Must subtract around the world constant at GEVP level"
else:
    MATRIX_SUBTRACTION = False
MATRIX_SUBTRACTION = False if not GEVP else MATRIX_SUBTRACTION
if MATRIX_SUBTRACTION:
    for i, _ in enumerate(ADD_CONST_VEC):
        ADD_CONST_VEC[i] = MATRIX_SUBTRACTION
ADD_CONST_VEC = list(map(int, ADD_CONST_VEC))

# library of functions to fit.  define them in the usual way

FITS = FitFunctions()

UP = namedtuple('update', ['add_const', 'log', 'lt', 'c', 'tstep', 'pionmass', 'pionratio'])
UP.add_const = ADD_CONST
UP.log = LOG
UP.c = C
# make global tstep equal to delta t so fit functions below will be setup correctly
UP.tstep = TSTEP if not GEVP or GEVP_DEBUG else DELTA_T_MATRIX_SUBTRACTION
UP.tstep2 = TSTEP if not GEVP or GEVP_DEBUG else DELTA_T2_MATRIX_SUBTRACTION
UP.pionmass = misc.MASS
UP.pionratio = PIONRATIO
UP.lent = LT
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
                return [RESCALE * FITS.f['fit_func_1p'][
                    ADD_CONST_VEC[j]](ctime, trial_params[j:j+1*ORIGL], LT_VEC[j])
                        for j in range(MULT)]
        else:
            def prefit_func(ctime, trial_params):
                """eff mass 3, fit func, rescaled
                """
                return [FITS.f['fit_func_1p'][ADD_CONST_VEC[j]](
                    ctime, trial_params[j:j+1*ORIGL], LT_VEC[j])
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

if ISOSPIN != 0:
    SIGMA = False
GEVP_DIRS = gevp_dirs(ISOSPIN, MOMSTR, IRREP, DIM, SIGMA)
print(GEVP_DIRS)
MULT = len(GEVP_DIRS) if GEVP else 1
if GEVP:
    assert DIM == MULT, "Error in GEVP_DIRS length."
assert not(LOG and PIONRATIO), "Taking a log is improper when doing a pion ratio fit."
assert len(LT_VEC) == MULT, "Must set time separation separately for"+\
    " each diagonal element of GEVP matrix"
assert len(ADD_CONST_VEC) == MULT, "Must separately set, whether or"+\
    " not to use an additive constant in the fit function, for each diagonal element of GEVP matrix"
assert not (PIONRATIO and EFF_MASS_METHOD == 1), "No exact inverse"+\
    " function exists for pion ratio method."
assert not (PIONRATIO and EFF_MASS_METHOD == 2), "Symbolic solve"+\
" not supported for pion ratio method."
START_PARAMS = (list(START_PARAMS)*MULT)*2**NUM_PENCILS
latfit.fit_funcs.USE_FIXED_MASS = USE_FIXED_MASS
UP.tstep = TSTEP # revert back
# MINTOL = True if not BIASED_SPEEDUP else MINTOL # probably better, but too slow
FITS.select(UP)
if PIONRATIO:
    FITS.test()
if EFF_MASS:
    if EFF_MASS_METHOD in [1, 3, 4]:
        print("rescale set to 1.0")
        RESCALE = 1.0
# change this if the slowest pion is not stationary
DELTA_E_AROUND_THE_WORLD = misc.dispersive(rf.procmom(MOMSTR))-misc.MASS if GEVP else 0
if DELTA_E2_AROUND_THE_WORLD is not None:
    DELTA_E2_AROUND_THE_WORLD -= DELTA_E_AROUND_THE_WORLD
print("Assuming slowest around the world term particle is stationary.  Emin=",
      DELTA_E_AROUND_THE_WORLD)
print("2nd order around the world term, delta E=",
      DELTA_E2_AROUND_THE_WORLD)
assert EFF_MASS_METHOD == 4 or not MATRIX_SUBTRACTION, "Matrix subtraction supported"+\
    " only with eff mass method 4"
assert JACKKNIFE_FIT == 'DOUBLE', "Other jackknife fitting methods no longer supported."
assert NUM_PENCILS == 0, "this feature is less tested, use at your own risk (safest to have NUM_PENCILS==0)"
