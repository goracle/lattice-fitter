"""Config for lattice fitter."""
from copy import copy
import numpy as np
import latfit.analysis.misc as misc
from latfit.analysis.gevp_dirs import gevp_dirs
from latfit.analysis.irr2tex import irr2tex
from latfit.fit_funcs import FitFunctions
import latfit.fit_funcs
from latfit.utilities import read_file as rf
from latfit.utilities import op_compose as opc
from latfit.logger import setup_logger
from latfit.utilities.postprod.h5jack import check_ids
from latfit import fitfunc
import latfit.checks.checks_and_statements as sands
import latfit.mathfun.elim_jkconfigs as elimjk
import latfit.extract.binout as binout

setup_logger()


# TYPE OF FIT

# Do a fit at all?

FIT = False
FIT = True

# solve the generalized eigenvalue problem (GEVP)

GEVP = False
GEVP = True

# METHODS/PARAMS

# isospin value, (0, 1, 2 supported)
ISOSPIN = 2

# group irrep
IRREP = 'T_1_2MINUS'
IRREP = 'T_1_MINUS'
IRREP = 'T_1_3MINUS'
IRREP = 'A1x_mom011'
IRREP = 'A1_avg_mom111'
IRREP = 'A1_avg_mom111'
IRREP = 'A1_mom1'
IRREP = 'A_1PLUS_mom000'

if ISOSPIN == 1:
    # too noisy to even plot
    IRREP = 'A_2MINUS_mom11' # very noisy, no go
    IRREP = 'A_1PLUS_mom11' # very noisy, no go
    IRREP = 'A_1PLUS_avg_mom111' # very noisy, no go
    # working
    IRREP = 'A_1PLUS_mom1' # t-t0=3 has decent t=8-10 plateau
    IRREP = 'A_2PLUS_mom11' # third state dies early, but otherwise decent
    IRREP = 'B_mom1' # strong 2nd and 3rd state overlap
    IRREP = 'B_mom111' # decent plateau, bottom state very well resolved
    # control
    IRREP = 'T_1_3MINUS_mom000'
    IRREP = 'T_1_MINUS'

# non-zero center of mass
MOMSTR = opc.get_comp_str(IRREP)

# lattice ensemble to take gauge config average over

LATTICE_ENSEMBLE = '32c'
LATTICE_ENSEMBLE = '24c'

## THE GOAL IS TO MINIMIZE EDITS BELOW THIS POINT

SYS_ENERGY_GUESS = 1.2
SYS_ENERGY_GUESS = None
SYS_ENERGY_GUESS = None if not FIT else SYS_ENERGY_GUESS
SYS_ENERGY_GUESS = None if ISOSPIN != 1 else SYS_ENERGY_GUESS
SYS_ENERGY_GUESS = None if not GEVP else SYS_ENERGY_GUESS

# T0 behavior for GEVP (t/2 or t-1)

T0 = 'ROUND' # ceil(t/2)
T0 = 'LOOP' # ceil(t/2)
T0 = 'TMINUS3' # t-1
if LATTICE_ENSEMBLE == '24c':
    T0 = 'TMINUS1' if ISOSPIN != 1 else 'TMINUS1'
    T0 = 'TMINUS1' if IRREP == 'A_1PLUS_mom000' else T0
elif LATTICE_ENSEMBLE == '32c':
    T0 = 'TMINUS1' if ISOSPIN != 2 else 'TMINUS3'
#T0 = 'TMINUS3' if ISOSPIN != 2 else 'TMINUS1'

# print raw gevp info (for debugging source construction)

GEVP_DEBUG = True
GEVP_DEBUG = False

if LATTICE_ENSEMBLE == '24c':
    DELTA_T_MATRIX_SUBTRACTION = 1 if not GEVP_DEBUG else 0
    DELTA_T2_MATRIX_SUBTRACTION = 1 if not GEVP_DEBUG else 0
if LATTICE_ENSEMBLE == '32c':
    if IRREP == 'A_1PLUS_mom000':
        DELTA_T_MATRIX_SUBTRACTION = 4 if not GEVP_DEBUG else 0
    else:
        DELTA_T_MATRIX_SUBTRACTION = 3 if not GEVP_DEBUG else 0
    DELTA_T2_MATRIX_SUBTRACTION = 3 if not GEVP_DEBUG else 0
# do the subtraction at the level of the eigenvalues

# Pion ratio?  Put single pion correlators in the denominator
# of the eff mass equation to get better statistics.
PIONRATIO = False
PIONRATIO = True
PIONRATIO = False if not GEVP else PIONRATIO

# use the pion ratio to correct systematic
# (lattice spacing) error?
# if not, we can use it to correct statistical error
MINIMIZE_STAT_ERROR_PR = True
MINIMIZE_STAT_ERROR_PR = False
if LATTICE_ENSEMBLE == '24c':
    MINIMIZE_STAT_ERROR_PR = False
elif LATTICE_ENSEMBLE == '32c':
    MINIMIZE_STAT_ERROR_PR = True

# take derivative of GEVP eigenvalues
GEVP_DERIV = True
GEVP_DERIV = False
GEVP_DERIV = False if not GEVP else GEVP_DERIV

# Plot Effective Mass? True or False

EFF_MASS = False
EFF_MASS = True
EFF_MASS = True if GEVP else EFF_MASS

# set the minimum number of points to qualify a data subset as a fit range
RANGE_LENGTH_MIN = 0
RANGE_LENGTH_MIN = 3
RANGE_LENGTH_MIN = 2 if not GEVP and EFF_MASS else RANGE_LENGTH_MIN

# only loop over fit ranges with one or two time slices
# (useful for error optimization after a full fit range loop)
ONLY_SMALL_FIT_RANGES = True
ONLY_SMALL_FIT_RANGES = False
ONLY_SMALL_FIT_RANGES = False if not RANGE_LENGTH_MIN else ONLY_SMALL_FIT_RANGES

# super jackknife cutoff:  first n configs have variance in exact, n to N=total length:
# variance in sloppy.  if n= 0 then don't do superjackknife (sloppy only)
SUPERJACK_CUTOFF = 0

# automatically generate free energies, no need to modify if GEVP
# (einstein dispersion relation sqrt(m^2+p^2))
if LATTICE_ENSEMBLE == '32c':
    L_BOX = 32
    AINVERSE = 1.3784
    PION_MASS = 0.10470*AINVERSE
    LT = 64
    SUPERJACK_CUTOFF = 17
elif LATTICE_ENSEMBLE == '24c':
    L_BOX = 24
    AINVERSE = 1.015
    PION_MASS = 0.13975*AINVERSE
    LT = 64
    SUPERJACK_CUTOFF = 14
elif LATTICE_ENSEMBLE == '16c':
    L_BOX = 16
    AINVERSE = 1.0*np.nan
    PION_MASS = 0.3*AINVERSE
    LT = 32
    SUPERJACK_CUTOFF = 0
SUPERJACK_CUTOFF = 0 if not check_ids()[-2] else SUPERJACK_CUTOFF
binout.SUPERJACK_CUTOFF = SUPERJACK_CUTOFF

# If the first SUPERJACK_CUTOFF configs are exact, this simple switch
# skips reading them in
# and only looks at the jackknife blocks for the remaining configs
# (but it doesn't and can't remove any ama constant)
SLOPPYONLY = True
SLOPPYONLY = False
binout.SLOPPYONLY = SLOPPYONLY

# dynamic binning of configs.  BINNUM is number of configs per bin.
BINNUM = 1
binout.BINNUM = BINNUM

# halve the data to check for consistencies (debug options)
HALF = 'first half'
HALF = 'drop fourth eighth'
HALF = 'first half'
HALF = 'drop fourth quarter'
HALF = 'full'
# jackknife correction? "YES" or "NO"
# correction only happens if multiple files are processed
# this JACKKNIFE paramter is obsolete.
JACKKNIFE = 'YES'
elimjk.JACKKNIFE = JACKKNIFE
if HALF != 'full':
    SUPERJACK_CUTOFF = 0
    print("HALF spec:", HALF)
    print("setting superjackknife cutoff to 0 (assuming no AMA)")
    assert not SUPERJACK_CUTOFF, \
        "AMA first half second half analysis not supported:"+str(
            SUPERJACK_CUTOFF)
elimjk.HALF = HALF
binout.HALF = HALF

# eliminate problematic configs.
# Simply set this to a list of ints indexing the configs,
# e.g. ELIM_JKCONF_LIST = [0, 1] will eliminate the first two configs

#ELIM_JKCONF_LIST = [7, 8, 9, 10, 11, 12, 13, 14, 15, 186, 187, 188, 189, 190]

ELIM_JKCONF_LIST = []
elimjk.ELIM_JKCONF_LIST = list(ELIM_JKCONF_LIST)
misc.ELIM_JKCONF_LIST = list(ELIM_JKCONF_LIST)

misc.LATTICE = str(LATTICE_ENSEMBLE)
misc.BOX_LENGTH = L_BOX
misc.MASS = PION_MASS/AINVERSE
misc.IRREP = IRREP
misc.PIONRATIO = PIONRATIO
DISP_ENERGIES = opc.free_energies(
    IRREP, misc.massfunc(), L_BOX) if GEVP else []

# switch to include the sigma in the gevp fits
SIGMA = True if ISOSPIN == 0 else False

# get dispersive energies
DIM = len(DISP_ENERGIES) + (1 if SIGMA or ISOSPIN == 1 else 0) # no need to change
DIM -= 2 if 'mom000' in IRREP and ISOSPIN == 0 else 0
DIM = 1 if not GEVP else DIM
DIM = 3 if 'mom1' in IRREP and ISOSPIN == 0 and 'avg' not in IRREP else DIM
DISP_ENERGIES = list(np.array(DISP_ENERGIES)[:DIM])

# time extent (1/2 is time slice where the mirroring occurs in periodic bc's)
TSEP_VEC = [0]
TSEP_VEC = [3, 3]
TSEP_VEC = [3, 0, 3]
if LATTICE_ENSEMBLE == '24c':
    TSEP_VEC = [3 for _ in range(DIM)] if GEVP else [0]
if LATTICE_ENSEMBLE == '32c':
    TSEP_VEC = [4 for _ in range(DIM)] if GEVP else [0]
if GEVP:
    assert check_ids()[0] == TSEP_VEC[0], "ensemble mismatch:"+str(check_ids()[0])

# block size of blocked jackknifed technique
# usual jackknife sets this to 1
JACKKNIFE_BLOCK_SIZE = 1

# Bootstrap params
NBOOT = 1000 # until it saturates (should be infinity)
# whether to get accurate p-values.  usually best to leave this false
# it is automatically turned on for final fit
BOOTSTRAP = False
BOOTSTRAP = True

# continuum dispersion relation corrected using fits (true) or phat (false)
FIT_SPACING_CORRECTION = True
FIT_SPACING_CORRECTION = False
FIT_SPACING_CORRECTION = True if LATTICE_ENSEMBLE == '32c'\
    else FIT_SPACING_CORRECTION
FIT_SPACING_CORRECTION = False if ISOSPIN != 2 else FIT_SPACING_CORRECTION
FIT_SPACING_CORRECTION = True if PIONRATIO else FIT_SPACING_CORRECTION
misc.CONTINUUM = FIT_SPACING_CORRECTION


# no around the world subtraction at all
NOATWSUB = True
NOATWSUB = False
NOATWSUB = False if ISOSPIN == 2 else NOATWSUB
NOATWSUB = True if ISOSPIN == 1 else NOATWSUB

# additive constant, due to around-the-world effect
# do the subtraction at the level of the GEVP matrix
MATRIX_SUBTRACTION = False
MATRIX_SUBTRACTION = True
MATRIX_SUBTRACTION = False if IRREP != 'A_1PLUS_mom000' else MATRIX_SUBTRACTION
MATRIX_SUBTRACTION = False if NOATWSUB else MATRIX_SUBTRACTION
MATRIX_SUBTRACTION = False if GEVP_DEBUG else MATRIX_SUBTRACTION
MATRIX_SUBTRACTION = False if not GEVP else MATRIX_SUBTRACTION
MATRIX_SUBTRACTION = False if ISOSPIN == 1 else MATRIX_SUBTRACTION
MATRIX_SUBTRACTION = False if not GEVP else MATRIX_SUBTRACTION
ADD_CONST_VEC = [MATRIX_SUBTRACTION for _ in range(DIM)] if GEVP else [False]
ADD_CONST_VEC = [False for _ in range(DIM)] if GEVP_DEBUG else ADD_CONST_VEC
ADD_CONST = ADD_CONST_VEC[0] or (MATRIX_SUBTRACTION and GEVP) # no need to modify
# second order around the world delta energy (E(k_max)-E(k_min)),
# set to None if only subtracting for first order or if all orders are constant

##### Around the world delta energies.
##### Should be automatic, no need to adjust

## first delta E

# change this if the slowest pion is not stationary
DELTA_E_AROUND_THE_WORLD = misc.dispersive(rf.procmom(
    MOMSTR), continuum=FIT_SPACING_CORRECTION)-misc.massfunc() if GEVP\
    and MATRIX_SUBTRACTION and ISOSPIN != 1 else 0

## second delta E
DELTA_E2_AROUND_THE_WORLD = None

if IRREP == 'A1_mom1':
    # the exception to the usual pattern for p1
    E21 = misc.massfunc()
    E22 = misc.dispersive(rf.procmom(MOMSTR), continuum=FIT_SPACING_CORRECTION)
else:
    # the general E2
    E21 = misc.dispersive(opc.mom2ndorder(IRREP)[0], continuum=FIT_SPACING_CORRECTION)
    E22 = misc.dispersive(opc.mom2ndorder(IRREP)[1], continuum=FIT_SPACING_CORRECTION)
assert E21 is None or (np.all(E21 > 0) and np.all(E22 > 0))
assert E21 is None or (np.all(E22-E21) >= 0)
DELTA_E2_AROUND_THE_WORLD = E22-E21 if E21 is not None and E22 is not None else None
#MINE2 = min(E21, E22)
MINE2 = None # second order around the world fit no longer supported
print("2nd order momenta for around the world:",
      opc.mom2ndorder('A1_mom1'),
      opc.mom2ndorder('A1_mom11'), opc.mom2ndorder('A1_mom111'))
# we do the following subtraction to compensate below:
# DELTA_E2_AROUND_THE_WORLD -= DELTA_E_AROUND_THE_WORLD

# set to None if not GEVP
DELTA_E2_AROUND_THE_WORLD = None if not GEVP else DELTA_E2_AROUND_THE_WORLD

# set to None if in the center of mass frame
DELTA_E2_AROUND_THE_WORLD = None if rf.norm2(rf.procmom(MOMSTR)) == 0\
    else DELTA_E2_AROUND_THE_WORLD

# set to None if doing Isospin = 1
DELTA_E2_AROUND_THE_WORLD = None if ISOSPIN == 1\
    else DELTA_E2_AROUND_THE_WORLD

if not MATRIX_SUBTRACTION:
    DELTA_E2_AROUND_THE_WORLD = None

### delta e around the world section conclusion
if FIT_SPACING_CORRECTION:
    DELTA_E_AROUND_THE_WORLD = misc.uncorrect_epipi(DELTA_E_AROUND_THE_WORLD)
    DELTA_E2_AROUND_THE_WORLD = misc.uncorrect_epipi(
        DELTA_E2_AROUND_THE_WORLD)

## final delta e processing
if DELTA_E_AROUND_THE_WORLD is not None:
    DELTA_E_AROUND_THE_WORLD = misc.select_subset(DELTA_E_AROUND_THE_WORLD)
if DELTA_E2_AROUND_THE_WORLD is not None:
    DELTA_E2_AROUND_THE_WORLD = misc.select_subset(DELTA_E2_AROUND_THE_WORLD)


# exclude from fit range these time slices.  shape = (GEVP dim, tslice elim)

FIT_EXCL = [[], [2, 5, 6, 7, 8]]
FIT_EXCL = [[], [], []]
FIT_EXCL = [[5], [5, 6], [5, 6], []]
FIT_EXCL = [[], [5, 10, 11, 12, 13, 14, 15, 16, 17],
            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
FIT_EXCL = [[8.0], [8.0, 9.0, 13.0, 14.0],
            [8.0, 9.0], [8.0, 12.0, 13.0, 14.0]]
FIT_EXCL = [[] for _ in range(DIM)] if GEVP else [[]]
FIT_EXCL = [[], [6.0, 7, 13.0, 14.0, 15.0, 16.0],
            [6, 7, 12.0, 13.0, 14.0, 15.0, 16.0],
            [6, 7, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]]
FIT_EXCL = [[] for _ in range(DIM)] if GEVP else [[]]
assert len(FIT_EXCL) == DIM or not GEVP

# if true, do not loop over fit ranges.
NOLOOP = True
NOLOOP = False

# hints to eliminate
HINTS_ELIM = {}
HINTS_ELIM[11] = (4, 0)
HINTS_ELIM[12] = [(4, 3), (3, 2)]
HINTS_ELIM[15] = [(4, 3), (3, 0), (2, 1)]
HINTS_ELIM = {}
HINTS_ELIM[9] = [(5, 4), (4, 3)]
HINTS_ELIM[10] = [(5, 0), (4, 0)]
HINTS_ELIM[11] = [(4, 0)]
HINTS_ELIM[12] = [(5, 0), (4, 0), (3, 0)]
HINTS_ELIM[13] = [(5, 0), (4, 0), (3, 0), (2, 0)]
HINTS_ELIM[14] = [(5, 1), (4, 0), (3, 0), (2, 0)]
HINTS_ELIM[15] = [(5, 0), (4, 0), (3, 0), (2, 0)]
HINTS_ELIM = {}
HINTS_ELIM[5] = [(5, 0)]
HINTS_ELIM[7] = [(5, 0), (4, 0), (3, 0)]
HINTS_ELIM[8] = [(5, 0)]
HINTS_ELIM[9] = [(5, 0)]
HINTS_ELIM[10] = [(5, 0), (4, 0)]
HINTS_ELIM[14] = [(3, 0), (2, 0)]
HINTS_ELIM[15] = [(3, 0), (2, 0)]
HINTS_ELIM[15] = [(5, 0), (4, 0), (3, 0)]
HINTS_ELIM = {}
if ISOSPIN == 1:
    HINTS_ELIM[15] = [(4, 0), (3, 0), (2, 1)]
    HINTS_ELIM[16] = [(4, 0), (3, 0), (2, 0)]
    HINTS_ELIM[11] = [(4, 0)]
    HINTS_ELIM[12] = [(4, 3), (3, 2)]

# Cut fit points when the relative error in the error bar is > ERR_CUT
ERR_CUT = 0.20

# stringent tolerance for minimizer?  true = stringent
MINTOL = True
MINTOL = False

# rescale the fit function by factor RESCALE
RESCALE = 1e12
RESCALE = 1.0

# use fixed pion mass in ratio fits?
USE_FIXED_MASS = False
USE_FIXED_MASS = True

# EFF_MASS_METHOD 1: analytic for arg to acosh
# (good for when additive const = 0, but noiser than 3 and 4)
# EFF_MASS_METHOD 2: numeric solve system of three transcendental equations
# (bad for all cases; DO NOT USE.  It doesn't converge very often.)
# EFF_MASS_METHOD 3: one param fit
# EFF_MASS_METHOD 4: same as 2, but equations have one free parameter (
# traditional effective mass method),
# typically a fast version of 3
# (3 may have better different error properties, though)

EFF_MASS_METHOD = 4

# starting values for fit parameters
if EFF_MASS and EFF_MASS_METHOD != 2:
    START_PARAMS = [.5, .2]
else:
    if ADD_CONST:
        START_PARAMS = [0.0580294, -0.003, 0.13920]
    else:
        START_PARAMS = [6.28203895e6, 4.6978036e-01]

print("start params:", START_PARAMS)

START_PARAMS = [0.5] if SYS_ENERGY_GUESS is None and EFF_MASS else\
    START_PARAMS

# how many loop iterations until we start using random samples
MAX_ITER = 1000 if not ONLY_SMALL_FIT_RANGES else np.inf
MAX_ITER = 1900 if SYS_ENERGY_GUESS is not None and GEVP else MAX_ITER
# MAX_RESULTS is the max number of usable fit ranges to average over
# (useful for random fitting; the fitter will otherwise take a long time)
# set this to np.inf to turn off
MAX_RESULTS = np.inf
MAX_RESULTS = 1 if SYS_ENERGY_GUESS is not None and GEVP else MAX_RESULTS
MAX_RESULTS = 3

# modify the configs used and bin

# Uncorrelated fit? True or False

UNCORR = True
UNCORR = False

# pvalue minimum; we reject model if a pvalue less than this is found
PVALUE_MIN = 0.1

# Log off, vs. log on; in eff_mass method 3, calculate log at the end vs. not

LOG = False
LOG = True
#LOG = False if PIONRATIO else LOG

# Jackknife fit? (keep double for correctness, others not supported)

JACKKNIFE_FIT = 'FROZEN'
JACKKNIFE_FIT = 'SINGLE'
JACKKNIFE_FIT = 'DOUBLE'

# pickle, unpickle

PICKLE = 'clean'
PICKLE = 'unpickle'
PICKLE = 'pickle'
PICKLE = None

#### DISPLAY PARAMETERS
# no title given takes the current working directory as the title

# title prefix

# p_cm = 001, no need to modify
PSTR_TITLE = r"$\vec{p}_{CM}=$"+rf.ptostr(rf.procmom(MOMSTR))+", "

if GEVP:
    if SIGMA and ISOSPIN == 0:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, $\pi\pi, \sigma$, ' + PSTR_TITLE + ' '
    elif ISOSPIN == 2:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, I2, $\pi\pi$, ' + PSTR_TITLE + ' '
    elif ISOSPIN == 1:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, I1, $\pi\pi, \rho$, ' + \
            irr2tex(IRREP) + PSTR_TITLE + ' '
    else:
        TITLE_PREFIX = str(DIM)+r'x'+str(DIM)+\
            r' GEVP, $\pi\pi$, ' + PSTR_TITLE + ' '

else:
    if LATTICE_ENSEMBLE == '24c':
        TITLE_PREFIX = '24c '
    elif LATTICE_ENSEMBLE == '32c':
        TITLE_PREFIX = '32c '
if GEVP:
    TITLE_PREFIX = TITLE_PREFIX + 't-t0=' + T0[6:] + " "
if GEVP_DERIV:
    TITLE_PREFIX = TITLE_PREFIX + r'$\partial t$, ' + " "
if SUPERJACK_CUTOFF and not SLOPPYONLY:
    TITLE_PREFIX = TITLE_PREFIX + 'exact '
else:
    if LATTICE_ENSEMBLE == '24c':
        TITLE_PREFIX = TITLE_PREFIX + '(zmobius) '
    elif LATTICE_ENSEMBLE == '32c':
        TITLE_PREFIX = TITLE_PREFIX + '(sloppy) '
if MATRIX_SUBTRACTION and DELTA_E2_AROUND_THE_WORLD is not None and GEVP:
    TITLE_PREFIX = TITLE_PREFIX + 'matdt'+\
        str(DELTA_T_MATRIX_SUBTRACTION)+', '+\
        str(DELTA_T2_MATRIX_SUBTRACTION)+' '
elif MATRIX_SUBTRACTION and GEVP:
    TITLE_PREFIX = TITLE_PREFIX + 'matdt'+\
        str(DELTA_T_MATRIX_SUBTRACTION)+' '
elif True in ADD_CONST_VEC:
    TITLE_PREFIX = TITLE_PREFIX + 'eigdt1 '
if HALF != 'full':
    TITLE_PREFIX = TITLE_PREFIX + HALF + ' '

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
CALC_PHASE_SHIFT = False if not GEVP else CALC_PHASE_SHIFT

# phase shift error cut, absolute, in degrees.
# if the error is bigger than this, skip this fit range
# I=2 is not very noisy
PHASE_SHIFT_ERR_CUT = 20 if ISOSPIN == 2 else np.inf

# skip overfit results (where chi^2/dof (t^2/dof) < 1)
SKIP_OVERFIT = False
SKIP_OVERFIT = True

# skip fit range if parameter (energy) errors greater than 100%
# I=2 is not very noisy
SKIP_LARGE_ERRORS = False
SKIP_LARGE_ERRORS = True if ISOSPIN == 2 else SKIP_LARGE_ERRORS

# box plot (for effective mass tolerance display)?
# doesn't look good with large systematics (no plateau)
BOX_PLOT = True
BOX_PLOT = False if SYS_ENERGY_GUESS is not None and GEVP else BOX_PLOT
BOX_PLOT = False if len(START_PARAMS) != 1 else BOX_PLOT

# plot a legend?
PLOT_LEGEND = False
PLOT_LEGEND = True

# dispersive lines
PLOT_DISPERSIVE = True
PLOT_DISPERSIVE = False if not GEVP else True

# Decrease variance in GEVP (avoid eigenvalue misordering due to large noise)
# should be < 1
DECREASE_VAR = 1
DECREASE_VAR = 1e-4
DECREASE_VAR = 1 if not GEVP else DECREASE_VAR

# delete operators which plausibly give rise to negative eigenvalues
DELETE_NEGATIVE_OPERATORS = True
DELETE_NEGATIVE_OPERATORS = False
DELETE_NEGATIVE_OPERATORS = False if ISOSPIN != 1 else DELETE_NEGATIVE_OPERATORS

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
BINDS = [[0, 2] for _ in range(2*DIM+1)]
# try to set bounds for the systematic error
BINDS[1::2] = [[-1, 1] for _ in enumerate(BINDS[1::2])]
BINDS = [[None, None] for _ in range(len(START_PARAMS)*DIM+(
    1 if SYS_ENERGY_GUESS is not None and EFF_MASS else 0))]
BINDS = [[None, None]] if not BINDS else BINDS
BINDS = [tuple(bind) for bind in BINDS]
BINDS[-1] = (None, 100) if SYS_ENERGY_GUESS is not None and EFF_MASS else BINDS[-1]
BINDS = tuple(BINDS)
print("Bounds on fit parameters:", BINDS)


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

# chi^2 (t^2) minimizer method used by the scipy.optimize.minimize
# other internals will need to be edited if you change this
# it's probably not a good idea

METHOD = 'L-BFGS-B'
METHOD = 'Nelder-Mead'
METHOD = 'minuit'

# print correlation function, and sqrt(diag(cov)) and exit

PRINT_CORR = True
PRINT_CORR = False

# plot anything at all?

NO_PLOT = True
NO_PLOT = False

# -------BEGIN POSSIBLY OBSOLETE------#

# log form (take log of GEVP matrices)
LOGFORM = True
LOGFORM = False

# estimate systematic error with function in chi_sq.py (not working yet)
SYSTEMATIC_EST = True
SYSTEMATIC_EST = False

# use very late time slices in the GEVP.
# these may have very large error bars and be numerically less well behaved,
# so it's usually safer to start with this option turned off
USE_LATE_TIMES = True
USE_LATE_TIMES = False

REINFLATE_BEFORE_LOG = True
REINFLATE_BEFORE_LOG = False

# multiply both sides of the gevp matrix by norms

# NORMS = [[1.0/(16**6), 1.0/(16**3)], [1.0/(16**3), 1]]

OPERATOR_NORMS = [(1+0j) for i in range(DIM)]
if ISOSPIN == 1 and GEVP and DIM > 1:
    OPERATOR_NORMS[1] = complex(0+1j)
if ISOSPIN == 0 and GEVP and 'avg' in IRREP and DIM > 1:
    OPERATOR_NORMS[0] = 1e-5
    OPERATOR_NORMS[1] = 1e-2
    if DIM == 3:
        OPERATOR_NORMS[2] = 1e-5
print("New Operator Norms:", OPERATOR_NORMS)
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

# average relative error on the parameter errors to attempt to achieve
# if achieved, exit the fit loop
FITSTOP = 0.0000000000001
# If set to True, speed up the fit loop by looking at models
# which resemble non-interacting (dispersive) energies first
# this biases the results, so turn off if doing a final fit
BIASED_SPEEDUP = True
BIASED_SPEEDUP = False

# ASSISTED_FIT = True
ASSISTED_FIT = False

# boundary scale for zero'ing out a failing inverse Hessian
# (neg diagaonal entrie(s)).  Below 1/(CUTOFF*SCALE), an entry is set to 0
CUTOFF = 10**(7)

# additive constant subtracted by hand from exact effective mass functions
# questionable, since this is an extra, badly optimized, fit parameter
C = 1.935*SCALE*0 if (
    ADD_CONST and EFF_MASS_METHOD == 1 and EFF_MASS) else 0

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
if not GEVP_DEBUG and any(ADD_CONST_VEC):
    assert MATRIX_SUBTRACTION or not GEVP, \
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

# update module variables
latfit.fit_funcs.USE_FIXED_MASS = USE_FIXED_MASS
latfit.fit_funcs.LOG = LOG
latfit.fit_funcs.C = C
latfit.fit_funcs.TSTEP = TSTEP if not GEVP or GEVP_DEBUG else\
    DELTA_T_MATRIX_SUBTRACTION
latfit.fit_funcs.TSTEP2 = TSTEP if not GEVP or GEVP_DEBUG else\
    DELTA_T2_MATRIX_SUBTRACTION
latfit.fit_funcs.TSTEP2 = 0 if DELTA_E2_AROUND_THE_WORLD is None else\
    latfit.fit_funcs.TSTEP2
latfit.fit_funcs.PION_MASS = misc.massfunc()
latfit.fit_funcs.PIONRATIO = False
latfit.fit_funcs.LT = LT
latfit.fit_funcs.GEVP = GEVP
latfit.fit_funcs.DELTAT = -1 if GEVP_DERIV else int(T0[6:])
latfit.fit_funcs.DELTAT = -1 if not GEVP else latfit.fit_funcs.DELTAT

# selects fit func class and updates with module vars
FITS.select_and_update(ADD_CONST)

# END DO NOT MODIFY

# -------END POSSIBLY OBSOLETE------#

ORIGL = len(START_PARAMS)

GEVP_DIRS = gevp_dirs(ISOSPIN, MOMSTR, IRREP, DIM, SIGMA)
MULT = len(GEVP_DIRS) if GEVP else 1


# perform check
fitfunc.check_start_params_len(EFF_MASS, EFF_MASS_METHOD, ORIGL,
                               MATRIX_SUBTRACTION,
                               DELTA_E2_AROUND_THE_WORLD)
# get initial blank fit function
PREFIT_FUNC = fitfunc.prelimselect()

if EFF_MASS:
    if EFF_MASS_METHOD == 1 or EFF_MASS_METHOD == 2 or EFF_MASS_METHOD == 4:
        # if no systematic, fit to const
        PREFIT_FUNC = fitfunc.constfit(len(START_PARAMS), RESCALE)
        if SYS_ENERGY_GUESS is not None:
            # append to start params the systematic parameter guess
            START_PARAMS = fitfunc.mod_start_params(START_PARAMS,
                                                    SYS_ENERGY_GUESS)
            if not (len(START_PARAMS)-1) % 2 and\
               DELTA_E2_AROUND_THE_WORLD is None:
                PREFIT_FUNC = fitfunc.const_plus_exp(START_PARAMS)
            elif not (len(START_PARAMS)-1) % 3:
                fitfunc.three_asserts(GEVP, MATRIX_SUBTRACTION, NOATWSUB)
                PREFIT_FUNC = fitfunc.atwfit(DELTA_E2_AROUND_THE_WORLD,
                                             DELTA_E_AROUND_THE_WORLD,
                                             START_PARAMS, DIM, LT)
            elif not (len(START_PARAMS)-1) % 4:
                fitfunc.three_asserts(GEVP, MATRIX_SUBTRACTION, NOATWSUB)
                PREFIT_FUNC = fitfunc.atwfit_second_order(
                    START_PARAMS, DIM, LT, DELTA_E_AROUND_THE_WORLD, MINE2)
            else:
                fitfunc.fit_func_die()
    elif EFF_MASS_METHOD == 3:
        PREFIT_FUNC = fitfunc.eff_mass_3_func((ORIGL, MULT), RESCALE, FITS,
                                              (ADD_CONST_VEC, LT_VEC))
    else:
        fitfunc.fit_func_die()
else:
    if GEVP:
        PREFIT_FUNC = fitfunc.expfit_gevp((ORIGL, MULT), FIT,
                                          RESCALE, FITS,
                                          (ADD_CONST_VEC, LT_VEC))
    else:
        PREFIT_FUNC = fitfunc.expfit(FIT, ORIGL, ADD_CONST, RESCALE, FITS)

# we've now got the form of the fit function, but it needs some
# code related modifications
PREFIT_FUNC = copy(PREFIT_FUNC) if FIT else PREFIT_FUNC
PREFIT_FUNC = fitfunc.pencil_mod(PREFIT_FUNC, FIT, NUM_PENCILS,
                                 RESCALE, START_PARAMS)

def fit_func(ctime, trial_params): # lower case hack
    """Function to fit; final form after config.py"""
    return PREFIT_FUNC(ctime, trial_params)

# make statements (asserts)
sands.gevp_statements(GEVP_DIRS, GEVP, DIM, MULT, (LT_VEC, ADD_CONST_VEC))
START_PARAMS = sands.start_params_pencils(START_PARAMS, ORIGL,
                                          NUM_PENCILS, MULT,
                                          SYS_ENERGY_GUESS)
sands.rescale_and_atw_statements(EFF_MASS, EFF_MASS_METHOD, RESCALE,
                                 DELTA_E_AROUND_THE_WORLD,
                                 DELTA_E2_AROUND_THE_WORLD)
sands.asserts_one(EFF_MASS_METHOD, MATRIX_SUBTRACTION,
                  JACKKNIFE_FIT, NUM_PENCILS, JACKKNIFE)
sands.asserts_two(IRREP, FIT_SPACING_CORRECTION, ISOSPIN, PIONRATIO)
sands.asserts_three(MOMSTR, DELTA_E_AROUND_THE_WORLD,
                    DELTA_E2_AROUND_THE_WORLD, GEVP, GEVP_DERIV)
sands.bin_time_statements(BINNUM, USE_LATE_TIMES, T0, BIASED_SPEEDUP)
sands.bin_statements(BINNUM, ELIM_JKCONF_LIST, HALF,
                     ONLY_SMALL_FIT_RANGES, RANGE_LENGTH_MIN)
DELTA_E2_AROUND_THE_WORLD = sands.delta_e2_mod(SYSTEMATIC_EST, PIONRATIO,
                                               DELTA_E2_AROUND_THE_WORLD,
                                               DELTA_E_AROUND_THE_WORLD)
sands.matsub_statements(MATRIX_SUBTRACTION, IRREP, ISOSPIN, GEVP, NOATWSUB)
sands.superjackknife_statements(check_ids()[-2], SUPERJACK_CUTOFF)
sands.deprecated(USE_LATE_TIMES, LOGFORM)
