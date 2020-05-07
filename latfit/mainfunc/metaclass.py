"""Class containing meta data and related functions"""
import sys
from itertools import combinations, chain, product
import ast
import numpy as np
import h5py
from recordtype import recordtype

from latfit.extract.errcheck.trials_err import trials_err
from latfit.extract.errcheck.xlim_err import xlim_err
from latfit.extract.errcheck.xlim_err import fitrange_err
from latfit.extract.errcheck.xstep_err import xstep_err
from latfit.config import GEVP, STYPE, MAX_ITER
from latfit.config import NOLOOP, MULT, FIT, VERBOSE
from latfit.config import MATRIX_SUBTRACTION
from latfit.config import RANGE_LENGTH_MIN
from latfit.config import ONLY_SMALL_FIT_RANGES
from latfit.config import DELTA_E2_AROUND_THE_WORLD
from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT
from latfit.analysis.errorcodes import DOFNonPos
import latfit.config
import latfit.makemin.mkmin as mkmin
from latfit.procargs import procargs

EXCL_ORIG = np.copy(EXCL_ORIG_IMPORT)

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def filter_sparse(sampler, fitwindow, xstep=1):
    """Find the items in the power set which do not generate
    arithmetic sequences in the fitwindow powerset (sampler)
    """
    frange = np.arange(fitwindow[0], fitwindow[1]+xstep, xstep)
    retsampler = []
    for excl in sampler:
        fdel = tuple(filter(lambda a, sk=excl: a not in sk, frange))
        if len(fdel) < RANGE_LENGTH_MIN and not ONLY_SMALL_FIT_RANGES:
            continue
        if len(fdel) >= RANGE_LENGTH_MIN and ONLY_SMALL_FIT_RANGES:
            continue
        # start = fdel[0]
        incr = xstep if len(fdel) < 2 else fdel[1]-fdel[0]
        skip = False
        for i, timet in enumerate(fdel):
            if i == 0:
                continue
            if fdel[i-1] + incr != timet:
                skip = True
        if skip:
            continue
        retsampler.append(excl)
    retsampler = tuple(retsampler)
    return retsampler

@PROFILE
def powerset(iterable):
    """powerset([1, 2, 3]) -->
    () (1, ) (2, ) (3, ) (1, 2) (1, 3) (2, 3) (1, 2, 3)"""
    siter = list(iterable)
    return chain.from_iterable(combinations(siter,
                                            r) for r in range(len(siter)+1))

@PROFILE
def update_num_configs(num_configs=None, input_f=None):
    """Update the number of configs in the case that FIT is False.
    """
    assert None, "not supported"
    num_configs = -1 if num_configs is None else num_configs
    if not FIT and STYPE == 'hdf5' and num_configs == -1:
        infile = input_f if input_f is not None else\
            latfit.config.GEVP_DIRS[0][0]
        fn1 = h5py.File(infile, 'r')
        for i in fn1:
            if GEVP:
                for j in fn1[i]:
                    num_configs = np.array(fn1[i + '/' + j]).shape[0]
                    break
            else:
                num_configs = np.array(fn1[i]).shape[0]
            break
    print("updating title number of configs to:", num_configs)
    latfit.finalout.mkplot.NUM_CONFIGS = num_configs


class FitRangeMetaData:
    """Meta data about fit range loop"""
    @PROFILE
    def __init__(self):
        """Define meta data container."""
        self.__skiploop = False
        self.lenprod = 0
        self.lenfit = 0
        #self.xmin = 0
        #self.options.xmax = np.inf
        #self.options.xstep = 1
        self.fitwindow = []
        self.random_fit = True
        self.input_f = None
        self.options = recordtype(
            'ops', 'xmin xmax xstep trials fitmin fitmax procs')

    def incr_xmin(self, problemx=None, inx=False):
        """Increment xmin by one*xstep"""
        assert self.fitwindow
        if VERBOSE:
            print("increasing xmin by one*xstep")
        if problemx is None:
            #self.options.xmin += self.options.xstep
            self.fitwindow = (
                self.fitwindow[0]+self.options.xstep, self.fitwindow[1])
            if inx:
                self.options.xmin += self.options.xstep
        else:
            #self.options.xmin = problemx + self.options.xstep
            if problemx >= self.fitwindow[0]:
                self.fitwindow = (
                    problemx + self.options.xstep, self.fitwindow[1])
            if inx:
                self.options.xmin = problemx + self.options.xstep
        try:
            assert self.fitwindow[0] < self.fitwindow[1]
            assert self.options.xmin < self.options.xmax
            assert self.options.xmin <= self.fitwindow[0]
        except AssertionError:
            if VERBOSE:
                print("current fit window too small:",
                      self.fitwindow)
            raise DOFNonPos
        self.pr_fit_window()

    def decr_xmax(self, problemx=None, dex=False):
        """Decrement xmax by one*xstep"""
        if VERBOSE:
            print("decreasing xmax by one*xstep")
        if problemx is None:
            #self.options.xmax -= self.options.xstep
            self.fitwindow = (self.fitwindow[0],
                              self.fitwindow[1]-self.options.xstep)
            if dex:
                self.options.xmax -= self.options.xstep
        else:
            #self.options.xmax = problemx - self.options.xstep
            if problemx <= self.fitwindow[1]:
                self.fitwindow = (
                    self.fitwindow[0], problemx-self.options.xstep)
            if dex:
                self.options.xmax = problemx - self.options.xstep
        try:
            assert self.fitwindow[0] < self.fitwindow[1]
            assert self.options.xmin < self.options.xmax
            assert self.options.xmax >= self.fitwindow[1]
        except AssertionError:
            if VERBOSE:
                print("current fit window too small:",
                      self.fitwindow)
            raise DOFNonPos
        self.pr_fit_window()

    @PROFILE
    def skip_loop(self):
        """Set the loop condition"""
        self.__skiploop = self.lenprod <= 1
        self.__skiploop = True if NOLOOP else self.__skiploop
        if not self.random_fit and not self.__skiploop:
            for excl in EXCL_ORIG:
                if len(excl) > 1:
                    #assert None
                    #self.skiploop = True
                    self.__skiploop = False
        return self.__skiploop

    @PROFILE
    def generate_combinations(self):
        """Generate all possible fit ranges"""
        posexcl = powerset(self.actual_range())
        sampler = filter_sparse(posexcl, self.fitwindow, self.options.xstep)
        sampler = [list(EXCL_ORIG)] if NOLOOP else sampler
        posexcl = [sampler for i in range(len(latfit.config.FIT_EXCL))]
        prod = product(*posexcl)
        return prod, sampler

    @PROFILE
    def xmin_mat_sub(self):
        """Shift xmin to be later in time in the case of
        around the world subtraction of previous time slices"""
        assert self.fitwindow
        delta = latfit.config.DELTA_T_MATRIX_SUBTRACTION
        if DELTA_E2_AROUND_THE_WORLD is not None:
            delta += latfit.config.DELTA_T2_MATRIX_SUBTRACTION
        delta = 0 if not MATRIX_SUBTRACTION else delta
        if GEVP:
            try:
                toadd = latfit.config.T0[6:]
                toadd = int(toadd)
                xmin_req = delta + toadd
            except TypeError:
                print(delta, latfit.config.T0)
                raise
            #print("xmin_req", xmin_req)
            if self.options.xmin <= xmin_req:
                #print("increasing xmin due to xmin_req =", xmin_req)
                self.incr_xmin(problemx=xmin_req, inx=True)

    @PROFILE
    def fit_coord(self):
        """Get xcoord to plot fit function."""
        return self.actual_range()


    @PROFILE
    def length_fit(self, prod, sampler):
        """Get length of fit window data"""
        self.lenfit = len(self.actual_range())
        assert self.lenfit > 0 or not FIT,\
            "length of fit range not > 0"
        self.lenprod = len(sampler)**(MULT)
        if NOLOOP:
            assert self.lenprod == 1,\
                "Number of fit ranges is too large:"+str(self.lenprod)
        latfit.config.MINTOL = True if self.lenprod == 0 else\
            latfit.config.MINTOL
        #latfit.config.BOOTSTRAP = True if self.lenprod == 0 else\
        #    latfit.config.BOOTSTRAP
        self.random_fit = True
        # fit range is small, use brute force
        if self.lenprod < MAX_ITER and not NOLOOP:
            self.random_fit = False
            mkmin.KICK = True
            prod = list(prod)
            prod = [str(i) for i in prod]
            prod = sorted(prod)
            prod = [ast.literal_eval(i) for i in prod]
            assert len(prod) == self.lenprod,\
                "powerset length mismatch"+\
                " vs. expected length."
        return prod

    def window_str(self):
        """Get a file string for the fit window
        the int float convention is for historical reasons.
        """
        ret = "_"+str(int(self.fitwindow[0]))+"_"+str(
            float(self.fitwindow[1]))
        return ret

    @PROFILE
    def actual_range(self):
        """Return the actual range spanned by the fit window"""
        try:
            ret = np.arange(self.fitwindow[0],
                            self.fitwindow[1]+self.options.xstep,
                            self.options.xstep)
        except TypeError:
            print(self.fitwindow)
            print(self.options.xstep)
            raise
        ret = tuple(ret)
        latfit.analysis.result_min.WINDOW = ret
        return ret

    def pr_fit_window(self):
        """Print the current fit window"""
        win = self.actual_range()
        assert self.fitwindow[0] == win[0]
        assert self.fitwindow[1] == win[-1]
        assert len(self.fitwindow) == 2, str(self.fitwindow)
        if VERBOSE:
            print("current fit window = ", self.fitwindow)
            print("current xmin, xmax = ",
                  self.options.xmin, self.options.xmax)

    @PROFILE
    def setup(self, plotdata):
        """Setup the fitter at the beginning of the run"""
        self.input_f, self.options = procargs(sys.argv[1:])
        self.options.fitmin = str2float(self.options.fitmin)
        self.options.fitmax = str2float(self.options.fitmax)
        self.options.procs = int(str2float(self.options.procs))
        self.options.xmin, self.options.xmax = xlim_err(self.options.xmin,
                                                        self.options.xmax)
        self.options.xstep = xstep_err(self.options.xstep, self.input_f)
        if not isinstance(self.options.fitmin, float) and not isinstance(
                self.options.fitmin, int):
            self.options.fitmin = self.options.xmin
        if not isinstance(self.options.fitmax, float) and not isinstance(
                self.options.fitmax, int):
            self.options.fitmax = self.options.xmax
        self.fitwindow = fitrange_err(self.options, self.options.fitmin,
                                      self.options.fitmax)
        assert isinstance(self.fitwindow[0], (float, int)), self.fitwindow
        self.xmin_mat_sub()
        self.actual_range()
        latfit.config.TSTEP = self.options.xstep
        plotdata.fitcoord = self.fit_coord()
        trials = trials_err(self.options.trials)
        if STYPE != 'hdf5':
            assert None, "not supported"
            update_num_configs(input_f=(
                self.input_f if not GEVP else None))

        return trials, plotdata, str(self.input_f)

def str2float(sstr):
    """Convert string to float"""
    if isinstance(sstr, float):
        ret = sstr
    elif isinstance(sstr, str):
        ret = float(sstr)
    else:
        ret = None
    return ret
