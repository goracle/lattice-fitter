"""Contains result min class used in jackknife fit loop; stores results
 of fit"""
from collections import namedtuple
import numpy as np
from scipy import stats
from latfit.config import START_PARAMS, UNCORR
from latfit.config import GEVP
from latfit.analysis.errorcodes import DOFNonPos
import latfit.config

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

WINDOW = []

class Param:
    """Storage for the average param, the array of the param,
    and the error on the param
    """
    def __init__(self):
        self.arr = np.array([])
        self.err = np.nan
        self.val = np.nan

    def zero(self, num_configs=None):
        """Zero out the array, value, and error"""
        self.val = 0
        self.err = 0
        if num_configs is not None:
            self.arr = np.zeros(num_configs)

    def mean(self, axis=0):
        """Take average of array"""
        #return em.acmean(self.arr, axis=axis)
        return np.mean(self.arr, axis=axis)

class ResultMin:
    """Store fit results for an individual fit range in this class"""
    def __init__(self, params, coords):
        self.energy = Param()
        self.systematics = Param()
        self.pvalue = Param()
        self.chisq = Param()
        self.phase_shift = Param()
        self.scattering_length = Param()
        self.misc = namedtuple(
            'misc', ['error_bars', 'dof', 'status', 'num_configs'])
        self.misc.error_bars = None
        self.misc.dof = None
        self.misc.num_configs = None
        self.misc.status = None
        self.compute_dof(params, coords)

    @PROFILE
    def alloc_sys_arr(self, params):
        """alloc array for systematics"""
        syslength = len(START_PARAMS)-params.dimops*(1 if GEVP else 0)
        syslength = max(1, syslength)
        self.systematics.arr = np.zeros((params.num_configs, syslength))
        self.misc.num_configs = params.num_configs
        # storage for fit by fit chi^2 (t^2)
        self.chisq.arr = np.zeros(params.num_configs)

    @PROFILE
    def funpvalue(self, chisq):
        """Give pvalue from Hotelling t^2 stastistic
        (often referred to incorrectly as chi^2;
        is actually a sort of correlated chi^2)
        """
        ret = None
        correction = (self.misc.num_configs-self.misc.dof)/(
            self.misc.num_configs-1)
        correction /= self.misc.dof
        correction = 1 if UNCORR else correction
        cor = correction
        if self.misc.dof is not None:
            ret = stats.f.sf(chisq*cor, self.misc.dof,
                             self.misc.num_configs-self.misc.dof)
        return ret

    @PROFILE
    def compute_dof(self, params, coords):
        """Correct the degrees of freedom based on the chosen fit range"""
        # compute degrees of freedom
        self.misc.dof = len(coords)*params.dimops-len(START_PARAMS)
        for i in coords[:, 0]:
            for j in latfit.config.FIT_EXCL:
                if i in j and i in WINDOW:
                    self.misc.dof -= 1
        if self.misc.dof < 1:
            print("dof < 1. dof =", self.misc.dof)
            print("fit window:", WINDOW)
            print("excl:", latfit.config.FIT_EXCL)
            raise DOFNonPos(dof=self.misc.dof)
