"""Library of fit functions to use"""

from math import log
from numbers import Number
import numpy as np
from numpy import exp
from sympy import exp as exps
from latfit.analysis.test_arg import zero_p, testsol

class FitFunctions:
    """Default fit functions."""

    def __init__(self):
        """Define functions from flist."""
        self.flist = ['fit_func_exp', 'ratio',  'acosh_ratio',
                      'fit_func_1p', 'fit_func_sym', 'fit_func_exp_gevp']
        self.f = {}
        self._select = {}
        self._fitfunc = FitFunc()
        self._fitfuncadd = FitFuncAdd()
        self._update_f()

    def _update_f(self):
        for func in self.flist:
            self.f[func] = [getattr(self._fitfunc, func),
                            getattr(self._fitfuncadd, func)]

    def select(self, add_const, log, lt, c, tstep):
        """Select which set of functions to use"""
        index = 1 if add_const else 0
        LOG = log
        LT = lt
        C = c
        TSTEP = tstep
        self._fitfuncadd.update(log, lt, c, tstep)
        self._fitfunc.update(log, lt, c, tstep)
        self._update_f()
        for func in self.f:
            self._select[func] = self.f[func][index]

    def __getitem__(self, key):
        """Get the function from the select set"""
        return self._select[key]
        
LOG = False
LOG = True
LT = 1
C = 0
TSTEP = 1

class FitFuncAdd:
    """Exponential fit functions with additive constant"""

    def __init__(self):
        self._log = LOG
        self._lt = LT
        self._c = C
        self._tstep = TSTEP

    def update(self, log, lt, c, tstep):
        self._log = log
        self._lt = lt
        self._c = c
        self._tstep = tstep

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) + exp(
            -trial_params[1]*(self._lt-ctime))) + trial_params[2]

    def ratio(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (and take log)"""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1], corrs[2], times)
        sol = (corrs[1]-corrs[0])/(corrs[2]-corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        sol = log(sol) if self._log else sol
        return sol

    def acosh_ratio(self, corrs, times=None, nocheck=False):
        """Process data into effective mass ratio,
        for an exact call to acosh."""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1]-self._c, times[1:])
        sol = (corrs[0]-corrs[1]+corrs[2]-corrs[3])/2.0/(corrs[1]-corrs[2])
        if not nocheck:
            testsol(sol, corrs, times)
        return sol

    def fit_func_sym(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv))
        for EFF_MASS_METHOD = 2
        """
        return trial_params[0]*(
            exps(-trial_params[1]*ctime) +
            exps(-trial_params[1]*(self._lt-ctime)))+trial_params[2]

    def fit_func_exp_gevp(self, ctime, trial_params, lt=None):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        lt = self._lt if lt is None else lt
        return ((exp(-trial_params[0]*ctime) +
                exp(-trial_params[1]*(lt-ctime))) + trial_params[2])/(
                    (exp(-trial_params[0]*(TRHS)) +
                    exp(-trial_params[1]*(lt-(TRHS)))) + trial_params[2])

    def fit_func_1p(self, ctime, trial_params, lt=None):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lt = self._lt if lt is None else lt
        corrs = [exp(-trial_params[0]*(ctime+i*self._tstep)) +
                exp(-trial_params[0]*(lt-(ctime+i*self._tstep)))
                for i in range(3)]
        return self.ratio(corrs, ctime, nocheck=True)

class FitFunc:
    """Exponential fit functions without additive constant"""

    def __init__(self):
        self._log = LOG
        self._lt = LT
        self._c = C
        self._tstep = TSTEP

    def update(self, log, lt, c, tstep):
        self._log = log
        self._lt = lt
        self._c = c
        self._tstep = tstep

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) +
                                exp(-trial_params[1]*(self._lt-ctime)))

    def ratio(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio
        (and take log), no additive constant
        """
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1], times[1])
        sol = (corrs[0])/(corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        sol = log(sol) if self._log else sol
        return sol

    def acosh_ratio(self, corrs, times=None, nocheck=False):
        """Process data into effective mass ratio,
        for an exact call to acosh (no additive constant)."""
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1]-self._c, times[1])
        sol = (corrs[0]+corrs[2]-2*self._c)/2/(corrs[1]-self._c)
        if not nocheck:
            testsol(sol, corrs, times)
        return sol

    def fit_func_sym(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv))
        for EFF_MASS_METHOD = 2
        """
        return trial_params[0]*(
            exps(-trial_params[1]*ctime) +
            exps(-trial_params[1]*(self._lt-ctime)))

    def fit_func_exp_gevp(self, ctime, trial_params, lt=None):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        lt = self._lt if lt is None else lt
        return (exp(-trial_params[0]*ctime) +
                exp(-trial_params[1]*(lt-ctime)))/(
                    (exp(-trial_params[0]*(TRHS)) +
                     exp(-trial_params[1]*(lt-(TRHS)))))

    def fit_func_1p(self, ctime, trial_params, lt=None):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lt = self._lt if lt is None else lt
        corrs = [exp(-trial_params[0]*(ctime+i*self._tstep)) +
                exp(-trial_params[0]*(lt-(ctime+i*self._tstep)))
                for i in range(2)]
        return self.ratio(corrs, ctime, nocheck=True)

