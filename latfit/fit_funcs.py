"""Library of fit functions to use"""

import sys
from collections import namedtuple
from math import log, cosh, sinh, tanh
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
                      'fit_func_1p', 'fit_func_sym', 'fit_func_exp_gevp',
                      'pion_ratio', 'fit_func_1p_exp', 'fit_func_1p_pionratio',
                      'ratio_pionratio', 'ratio_exp']
        self.f = {}
        self._select = {}
        self._fitfunc = FitFunc()
        self._fitfuncadd = FitFuncAdd()
        self._update_f()

    def _update_f(self):
        for func in self.flist:
            self.f[func] = [getattr(self._fitfunc, func),
                            getattr(self._fitfuncadd, func)]

    def select(self, up):
        """Select which set of functions to use"""
        index = 1 if up.add_const else 0
        LOG = up.log
        LT = up.lt
        C = up.c
        TSTEP = up.tstep
        PION_MASS = up.pionmass
        PIONRATIO = up.pionratio
        self._fitfuncadd.update(up)
        self._fitfunc.update(up)
        self._update_f()
        for func in self.f:
            self._select[func] = self.f[func][index]

    def __getitem__(self, key):
        """Get the function from the select set"""
        return self._select[key]

    def test(self):
        if USE_FIXED_MASS:
            print("Using fixed pion mass in pion ratio fits.")
        else:
            print("Not using fixed pion mass in pion ratio fits.")
        
LOG = False
LOG = True
LT = 1
C = 0
TSTEP = 1
PION_MASS = 0
PIONRATIO = False
USE_FIXED_MASS = True


class FitFuncAdd:
    """Exponential fit functions with additive constant"""

    def __init__(self):
        self._log = LOG
        self._lt = LT
        self._c = C
        self._tstep = TSTEP
        self._pionmass = PION_MASS
        self._pionratio = PIONRATIO

    def update(self, up):
        self._log = up.log
        self._lt = up.lt
        self._c = up.c
        self._tstep = up.tstep
        self._pionmass = up.pionmass
        self._pionratio = up.pionratio 

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) + exp(
            -trial_params[1]*(self._lt-ctime))) + trial_params[2]

    def ratio(self, corrs, times=None, nocheck=False):
        ret = self.ratio_pionratio(corrs, times, nocheck) if self._pionratio else self.ratio_exp(corrs, times, nocheck)
        return ret

    def ratio_exp(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (and take log)"""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[3], corrs[2], times)
        sol = (corrs[1]-corrs[0])/(corrs[3]-corrs[2])
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

    def fit_func_1p(self, ctime, trial_params, lt=None, tstep=None):
        """Meta function for effective mass."""
        ret = self.fit_func_1p_pionratio(ctime, trial_params, lt) if self._pionratio else self.fit_func_1p_exp(ctime, trial_params, lt, tstep)
        return ret

    def fit_func_1p_exp(self, ctime, trial_params, lt=None, tstep=None):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lt = self._lt if lt is None else lt
        tstep = self._tstep if tstep is None else tstep
        corrs_num = [exp(-trial_params[0]*(ctime+i*tstep)) +
                 exp(-trial_params[0]*(lt-(ctime+i*tstep)))
                 for i in range(2)]
        corrs_denom = [exp(-trial_params[0]*(ctime+1+i*tstep)) +
                 exp(-trial_params[0]*(lt-(ctime+1+i*tstep)))
                 for i in range(2)]
        corrs = [*corrs_num, *corrs_denom]
        return self.ratio_exp(corrs, ctime, nocheck=True)

    def pion_ratio(self, ctime, trial_params, lt=None):
        """Include pions in the denominator of eff mass ratio."""
        tp = ctime+1/2-self._lt/2.0
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[2]
        return trial_params[0]*(cosh(tp*trial_params[1])+sinh(tp*trial_params[1])/tanh(2*tp*pionmass))

    def ratio_pionratio(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (and take log)"""
        times = [-99999, -99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1], corrs[2], times)
        sol = (corrs[0])/(corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        return corrs[0]

    def fit_func_1p_pionratio(self, ctime, trial_params, lt=None, tstep=None):
        lt = self._lt if lt is None else lt
        tstep = self._tstep if tstep is None else tstep
        tp = [ctime+i*tstep+1/2-lt/2.0 for i in range(3)]
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[1]
        corrs = [trial_params[0]*(sinh(tp[i]*trial_params[1])+cosh(tp[i]*trial_params[1])/tanh(2*tp[i]*pionmass)) for i in range(3)]
        #return self.ratio_pionratio(corrs, ctime, nocheck=True)
        return corrs[0]

class FitFunc:
    """Exponential fit functions without additive constant"""

    def __init__(self):
        self._log = LOG
        self._lt = LT
        self._c = C
        self._tstep = TSTEP
        self._pionmass = PION_MASS
        self._pionratio = PIONRATIO

    def update(self, up):
        self._log = up.log
        self._lt = up.lt
        self._c = up.c
        self._tstep = up.tstep
        self._pionmass = up.pionmass
        self._pionratio = up.pionratio 

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) +
                                exp(-trial_params[1]*(self._lt-ctime)))

    def ratio(self, corrs, times=None, nocheck=False):
        ret = self.ratio_pionratio(corrs, times, nocheck) if self._pionratio else self.ratio_exp(corrs, times, nocheck)
        return ret

    def ratio_exp(self, corrs, times=None, nocheck=False):
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

    def fit_func_1p(self, ctime, trial_params, lt=None, tstep=None):
        """Meta function for effective mass."""
        ret = self.fit_func_1p_pionratio(ctime, trial_params, lt) if self._pionratio else self.fit_func_1p_exp(ctime, trial_params, lt, tstep)
        return ret

    def fit_func_1p_exp(self, ctime, trial_params, lt=None, tstep=None):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        tstep = self._tstep if tstep is None else tstep
        lt = self._lt if lt is None else lt
        corrs = [exp(-trial_params[0]*(ctime+i*tstep)) +
                exp(-trial_params[0]*(lt-(ctime+i*tstep)))
                for i in range(2)]
        return self.ratio_exp(corrs, ctime, nocheck=True)

    def fit_func_1p_pionratio(self, ctime, trial_params, lt=None, tstep=None):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lt = self._lt if lt is None else lt
        tstep = self._tstep if tstep is None else tstep
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[1]
        tp = [ctime+i*tstep+1/2-lt/2.0 for i in range(2)]
        corrs = [trial_params[0]*(sinh((tp[i]-1/2)*trial_params[1]-1/2*pionmass)+cosh((tp[i]-1/2)*trial_params[1]-1/2*pionmass))/tanh(2*tp[i]*pionmass) for i in range(2)]
        #return self.ratio_pionratio(corrs, ctime, nocheck=True)
        return corrs[0]

    def ratio_pionratio(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (pion ratio),
        no additive constant
        """
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        assert USE_FIXED_MASS, "Only fixed pion mass supported in eff mass pion ratio fits."
        if nocheck:
            np.seterr(invalid='ignore')
        else:
            np.seterr(invalid='raise')
            zero_p(corrs[1], times[1])
        sol = (corrs[0])/(corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        sol = log(sol) if self._log else sol
        return corrs[0]

    def pion_ratio(self, ctime, trial_params, lt=None):
        """Include pions in the denominator of eff mass ratio."""
        lt = self._lt if lt is None else lt
        tp = ctime+1/2-lt/2.0
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[2]
        return trial_params[0]*(sinh(tp*trial_params[1])+cosh(tp*trial_params[1])/tanh(2*tp*pionmass))
