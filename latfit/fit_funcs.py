"""Library of fit functions to use"""
import sys
from math import log, cosh, sinh, tanh
from numbers import Number
import numpy as np
from numpy import exp
from sympy import exp as exps
from latfit.analysis.test_arg import zero_p, testsol
# from latfit.config import TRHS

class FitFunctions:
    """Default fit functions."""

    def __init__(self):
        """Define functions from flist."""
        self.flist = ['fit_func_exp', 'ratio', 'acosh_ratio',
                      'fit_func_1p', 'fit_func_sym', 'fit_func_exp_gevp',
                      'pion_ratio', 'fit_func_1p_exp', 'fit_func_1p_pionratio',
                      'ratio_pionratio', 'ratio_exp']
        self.fid = {}
        self._select = {}
        self._fitfunc = FitFunc()
        self._fitfuncadd = FitFuncAdd()
        self._update_f()

    def _update_f(self):
        for func in self.flist:
            self.f[func] = [getattr(self._fitfunc, func),
                            getattr(self._fitfuncadd, func)]

    def use(self, func):
        """Use a specific function"""
        func = str(func)
        return self._select[func]

    def select(self, upd):
        """Select which set of functions to use"""
        index = 1 if upd.add_const else 0
        global LOG, LT, C, TSTEP, TSTEP2, PION_MASS, PIONRATIO
        global DELTAT
        LOG = upd.log
        LT = upd.lent
        C = upd.c
        TSTEP = upd.tstep
        TSTEP2 = upd.tstep2
        PION_MASS = upd.pionmass
        PIONRATIO = upd.pionratio
        DELTAT = upd.deltat
        self._fitfuncadd.update(upd)
        self._fitfunc.update(upd)
        self._update_f()
        for func in self.f:
            self._select[func] = self.f[func][index]

    def __getitem__(self, key):
        """Get the function from the select set"""
        return self._select[key]

    @staticmethod
    def test():
        """Test if we are using a fixed mass for pion ratio"""
        if USE_FIXED_MASS:
            print("Using fixed pion mass in pion ratio fits.")
        else:
            print("Not using fixed pion mass in pion ratio fits.")

LOG = False
LOG = True
LT = 1
C = 0
TSTEP = 1
TSTEP2 = 1
PION_MASS = 0
PIONRATIO = False
USE_FIXED_MASS = True
TRHS = None
GEVP = False
DELTAT = 1

def takelog(sol):
    """more permissive form of log"""
    try:
        sol = log(sol)
    except ValueError:
        if not sol:
            sol = -1*np.inf
        else:
            print("Math domain error in taking log of:")
            print(sol)
            sys.exit(1)
    return sol

class FitFuncAdd:
    """Exponential fit functions with additive constant"""

    def __init__(self):
        """Init class params"""
        self._log = LOG
        self._lent = LT
        self._c = C
        self._tstep = TSTEP
        self._tstep2 = TSTEP2
        self._deltat = DELTAT
        self._pionmass = PION_MASS
        self._pionratio = PIONRATIO
        self._gevp = GEVP

    def update(self, upd):
        """Update class params"""
        self._log = upd.log
        self._lent = upd.lent
        self._c = upd.c
        self._tstep = upd.tstep
        self._tstep2 = upd.tstep2
        self._pionmass = upd.pionmass
        self._deltat = upd.deltat
        self._pionratio = upd.pionratio
        self._gevp = upd.gevp

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) + exp(
            -trial_params[1]*(self._lent-ctime))) + trial_params[2]

    def ratio(self, corrs, times=None, nocheck=False):
        """Meta function, find ratio of corrs"""
        ret = self.ratio_pionratio(
            corrs, times, nocheck) if self._pionratio else self.ratio_exp(
                corrs, times, nocheck)
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
        sol = (corrs[1]-corrs[0]-(corrs[3]-corrs[2]))/(
            (corrs[5]-corrs[4])-(corrs[7]-corrs[6])
        )
        if not nocheck:
            testsol(sol, corrs, times)
        if self._log:
            sol = takelog(sol)
        try:
            assert sol >= 0 or np.isnan(sol)
        except AssertionError:
            print("bad ratio.  should be 1/ratio")
            print(sol)
            raise
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
            exps(-trial_params[1]*(self._lent-ctime)))+trial_params[2]

    def fit_func_exp_gevp(self, ctime, trial_params, lent=None):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        lent = self._lent if lent is None else lent
        return ((exp(-trial_params[0]*ctime) +
                 exp(-trial_params[1]*(lent-ctime))) + trial_params[2])/(
                     (exp(-trial_params[0]*(TRHS)) +
                      exp(-trial_params[1]*(lent-(TRHS)))) + trial_params[2])

    def fit_func_1p(self, ctime, trial_params,
                    lent=None, tstep_arr=(None, None)):
        """Meta function for effective mass."""
        ret = self.fit_func_1p_pionratio(
            ctime, trial_params, lent) if self._pionratio\
            else self.fit_func_1p_exp(
                ctime, trial_params, lent, tstep_arr)
        return ret

    def fit_func_1p_exp(self, ctime, trial_params,
                        lent=None, tstep_arr=(None, None)):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lent = self._lent if lent is None else lent
        tstep = self._tstep if tstep_arr[0] is None else tstep_arr[0]
        tstep2 = self._tstep2 if tstep_arr[1] is None else tstep_arr[1]
        if tstep_arr[0] is None:
            assert not self._tstep
        if tstep_arr[1] is None:
            assert not self._tstep2
        deltat = self._deltat
        corrs_num = [exp(-trial_params[0]*(ctime-deltat+i*tstep+j*tstep2)) +
                     exp(-trial_params[0]*(lent-(
                         ctime-deltat+i*tstep+j*tstep2)))
                     for j in range(2) for i in range(2)]
        corrs_num[2:] = [0, 0] if not tstep2 else [*corrs_num[2:]]
        corrs_num[2:] = [0, 0] if not self._gevp else corrs_num[2:]
        corrs_denom = [exp(-trial_params[0]*(ctime+i*tstep+j*tstep2)) +
                       exp(-trial_params[0]*(lent-(ctime+i*tstep+j*tstep2)))
                       for j in range(2) for i in range(2)]
        corrs_denom[2:] = [0, 0] if not tstep2 else [*corrs_denom[2:]]
        corrs_denom[2:] = [0, 0] if not self._gevp else corrs_denom[2:]
        if deltat < 0:
            corrs_num, corrs_denom = corrs_denom, corrs_num
        try:
            assert corrs_num[0] >= corrs_denom[0]
        except AssertionError:
            print("bug.  bad ratio")
            print(corrs_num[0], corrs_denom[0])
            raise
        ret = None
        try:
            assert corrs_denom[0] != corrs_denom[
                1] or corrs_denom[2] != corrs_denom[3]
        except AssertionError:
            ret = np.inf
        if ret is None:
            corrs = [*corrs_num, *corrs_denom]
            ret = self.ratio_exp(corrs, ctime, nocheck=True)
        return ret

    def pion_ratio(self, ctime, trial_params, _):
        """Include pions in the denominator of eff mass ratio."""
        tpion = ctime+1/2-self._lent/2.0
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[2]
        return trial_params[0]*(
            cosh(tpion*trial_params[1])+sinh(
                tpion*trial_params[1])/tanh(2*tpion*pionmass))

    @staticmethod
    def ratio_pionratio(corrs, times=None, nocheck=False):
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

    def fit_func_1p_pionratio(self, ctime, trial_params,
                              lent=None, tstep_arr=(None, None)):
        """Find the pion ratio (single pions^2 in the denominator
        of pipi eff mass)"""
        lent = self._lent if lent is None else lent
        tstep = self._tstep if tstep_arr[0] is None else tstep_arr[0]
        # tstep2 = self._tstep2 if tstep_arr[1] is None else tstep_arr[1]
        tpion = [ctime+i*tstep+1/2-lent/2.0 for i in range(3)]
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[1]
        corrs = [trial_params[0]*(
            sinh(tpion[i]*trial_params[1])+cosh(
                tpion[i]*trial_params[1])/tanh(
                    2*tpion[i]*pionmass)) for i in range(3)]
        #return self.ratio_pionratio(corrs, ctime, nocheck=True)
        return corrs[0]

class FitFunc:
    """Exponential fit functions without additive constant"""

    def __init__(self):
        """Init the class components"""
        self._log = LOG
        self._lent = LT
        self._c = C
        self._tstep = TSTEP
        self._tstep2 = TSTEP2
        self._deltat = DELTAT
        self._pionmass = PION_MASS
        self._pionratio = PIONRATIO
        self._gevp = GEVP

    def update(self, upd):
        """Update the class components"""
        self._log = upd.log
        self._lent = upd.lent
        self._c = upd.c
        self._tstep = upd.tstep
        self._tstep2 = upd.tstep2
        self._deltat = upd.deltat
        self._pionmass = upd.pionmass
        self._pionratio = upd.pionratio
        self._gevp = upd.gevp

    def fit_func_exp(self, ctime, trial_params):
        """Give result of function,
        computed to fit the data given in <inputfile> (See procargs(argv))
        """
        return trial_params[0]*(exp(-trial_params[1]*ctime) +
                                exp(-trial_params[1]*(self._lent-ctime)))

    def ratio(self, corrs, times=None, nocheck=False):
        """meta function:  Find the ratio of corrs"""
        ret = self.ratio_pionratio(
            corrs, times, nocheck) if self._pionratio else self.ratio_exp(
                corrs, times, nocheck)
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
        if corrs[1] == 0:
            sol = np.nan
        else:
            sol = (corrs[0])/(corrs[1])
        if not nocheck:
            testsol(sol, corrs, times)
        if self._log:
            sol = takelog(sol)
        try:
            assert sol >= 0 or np.isnan(sol)
        except AssertionError:
            print("bad ratio.  should be 1/ratio")
            print(sol)
            sol = np.nan
            # raise
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
            exps(-trial_params[1]*(self._lent-ctime)))

    def fit_func_exp_gevp(self, ctime, trial_params, lent=None):
        """Give result of function,
        computed to fit the data given in <inputfile>
        (See procargs(argv)) GEVP, cosh+const
        """
        lent = self._lent if lent is None else lent
        return (exp(-trial_params[0]*ctime) +
                exp(-trial_params[1]*(lent-ctime)))/(
                    (exp(-trial_params[0]*(TRHS)) +
                     exp(-trial_params[1]*(lent-(TRHS)))))

    def fit_func_1p(self, ctime, trial_params,
                    lent=None, tstep_arr=(None, None)):
        """Meta function for effective mass."""
        ret = self.fit_func_1p_pionratio(
            ctime, trial_params, lent) if self._pionratio\
            else self.fit_func_1p_exp(
                ctime, trial_params, lent, tstep_arr)
        return ret

    def fit_func_1p_exp(self, ctime, trial_params, lent=None,
                        tstep_arr=(None, None)):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        # tstep = self._tstep if tstep_arr[0] is None else tstep_arr[0]
        # tstep2 = self._tstep2 if tstep_arr[1] is None else tstep_arr[1]
        if tstep_arr:
            pass
        deltat = self._deltat
        lent = self._lent if lent is None else lent
        corrs = [exp(-trial_params[0]*(ctime-i*deltat)) +
                 exp(-trial_params[0]*(lent-(ctime-i*deltat)))
                 for i in range(2)]
        if deltat > 0:
            corrs[0], corrs[1] = corrs[1], corrs[0]
        return self.ratio_exp(corrs, ctime, nocheck=True)

    def fit_func_1p_pionratio(self, ctime, trial_params,
                              lent=None, tstep_arr=(None, None)):
        """one parameter eff. mass fit function
        for EFF_MASS_METHOD = 3
        """
        lent = self._lent if lent is None else lent
        tstep = self._tstep if tstep_arr[0] is None else tstep_arr[0]
        # tstep2 = self._tstep2 if tstep_arr[1] is None else tstep_arr[1]
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[1]
        tpion = [ctime+i*tstep+1/2-lent/2.0 for i in range(2)]
        corrs = [trial_params[0]*(sinh((tpion[i]-1/2)*trial_params[
            1]-1/2*pionmass)+cosh(
                (tpion[i]-1/2)*trial_params[1]-1/2*pionmass))/tanh(
                    2*tpion[i]*pionmass) for i in range(2)]
        #return self.ratio_pionratio(corrs, ctime, nocheck=True)
        return corrs[0]

    def ratio_pionratio(self, corrs, times=None, nocheck=False):
        """Process data points into effective mass ratio (pion ratio),
        no additive constant
        """
        times = [-99999, -99999] if times is None else times
        times = [times, None, None] if isinstance(times, Number) else times
        assert USE_FIXED_MASS, "Only fixed pion mass supported"+\
            " in eff mass pion ratio fits."
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

    def pion_ratio(self, ctime, trial_params, lent=None):
        """Include pions in the denominator of eff mass ratio."""
        lent = self._lent if lent is None else lent
        tpion = ctime+1/2-lent/2.0
        pionmass = self._pionmass if USE_FIXED_MASS else trial_params[2]
        return trial_params[0]*(sinh(tpion*trial_params[1])+cosh(
            tpion*trial_params[1])/tanh(2*tpion*pionmass))
