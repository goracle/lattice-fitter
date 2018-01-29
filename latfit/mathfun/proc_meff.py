"""Get the effective mass from lines/files provided."""
#from math import log, acosh
import sys
import re
from math import acosh
import numbers
from sympy import nsolve
from sympy.abc import x, y, z
from sympy import log as logs
from math import log
import numpy as np
from scipy.optimize import minimize_scalar, brentq

from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import PROFILE
from latfit.config import FIT
from latfit.config import fit_func_3pt_sym
from latfit.config import fit_func_exp
from latfit.config import START_PARAMS
from latfit.config import LOG
from latfit.config import ADD_CONST
from latfit.config import fit_func_1p
from latfit.config import ratio
from latfit.analysis.test_arg import test_arg

#almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
    def proc_meff(lines, files=None, time_arr=None):
        """Gets the effective mass, given three points
        Solve an acosh.  (Needs a user input on the addititive const)
        (See config)
        """
        corrs, times = pre_proc_meff(lines, files, time_arr)
        sol = acosh_ratio(corrs, times)
        if sol < 1:
            print("***ERROR***")
            print("argument to acosh in effective mass" + \
                  " calc is less than 1:", sol)
            if files:
                for fn1 in files:
                    print(fn1)
            if time_arr is not None:
                print("problematic time slice(s):", time_arr)
            sys.exit(1)
        return acosh(sol)

#sliding window method
elif EFF_MASS_METHOD == 2:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        """numerically solve a system of three transcendental equations
        return the eff. mass
        """
        if not ADD_CONST:
            print("***ERROR***", "eff_mass method 2 No longer actively supported")
            sys.exit(1)
        if time_arr:
            pass
        if not files:
            corr1 = line1
            corr2 = line2
            corr3 = line3
        else:
            try:
                time1 = float(re.search('t([0-9]+)', files[0]).group(1))
                time2 = float(re.search('t([0-9]+)', files[1]).group(1))
                time3 = float(re.search('t([0-9]+)', files[2]).group(1))
            except ValueError:
                print("Bad blocks:", files[0], files[1], files[2])
                print("must have t[0-9] in name, e.g. blk.t3")
                sys.exit(1)
            corr1 = proc_line(line1, files[0])
            corr2 = proc_line(line2, files[1])
            corr3 = proc_line(line3, files[2])
        try:
            sol = nsolve((fit_func_3pt_sym(
                time1, [x, y, z])-corr1, fit_func_3pt_sym(
                    time2, [x, y, z])-corr2, fit_func_3pt_sym(
                        time3, [x, y, z])-corr3), (x, y, z), START_PARAMS)
        except ValueError:
            print("Solution not within tolerance.")
            if files:
                print(corr1, files[0])
                print(corr2, files[1])
                print(corr3, files[2])
            else:
                print(corr1, corr2, corr3)
            return 0
        if sol[1] < 0:
            print("***ERROR***")
            print("negative energy found:", sol[1])
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            sys.exit(1)
        print("Found solution:", sol[1])
        return sol[1]

#one parameter fit, additive constant
elif EFF_MASS_METHOD == 3 and not ADD_CONST:
    def proc_meff(line1, line2, _, files=None, time_arr=None):
        """fit to a function with one free parameter
        [ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
        """
        corr1, corr2, _ = pre_proc_meff(line1, line2, line3, files, time_arr)
        sol = ratio(corr1, corr2, None, time_arr)
        return sol

#one param fit, no add const.
elif EFF_MASS_METHOD == 3 and ADD_CONST:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        """fit to a function with one free parameter
        [ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
        """
        corr1, corr2, corr3 = pre_proc_meff(line1, line2, line3, files, time_arr)
        sol = ratio(corr1, corr2, corr3, time_arr)
        return sol


#sliding window, no additive constant.
elif EFF_MASS_METHOD == 4 and not ADD_CONST:
    def proc_meff(line1, line2, _, files=None, time_arr=None):
        corr1, corr2, _ = pre_proc_meff(line1, line2, line3, files, time_arr)
        return proc_meff4(corr1, corr2, None, files, time_arr)

#sliding window, additive constant.
elif EFF_MASS_METHOD == 4:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        corr1, corr2, corr3 = pre_proc_meff(line1, line2, line3, files, time_arr)
        return proc_meff4(corr1, corr2, corr3, files, time_arr)

elif FIT:
    print("Bad method for finding the effective mass specified:",
          EFF_MASS_METHOD, "with fit set to", FIT)
    sys.exit(1)
else:
    def proc_meff(*args):
        """Do nothing"""
        if args:
            pass
proc_meff.sent = object()

def eff_mass_tomin(energy, ctime, sol):
    """Minimize this
    (quadratic) to solve a sliding window problem."""
    return (fit_func_1p(ctime, [energy])-sol)**2
def eff_mass_root(energy, ctime, sol):
    """Minimize this
    (find a root) to solve a sliding window problem."""
    return (fit_func_1p(ctime, [energy])-sol)

def pre_proc_meff(line1, line2, line3, files=None, time_arr=None):
    """Extract values from files or from fake files for proc_meff"""
    time1 = time_arr
    if not files:
        corr1 = line1
        corr2 = line2
        corr3 = line3
    else:
        corr1 = proc_line(line1, files[0])
        corr2 = proc_line(line2, files[1])
        if line3 is not None:
            corr3 = proc_line(line3, files[2])
        else:
            corr3 = None
    return corr1, corr2, corr3

def proc_meff4(corr1, corr2, corr3, files=None, time1=None):
    """numerically solve a function with one free parameter
    (e.g.) [ C(t) ]/[ C(t+1) ]
    This is the conventional effective mass formula.
    """
    sol = ratio(corr1, corr2, corr3, time1)
    try:
        sol = minimize_scalar(eff_mass_tomin, args=(time1, sol), bounds=(0, None))
        fun = sol.fun
        sol = sol.x
        #other solution methods:
        #sol = brentq(eff_mass_root, 0, 5, args=(time1, sol)) #too unstable
        #sol = nsolve((logs(fit_func_3pt_sym( #too slow
        #    time1, [1, y, 0])/fit_func_3pt_sym(
        #        time1+1, [1, y, 0]))-sol), (y), START_PARAMS) 
        sol = float(sol)
    except ValueError:
        print("***ERROR***\nSolution not within tolerance.")
        print(sol, time_arr)
        print(corr1, corr2, corr3)
        print(minimize_scalar(eff_mass_tomin, args=(time1, sol)))
        sys.exit(1)
    if sol < 0:
        if (eff_mass_tomin(-sol, time1,
                            ratio(corr1, corr2, corr3, time1)) -fun)/fun < 10:
            sol = -sol
            print("positive solution close to negative solution; switching.")
        else:
            print("***ERROR***\nnegative energy found:", sol, time_arr)
            print(eff_mass_tomin(sol, time_arr, ratio(corr1, corr2, corr3, time_arr)))
            print(eff_mass_tomin(-sol, time1, ratio(corr1, corr2, corr3, time_arr)))
            sys.exit(1)
    return sol
