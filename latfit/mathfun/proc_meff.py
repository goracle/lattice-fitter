"""Get the effective mass from lines/files provided."""
# from math import log, acosh
import sys
import re
from math import acosh
import numbers
from sympy import nsolve
from sympy.abc import x, y, z
# from scipy.optimize import minimize_scalar, brentq
from scipy.optimize import minimize_scalar

from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS_METHOD
from latfit.config import FIT
from latfit.config import fit_func_sym
from latfit.config import START_PARAMS
from latfit.config import ADD_CONST
from latfit.config import STYPE
from latfit.config import RANGE1P
from latfit.config import fit_func_1p
from latfit.config import ratio
from latfit.config import acosh_ratio
# from latfit.analysis.profile import PROFILE
import latfit.config

# almost solve a cosh, analytic
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
            print("argument to acosh in effective mass" +
                  " calc is less than 1:", sol)
            if files:
                for fn1 in files:
                    print(fn1)
            if time_arr is not None:
                print("problematic time slice(s):", time_arr)
            sys.exit(1)
        return acosh(sol)

# sliding window method,
# solved by solving a system of (transcendental) equations
elif EFF_MASS_METHOD == 2:
    def proc_meff(lines, files=None, time_arr=None):
        """numerically solve a system of three transcendental equations
        return the eff. mass
        """
        corrs, times = pre_proc_meff(lines, files, time_arr)
        try:
            sol = proc_meff_systemofeqns(corrs, times)
        except ValueError:
            print("Solution not within tolerance.")
            if files:
                for corr, cfile in zip(corrs, files):
                    print(corr, cfile)
            else:
                print(corrs)
            return 0
        if sol[1] < 0:
            print("***ERROR***")
            print("negative energy found:", sol[1])
            if files:
                for cfile in files:
                    print(files)
            sys.exit(1)
        print("Found solution:", sol[1])
        return sol[1]

    if ADD_CONST:
        def proc_meff_systemofeqns(corrs, times):
            """solve system of 3 equations numerically."""
            return nsolve((fit_func_sym(times[0], [x, y, z])-corrs[0],
                           fit_func_sym(times[1], [x, y, z])-corrs[1],
                           fit_func_sym(times[2], [x, y, z])-corrs[2]),
                          (x, y, z), START_PARAMS)
    else:
        def proc_meff_systemofeqns(corrs, times):
            """solve system of 2 equations numerically."""
            return nsolve((fit_func_sym(times[0], [x, y])-corrs[0],
                           fit_func_sym(times[1], [x, y])-corrs[1]),
                          (x, y), START_PARAMS)

# one parameter fit, optional additive constant (determined in config)
elif EFF_MASS_METHOD == 3:
    def proc_meff(lines, files=None, time_arr=None):
        """fit to a function with one free parameter
        [ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
        """
        corrs, times = pre_proc_meff(lines, files, time_arr)
        sol = ratio(corrs, times)
        return sol

# sliding window, solved by minimizing a one parameter cost function
elif EFF_MASS_METHOD == 4:

    def proc_meff(lines, files=None, time_arr=None):
        """Process data, meff (traditional definition)"""
        corrs, times = pre_proc_meff(lines, files, time_arr)
        corrs[2], corrs[3] = (None,
                              None) if not ADD_CONST else (corrs[2], None)
        return proc_meff4(corrs, files, times)

    def eff_mass_tomin(energy, ctime, sol):
        """Minimize this
        (quadratic) to solve a sliding window problem."""
        return (fit_func_1p(ctime, [energy])-sol)**2

    def eff_mass_root(energy, ctime, sol):
        """Minimize this
        (find a root) to solve a sliding window problem."""
        return fit_func_1p(ctime, [energy])-sol

    def proc_meff4(corrs, _, times=(None)):
        """numerically solve a function with one free parameter
        (e.g.) [ C(t) ]/[ C(t+1) ]
        This is the conventional effective mass formula.
        """
        sol = ratio(corrs, times)
        try:
            sol = minimize_scalar(eff_mass_tomin,
                                  args=(times[0], sol), bounds=(0, None))
            fun = sol.fun
            sol = sol.x
            # other solution methods:
            # too unstable
            # sol = brentq(eff_mass_root, 0, 5, args=(times[0], sol))
            # too slow
            # sol = nsolve((logs(fit_func_3pt_sym( #too slow
            #    time1, [1, y, 0])/fit_func_3pt_sym(
            #        time1+1, [1, y, 0]))-sol), (y), START_PARAMS)
            sol = float(sol)
        except ValueError:
            print("***ERROR***\nSolution not within tolerance.")
            print("sol, time_arr:", sol, times)
            print("corrs:", corrs)
            print(minimize_scalar(eff_mass_tomin, args=(times[0], sol)))
            sys.exit(1)
        if sol < 0:
            if (eff_mass_tomin(-sol, times[0],
                               ratio(corrs, times)) - fun)/fun < 10:
                sol = -sol
                print("positive solution close to" +
                      " negative solution; switching.")
            else:
                print("***ERROR***\nnegative energy found:", sol, times)
                print(eff_mass_tomin(sol, times[0], ratio(corrs, times)))
                print(eff_mass_tomin(-sol, times[0], ratio(corrs, times)))
                sys.exit(1)
        return sol

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

if STYPE == 'hdf5':
    def pre_proc_meff(lines, files=None, times=None):
        """Extract values from files or from fake files for proc_meff"""
        corrs = lines
        # delete this next line if everything is workign
        assert isinstance(times,
                          numbers.Number), "time_arr is not number. hdf5."
        times = [times+i*latfit.config.TSTEP for i in range(RANGE1P)]
        if files:
            corrs = [proc_line(line, cfile) for
                     line, cfile in zip(lines, files)]
        else:
            corrs = lines
        return corrs, times

elif STYPE == 'ascii':
    def pre_proc_meff(lines, files=None, times=None):
        """Extract values from files or from fake files for proc_meff"""
        corrs = lines
        if files and times is None:
            try:
                times = [float(re.search('t([0-9]+)', files[i]).group(1))
                         for i in range(RANGE1P)]
            except ValueError:
                print("Bad blocks:", files[0], files[1], files[2])
                print("must have t[0-9] in name, e.g. blk.t3")
                sys.exit(1)
        if files:
            corrs = [proc_line(line, cfile) for
                     line, cfile in zip(lines, files)]
        else:
            corrs = lines
        return corrs, times
else:
    raise Exception("Unsupported file type:", STYPE)
