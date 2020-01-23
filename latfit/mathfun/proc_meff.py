"""Get the effective mass from lines/files provided."""
# from math import log, acosh
import sys
import re
import collections
from math import acosh
import numbers
from sympy import nsolve
from sympy.abc import x, y, z
# from scipy.optimize import minimize_scalar, brentq
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
import numpy as np

from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS_METHOD
from latfit.config import FIT
from latfit.config import START_PARAMS
from latfit.config import ADD_CONST
from latfit.config import STYPE
from latfit.config import LOGFORM
from latfit.config import GEVP, ADD_CONST_VEC, LT_VEC
from latfit.config import METHOD
from latfit.config import ORIGL
from latfit.config import MATRIX_SUBTRACTION
from latfit.config import DELTA_E2_AROUND_THE_WORLD
from latfit.analysis.test_arg import NegLogArgument
from latfit.analysis.errorcodes import NegativeEnergy, PrecisionLossError
# from latfit.analysis.profile import PROFILE
import latfit.config

ADD_CONST_VEC = list(ADD_CONST_VEC)
if MATRIX_SUBTRACTION and GEVP:
    ADD_CONST_VEC = [0 for i in ADD_CONST_VEC]

RANGE1P = 3 if ADD_CONST else 2

# almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
    def proc_meff(lines, index=None, files=None, time_arr=None):
        """Gets the effective mass, given three points
        Solve an acosh.  (Needs a user input on the addititive const)
        (See config)
        """
        corrs, times = pre_proc_meff(lines, files, time_arr)
        sol = latfit.config.FITS['acosh_ratio'](
            corrs, times) if index is None else\
            latfit.config.FITS.fid['acosh_ratio'][
                ADD_CONST_VEC[index]](corrs, times)
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
    def proc_meff(lines, _, files=None, time_arr=None):
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
            assert None
            if files:
                for cfile in files:
                    print(files)
            raise NegativeEnergy
        return sol[1]
        #print("Found solution:", sol[1])

    if ADD_CONST:
        def proc_meff_systemofeqns(corrs, times):
            """solve system of 3 equations numerically."""
            return nsolve((latfit.config.FITS['fit_func_sym'](
                times[0], [x, y, z])-corrs[0],
                           latfit.config.FITS['fit_func_sym'](
                               times[1], [x, y, z])-corrs[1],
                           latfit.config.FITS['fit_func_sym'](
                               times[2], [x, y, z])-corrs[2]),
                          (x, y, z), START_PARAMS)
    else:
        def proc_meff_systemofeqns(corrs, times):
            """solve system of 2 equations numerically."""
            return nsolve((latfit.config.FITS['fit_func_sym'](
                times[0], [x, y])-corrs[0],
                           latfit.config.FITS['fit_func_sym'](
                               times[1], [x, y])-corrs[1]),
                          (x, y), START_PARAMS)

# one parameter fit, optional additive constant (determined in config)
elif EFF_MASS_METHOD == 3:
    def proc_meff(lines, index=None, files=None, time_arr=None):
        """fit to a function with one free parameter
        [ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
        """
        corrs, times = pre_proc_meff(lines, files, time_arr)
        sol = latfit.config.FITS['ratio'](corrs, times) if index is None else\
            latfit.config.FITS.fid['ratio'][ADD_CONST_VEC[index]](corrs, times)
        return sol

# sliding window, solved by minimizing a one parameter cost function
elif EFF_MASS_METHOD == 4:

    def proc_meff(lines, index=None, files=None, time_arr=None):
        """Process data, meff (traditional definition)"""
        corrs, times = pre_proc_meff(lines, files, time_arr)
        corrs = list(corrs)
        if GEVP:
            corrs[2], corrs[3] = (None,
                                  None) if not (ADD_CONST) else (
                                      corrs[2], None)
        else:
            corrs[2], corrs[3] = (None,
                                  None) if not (ADD_CONST) else (
                                      corrs[2], None)
        return proc_meff4(corrs, index, files, times)

    def make_eff_mass_tomin(ini, add_const_bool, tstep_arr):
        """Create the functions to be minimized."""
        def eff_mass_tomin(energy, ctime, sol):
            """Minimize this
            (quadratic) to solve a sliding window problem."""
            return (latfit.config.FITS.fid['fit_func_1p'][add_const_bool](
                ctime, [energy], LT_VEC[ini], tstep_arr)-sol)**2
        return eff_mass_tomin

    def create_funcs():
        """Create eff mass functions; return list"""
        ret = []
        for i, j in enumerate(ADD_CONST_VEC):
            tstep = None
            tstep2 = None
            if MATRIX_SUBTRACTION and GEVP:
                j = 1
                tstep = -1*latfit.config.DELTA_T_MATRIX_SUBTRACTION
                tstep2 = -1*latfit.config.DELTA_T2_MATRIX_SUBTRACTION if\
                    DELTA_E2_AROUND_THE_WORLD is not None else None
            ret.append(make_eff_mass_tomin(i, j, (tstep, tstep2)))
        return ret

    EFF_MASS_TOMIN = create_funcs()

    def eff_mass_root(energy, ctime, sol):
        """Minimize this
        (find a root) to solve a sliding window problem."""
        assert None, 'not supported.'
        return latfit.config.FITS.fid['fit_func_1p'](ctime, [energy])-sol

    def proc_meff4(corrs, index, _, times=(None)):
        """numerically solve a function with one free parameter
        (e.g.) [ C(t) ]/[ C(t+1) ]
        This is the conventional effective mass formula.
        """
        #if any(np.isnan(np.array(corrs, dtype=np.complex)[:2])):
        #    sol = np.nan
        #else:
        if not LOGFORM:
            try:
                sol = latfit.config.FITS['ratio'](
                    corrs, times) if index is None else latfit.config.FITS.fid[
                        'ratio'][ADD_CONST_VEC[
                            index]](corrs, times)
                assert not np.isnan(sol) or any(np.isnan(np.array(
                    corrs, dtype=np.complex))),\
                    "solution to energy from eval is unexpectedly nan."+str(
                        corrs)
            except NegLogArgument:
                sol = np.nan
        else:
            sol = corrs[0]
            sol = np.nan if sol < 0 else sol
        if np.isnan(sol):
            assert None, "check this"
            raise PrecisionLossError
            #errstr = "bad time/op combination in fit range."+\
                #" (time, op index)=("+str(times[0])+","+str(index)+")"
            #if FIT:
                #if not(times and times[0] in latfit.config.FIT_EXCL[index]):
                    #latfit.config.FIT_EXCL[index].append(times[0])
            #else:
            #    if not times[0] in proc_meff4.badtimes:
                    #print(errstr)
                    #print('operator index=', index)
                    #print("times=", times)
                    #proc_meff4.badtimes.append(times[0])

        index = 0 if index is None else index
        try:
            if ORIGL > 4:
                assert None, "This method is not supported and is based on flawed assumptions."
                sol = minimize(EFF_MASS_TOMIN[index], START_PARAMS,
                               args=(times[0], sol),
                               method=METHOD, tol=1e-20,
                               options={'disp': True,
                                        'maxiter': 10000,
                                        'maxfev': 10000,})
            else:
                if not np.isnan(sol):
                    sol = minimize_scalar(EFF_MASS_TOMIN[index],
                                          args=(times[0], sol),
                                          bounds=(0, None))
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
            print(minimize_scalar(EFF_MASS_TOMIN[index], args=(times[0], sol)))
            sys.exit(1)
        if not np.isnan(sol):
            sol = checksol(sol, index, times, corrs, fun)
        return sol
    proc_meff4.badtimes = []

    def checksol(sol, index, times, corrs, fun):
        """Check the solution."""
        if isinstance(sol, collections.Iterable):
            test = any(i < 0 for i in sol[1:])
        else:
            test = sol < 0
        if test:
            ratioval = latfit.config.FITS.fid[
                'ratio'] if index is None else latfit.config.FITS.fid[
                    'ratio'][ADD_CONST_VEC[index]](corrs, times)
            ratioval = corrs[0] if LOGFORM else ratioval
            sol = np.array(sol)
            tryfun = (EFF_MASS_TOMIN[index](
                -1*sol, times[0], ratioval) - fun)
            if tryfun/(fun+1e-24) < 10:
                sol = -1*sol
                #print("positive solution close to" +
                #      " negative solution; switching; new tol", tryfun)
                assert abs(tryfun) < 1e-12, "New tolerance too large:"+str(
                    tryfun)
            else:
                print("***ERROR***\nnegative energy found:", sol, times)
                print(corrs[0], fun, tryfun)
                print(EFF_MASS_TOMIN[index](sol, times[0], ratioval))
                print(EFF_MASS_TOMIN[index](-sol, times[0], ratioval))
                assert None
                raise NegativeEnergy
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

if STYPE == 'hdf5':
    def pre_proc_meff(lines, files=None, times=None):
        """Extract values from files or from fake files for proc_meff"""
        corrs = lines
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
