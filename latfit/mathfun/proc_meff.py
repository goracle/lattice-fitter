"""Get the effective mass from lines/files provided."""
#from math import log, acosh
import sys
import re
from warnings import warn
from math import acosh
from sympy import nsolve
from sympy.abc import x, y, z

from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import FIT
from latfit.config import fit_func_3pt_sym
from latfit.config import START_PARAMS

#almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        """Gets the effective mass, given three points
        Solve an acosh.  (Needs a user input on the addititive const)
        (See config)
        """
        if time_arr:
            pass
        if not files:
            corr1 = line1
            corr2 = line2
            corr3 = line3
        else:
            corr1 = proc_line(line1, files[0])
            corr2 = proc_line(line2, files[1])
            corr3 = proc_line(line3, files[2])
        arg = (corr1+corr3-2*C)/2/(corr2-C)
        if arg < 1:
            print("***ERROR***")
            print("argument to acosh in effective mass calc is less than 1:", arg)
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            sys.exit(1)
        return acosh(arg)

elif EFF_MASS_METHOD == 2:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        """numerically solve a system of three transcendental equations
        return the eff. mass
        """
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

elif EFF_MASS_METHOD == 3 and FIT:
    def proc_meff(line1, line2, line3, files=None, time_arr=None):
        """fit to a function with one free parameter
        [ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
        """
        if not files:
            corr1 = line1
            corr2 = line2
            corr3 = line3
        else:
            corr1 = proc_line(line1, files[0])
            corr2 = proc_line(line2, files[1])
            corr3 = proc_line(line3, files[2])
        arg = (corr2-corr1)/(corr3-corr2)
        if arg < 0 and proc_meff.sent != 0:
            #print("***ERROR***")
            warn("argument to log in eff. mass calc is than 0: "+str(arg))
            print("argument to log in effective mass calc is less than 0:", arg)
            print(corr1, corr2, corr3)
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            if not time_arr is None:
                print(time_arr)
            proc_meff.sent = 0
        return arg
else:
    print("Bad method for finding the effective mass specified:",
          EFF_MASS_METHOD, "with fit set to", FIT)
    sys.exit(1)
proc_meff.sent = object()
