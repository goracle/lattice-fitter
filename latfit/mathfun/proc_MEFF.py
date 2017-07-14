#from math import log,acosh
from numpy import arccosh,log
from sympy import nsolve,cosh
from sympy.abc import x,y,z
import sys
from warnings import warn

from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import FIT
from latfit.config import fit_func_3pt_sym

#almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
    def proc_MEFF(line1,line2,line3,files=None,time_arr=None):
        if not files:
            C1=line1
            C2=line2
            C3=line3
        else:
            C1 = proc_line(line1,files[0])
            C2 = proc_line(line2,files[1])
            C3 = proc_line(line3,files[2])
        arg = (C1+C3-2*C)/2/(C2-C)
        if arg < 1:
            print("***ERROR***")
            print("argument to acosh in effective mass calc is less than 1:",arg)
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            sys.exit(1)
        return arccosh(arg)

#numerically solve a system of three transcendental equations
elif EFF_MASS_METHOD == 2:
    def proc_MEFF(line1,line2,line3,files=None,time_arr=None):
        if not files:
            C1=line1
            C2=line2
            C3=line3
        else:
            try:
                t1=float(re.search('t([0-9]+)',files[0]).group(1))
                t2=float(re.search('t([0-9]+)',files[1]).group(1))
                t3=float(re.search('t([0-9]+)',files[2]).group(1))
            except:
                print("Bad blocks:",files[0],files[1],files[2])
                print("must have t[0-9] in name, e.g. blk.t3")
                sys.exit(1)
            C1 = proc_line(line1,files[0])
            C2 = proc_line(line2,files[1])
            C3 = proc_line(line3,files[2])
        try:
            sol = nsolve((fit_func_3pt_sym(t1,[x,y,z])-C1, fit_func_3pt_sym(t2,[x,y,z])-C2,fit_func_3pt_sym(t3,[x,y,z])-C3), (x,y,z), START_PARAMS)
        except ValueError:
            print("Solution not within tolerance.")
            if files:
                print(C1,files[0])
                print(C2,files[1])
                print(C3,files[2])
            else:
                print(C1,C2,C3)
            return 0
        if sol[1] < 0:
            print("***ERROR***")
            print("negative energy found:",sol[1])
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            sys.exit(1)
        print("Found solution:",sol[1])
        return sol[1]
#fit to a function with one free parameter
#[ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
elif EFF_MASS_METHOD == 3 and FIT:
    def proc_MEFF(line1,line2,line3,files=None,time_arr=None):
        if not files:
            C1=line1
            C2=line2
            C3=line3
        else:
            C1 = proc_line(line1,files[0])
            C2 = proc_line(line2,files[1])
            C3 = proc_line(line3,files[2])
        arg = (C2-C1)/(C3-C2)
        if arg < 0 and proc_MEFF.sent != 0:
            #print("***ERROR***")
            warn("argument to log in effective mass calc is less than 0:"+str(arg))
            print(C1,C2,C3)
            if files:
                print(files[0])
                print(files[1])
                print(files[2])
            if not time_arr == None:
                print(time_arr)
            #sys.exit(1)
            proc_MEFF.sent = 0
        return (arg)
else:
    print("Bad method for finding the effective mass specified:", EFF_MASS_METHOD, "with fit set to", FIT)
    sys.exit(1)
proc_MEFF.sent=object()
