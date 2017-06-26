from math import log,acosh
from sympy import nsolve,cosh
from sympy.abc import x,y,z

from latfit.config import EFF_MASS_METHOD
from latfit.config import C
from latfit.config import FIT
from latfit.config import fit_func_3pt_sym

#almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
    def proc_MEFF(line1,line2,line3,files):
        fn1=files[0]
        fn2=files[1]
        fn3=files[2]
        C1 = proc_line(line1,fn1)
        C2 = proc_line(line2,fn2)
        C3 = proc_line(line3,fn3)
        arg = (C1+C3-2*C)/2/(C2-C)
        if arg < 1:
            print("***ERROR***")
            print("argument to acosh in effective mass calc is less than 1:",arg)
            print(fn1)
            print(fn2)
            print(fn3)
            sys.exit(1)
        return acosh(arg)

#numerically solve a system of three transcendental equations
elif EFF_MASS_METHOD == 2:
    def proc_MEFF(line1,line2,line3,files):
        fn1=files[0]
        fn2=files[1]
        fn3=files[2]
        try:
            t1=float(re.search('t([0-9]+)',fn1).group(1))
            t2=float(re.search('t([0-9]+)',fn2).group(1))
            t3=float(re.search('t([0-9]+)',fn3).group(1))
        except:
            print("Bad blocks:",fn1,fn2,fn3)
            print("must have t[0-9] in name, e.g. blk.t3")
            sys.exit(1)
        C1 = proc_line(line1,fn1)
        C2 = proc_line(line2,fn2)
        C3 = proc_line(line3,fn3)
        try:
            sol = nsolve((fit_func_3pt_sym(t1,[x,y,z])-C1, fit_func_3pt_sym(t2,[x,y,z])-C2,fit_func_3pt_sym(t3,[x,y,z])-C3), (x,y,z), START_PARAMS)
        except ValueError:
            print("Solution not within tolerance.")
            print(C1,fn1)
            print(C2,fn2)
            print(C3,fn3)
            return 0
        if sol[1] < 0:
            print("***ERROR***")
            print("negative energy found:",sol[1])
            print(fn1)
            print(fn2)
            print(fn3)
            sys.exit(1)
        print("Found solution:",sol[1])
        return sol[1]
#fit to a function with one free parameter
#[ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
elif EFF_MASS_METHOD == 3 and FIT:
    def proc_MEFF(line1,line2,line3,files):
        fn1=files[0]
        fn2=files[1]
        fn3=files[2]
        C1 = proc_line(line1,fn1)
        C2 = proc_line(line2,fn2)
        C3 = proc_line(line3,fn3)
        arg = (C2-C1)/(C3-C2)
        if arg < 1:
            print("***ERROR***")
            print("argument to acosh in effective mass calc is less than 1:",arg)
            print(fn1)
            print(fn2)
            print(fn3)
            sys.exit(1)
        #print 'solution =',sol
        return log(arg)
else:
    print("Bad method for finding the effective mass specified:", EFF_MASS_METHOD, "with fit set to", FIT)
    sys.exit(1)
