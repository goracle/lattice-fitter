import numdifftools as nd
from numpy.linalg import inv
from math import sqrt
import sys

from latfit.mathfun.chi_sq import chi_sq

def geterr(result_min, covinv, coords):
    param_err = ["err" for i in range(len(result_min.x))]
    if result_min.fun > 0 and result_min.status == 0:
        HFUNC = lambda xrray: chi_sq(xrray, covinv, coords)
        HFUN = nd.Hessian(HFUNC)
        #compute hessian inverse
        HINV = inv(HFUN(result_min.x))
        #compute errors in fit parameters
        for i in range(len(HINV)):
            try:
                param_err[i] = sqrt(2*HINV[i][i])
            except ValueError:
                print "***ERROR***"
                print "Examine fit domain.  Hessian inverse has negative"
                print "diagonal entries, leading to complex errors in some"
                print "or all of the fit parameters."
                sys.exit(1)
    return param_err
           
