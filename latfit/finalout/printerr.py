import numdifftools as nd
from numpy.linalg import inv
from math import sqrt

from latfit.mathfun.chi_sq import chi_sq

def printerr(result_min, covinv, coords, switch):
    if result_min.fun > 0 and result_min.status == 0:
        HFUNC = lambda xrray: chi_sq(xrray, covinv, coords, switch)
        HFUN = nd.Hessian(HFUNC)
        #compute hessian inverse
        HINV = inv(HFUN(result_min.x))
        #compute errors in fit parameters
        for i in range(len(HINV)):
            print "Minimized parameter #", i, " = "
            try:
                print result_min.x[i], "+/-", sqrt(2*HINV[i][i])
            except ValueError:
                print "***ERROR***"
                print "Examine fit domain.  Hessian inverse has negative"
                print "diagonal entries, leading to complex errors in some"
                print "or all of the fit parameters."
                sys.exit(1)
    return 0
        #ERR_A0 = sqrt(2*HINV[0][0])
        #ERR_ENERGY = sqrt(2*HINV[1][1])
        #print "a0 = ", result_min.x[0], "+/-", ERR_A0
        #print "energy = ", result_min.x[1], "+/-", ERR_ENERGY
