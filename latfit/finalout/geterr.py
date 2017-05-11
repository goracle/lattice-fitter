import numdifftools as nd
from numpy.linalg import inv
from math import sqrt
import sys
from latfit.config import fit_func
from latfit.config import SCALE
from latfit.config import CUTOFF

from latfit.mathfun.chi_sq import chi_sq

def geterr(result_min, covinv, coords):
    param_err = ["err" for i in range(len(result_min.x))]
    if result_min.fun > 0 and result_min.status == 0:
        HFUNC = lambda xrray: chi_sq(xrray, covinv, coords)
        HFUN = nd.Hessian(HFUNC)
        #compute hessian inverse
        HINV = inv(HFUN(result_min.x))
        #compute errors in fit parameters
        flag = 0
        cutoff = 1/(CUTOFF*SCALE)
        for i in range(len(HINV)):
            try:
                param_err[i] = sqrt(2*HINV[i][i])
            except ValueError:
                print "***Warning***"
                print "Hessian inverse has negative diagonal entries,"
                print "leading to complex errors in some or all of the"
                print "fit parameters. failing entry:",2*HINV[i][i]
                print "Attempting to continue with entries zero'd below", cutoff
                flag = 1
                break
        if flag == 1:
            l=len(HFUN(result_min.x))
            HFUN_CUT=[[HFUN(result_min.x)[j][i] if HFUN(result_min.x)[j][i] > cutoff else 0 for i in range(l)] for j in range(l)]
            try:
                HINV = inv(HFUN_CUT)
            except:
                print "Cut Hessian is singular. Cut Hessian:"
                print HFUN_CUT
                sys.exit(1)
            for i in range(len(HINV)):
                try:
                    param_err[i] = sqrt(2*HINV[i][i])
                except ValueError:
                    print "***Warning***"
                    print "Hessian inverse has negative diagonal entries,"
                    print "leading to complex errors in some or all of the"
                    print "fit parameters. failing entry:",2*HINV[i][i]
                    print "Giving up."
                    sys.exit(1)
    print 'Debugging info:'
    print '2x Hessian Inverse:'
    print 2*HINV
    print 'Hessian:'
    print HFUN(result_min.x)
    print 'x: (f(x)-y(x))/sigma(x):'
    for j in range(len(covinv)):
        print j
        print covinv[j][j]*(fit_func(coords[j][0],result_min.x)-coords[j][1])
    return param_err
           
