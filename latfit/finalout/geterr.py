import numdifftools as nd
from numpy.linalg import inv
from math import sqrt
import sys
from latfit.config import fit_func
from latfit.config import SCALE
from latfit.config import CUTOFF
import numpy as np
from math import fsum

from latfit.mathfun.chi_sq import chi_sq

#compute errors in fit parameters
def geterr(result_min, covinv, coords):
    param_err = ["err" for i in range(len(result_min.x))]
    if result_min.fun > 0 and result_min.status == 0:
        HFUNC = lambda xrray: chi_sq(xrray, covinv, coords)
        HFUN = nd.Hessian(HFUNC)
        flag = 0
        cutoff = 1/(CUTOFF*SCALE)
        l_coords=len(coords)
        l_params=len(result_min.x)

        #delete this!
        if len(HFUN(result_min.x)) != len(result_min.x):
            print "huh."
            sys.exit(1)

        #compute the gradient and hessian of the fit function
        def g(x):
            return np.array([fit_func(coords[i][0],x) for i in range(len(coords))])
        fhess=nd.Gradient(nd.Gradient(g))(result_min.x)
        grad=nd.Gradient(g)(result_min.x)

        #compute analytically hessian of chi^2 with respect to fit parameters
        if l_params!=1:
            Hess=[[2*fsum(np.dot(fhess[i][a][b]*covinv[i][j],(g(result_min.x)[j]-coords[j][1]))+np.dot(grad[i][a]*covinv[i][j],grad[j][b]) for i in range(l_coords) for j in range(l_coords)) for b in range(l_params)] for a in range(l_params)]
        else:
            Hess=[[2*fsum(np.dot(fhess[i]*covinv[i][j],(g(result_min.x)[j]-coords[j][1]))+np.dot(grad[i]*covinv[i][j],grad[j]) for i in range(l_coords) for j in range(l_coords))]]

        #compute hessian inverse of chi^2
        try:
            #HINV = inv(nd.Hessian(HFUNC)(result_min.x)) #debugging function; compare to numerically calculated Hessian
            HINV=inv(Hess)
        except:
            print "Hessian is singular.  Check your fit function/params."
            print "Hessian:"
            print Hess
            sys.exit(1)

        #compute error matrix; errors in fit parameters are +/- of the diagonal elements
        if l_params!=1:
            delta=[[4*fsum(HINV[a][c]*grad[i][c]*covinv[i][j]*grad[j][d]*HINV[d][b] for i in range(l_coords) for j in range(l_coords) for c in range(l_params) for d in range(l_params)) for b in range(l_params)] for a in range(l_params)]
        else:
            delta=[[4*fsum(HINV[a][c]*grad[i]*covinv[i][j]*grad[j]*HINV[d][b] for i in range(l_coords) for j in range(l_coords) for c in range(l_params) for d in range(l_params)) for b in range(l_params)] for a in range(l_params)]

        #take diagonal elements to be errors in fit parameters, return these
        for i in range(l_params):
            try:
                #approx = sqrt(2*HINV[i][i]) #debugging, compare analytic results to numeric
                #print "percent err between exact param error:",i,", and approx:", abs(approx-sqrt(delta[i][i]))/sqrt(delta[i][i])
                param_err[i] = sqrt(delta[i][i])
            #let rounding errors work for us.  given community experience, probably not mathematically sound
            except ValueError:
                print "***Warning***"
                print "Hessian inverse has negative diagonal entries,"
                print "leading to complex errors in some or all of the"
                print "fit parameters. failing entry:",2*HINV[i][i]
                print "Attempting to continue with entries zero'd below", cutoff
                print 'Debugging info:'
                print '2x Hessian Inverse:'
                print 2*HINV
                print 'Hessian:'
                print HFUN(result_min.x)
                print 'x: (f(x)-y(x))/sigma(x):'
                for j in range(len(covinv)):
                    print j
                    print covinv[j][j]*(fit_func(coords[j][0],result_min.x)-coords[j][1])**2
                flag = 1
                break
        if flag == 1:
            HFUN_CUT=[[HFUN(result_min.x)[j][i] if HFUN(result_min.x)[j][i] > cutoff else 0 for i in range(l_params)] for j in range(l_params)]
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
    return param_err
           
