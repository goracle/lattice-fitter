"""Get parameter error."""
import sys
from math import sqrt, fsum
import numdifftools as nd
from numpy.linalg import inv
import numpy as np
from accupy import kdot

from latfit.config import fit_func
from latfit.config import SCALE
from latfit.config import CUTOFF

from latfit.mathfun.chi_sq import chi_sq

# compute errors in fit parameters


def geterr(result_min, covinv, coords):
    """Get parameter error."""
    if result_min.fun > 0 and result_min.status == 0:

        # #param setup
        # Numeric Hessian of chi^2 (t^2) is unstable.  Use at own risk.
        hfun = nd.Hessian(lambda xrray:
                          chi_sq(xrray, covinv, coords))(result_min.x)
        l_coords = len(coords)
        l_params = len(result_min.x)
        debug_str = [covinv[j][j]*(
            fit_func(coords[j][0], result_min.x)-coords[j][1])**2
                     for j in range(l_coords)]

        # #main
        grad = compute_grad(result_min, coords)

        hess = compute_hess(grad, covinv, l_params, l_coords)

        hinv = compute_hess_inv(hess)

        delta = compute_err(hinv, grad, covinv, l_coords, l_params)

        param_err = diag_delta(delta, debug_str, result_min, hfun, hinv)

    return np.array(param_err)


def compute_grad(result_min, coords, analytic_gradf=None):
    """compute the gradient of the fit function
    """
    # fhess = nd.Gradient(nd.Gradient(g))(result_min.x)
    if analytic_gradf is None:
        def gradf(params):
            """Compute the gradient of the fit function
            at all points in the fit range
            """
            return np.array([
                fit_func(coords[i][0], params) for i in range(len(coords))])
        grad = nd.Gradient(gradf)(result_min.x)

    else:
        grad = np.array([
            analytic_gradf(coords[i][0], result_min.x)
            for i in range(len(coords))])
    return grad


def compute_hess(grad, covinv, l_params, l_coords):
    """compute analytically hessian
    of chi^2 (t^2) with respect to fit parameters
    """
    if l_params != 1:
        # use numeric second deriv of fit fit func.
        # also unstable (and often wrong) and out of date
        # Hess = [[2*fsum(kdot(
        # fhess[i][a][b]*covinv[i][j],
        # (g(result_min.x)[j]-coords[j][1]))
        # +kdot(grad[i][a]*covinv[i][j],
        # grad[j][b]) for i in range(l_coords)
        # for j in range(l_coords)) for
        # b in range(l_params)] for a in range(l_params)]
        hess = [[2*fsum(
            kdot(kdot(grad[i][a], covinv[i][j]), grad[j][b])
            for i in range(l_coords) for j in range(l_coords))
                 for b in range(l_params)] for a in range(l_params)]
    else:
        # see above about instability of fit func hessian.
        # this else block is for one parameter fits.
        # hess = [[2*fsum(
        # kdot(fhess[i]*covinv[i][j],
        # (g(result_min.x)[j]-coords[j][1]))+kdot(grad[i]*covinv[i][j],
        # grad[j]) for i in range(l_coords) for j in range(l_coords))]]
        hess = [[2*fsum(
            kdot(kdot(grad[i], covinv[i][j]), grad[j])
            for i in range(l_coords) for j in range(l_coords))]]
    return hess


def compute_hess_inv(hess, hfun=None):
    """compute hessian inverse of chi^2 (t^2)
    """
    try:
        # debugging function; compare to numerically calculated Hessian
        if hfun is not None:
            hinv = inv(hfun)
        else:
            hinv = inv(hess)
    except np.linalg.linalg.LinAlgError as err:
        if err == 'Singular matrix':
            print("Hessian is singular.",
                  "Check your fit function/params.\nHessian:")
            print(hess)
            sys.exit(1)
    return hinv


def compute_err(hinv, grad, covinv, l_coords, l_params):
    """Compute error matrix; errors in fit parameters are
    +/- of the diagonal elements
    """
    #
    if l_params != 1:
        delta = [[4*fsum(
            hinv[a][c]*kdot(
                kdot(grad[i][c], covinv[i][j]),
                grad[j][d])*hinv[d][b]
            for i in range(l_coords) for j in range(l_coords)
            for c in range(l_params) for d in range(l_params))
                  for b in range(l_params)] for a in range(l_params)]
    else:
        delta = [[4*fsum(
            hinv[a][c]*kdot(
                kdot(grad[i], covinv[i][j]), grad[j])*hinv[d][b]
            for i in range(l_coords) for j in range(l_coords)
            for c in range(l_params) for d in range(l_params))
                  for b in range(l_params)] for a in range(l_params)]
    return delta


def diag_delta(delta, debug_str, result_min, hfun, hinv):
    """Take the diagonal of the delta matrix, return parameter error.
    """
    l_params = len(result_min.x)
    # take diagonal elements to be errors in fit parameters, return these
    param_err = ["err" for i in range(l_params)]
    cutoff = 1/(CUTOFF*SCALE)
    flag = 0
    for i in range(l_params):
        try:
            # debugging, compare analytic results to numeric
            # approx = sqrt(2*hinv[i][i])
            # print("percent err between exact param error:",
            # i,", and approx:",
            # abs(approx-sqrt(delta[i][i]))/sqrt(delta[i][i]))
            param_err[i] = sqrt(delta[i][i])
        # let rounding errors work for us.
        # given community experience, probably not mathematically sound
        except ValueError:
            print("***Warning***")
            print("Hessian inverse has negative diagonal entries,")
            print("leading to complex errors in some or all of the")
            print("fit parameters. failing entry:", 2*hinv[i][i])
            print("Attempting to continue with entries zero'd below", cutoff)
            print("The errors in fit parameters which" +
                  " result shouldn't be taken too seriously.")
            print('Debugging info:')
            print('2x Hessian Inverse:')
            print(2*hinv)
            print('Hessian:')
            print(hfun)
            print('x: (f(x)-y(x))/sigma(x):')
            for j, dstr in enumerate(debug_str):
                print(j)
                print(dstr)
            flag = 1
            break
    if flag == 1:
        hfun_cut = [[hfun[j][i] if hfun[j][i] > cutoff else 0
                     for i in range(l_params)] for j in range(l_params)]
        try:
            hinv = inv(hfun_cut)
        except np.linalg.linalg.LinAlgError as err:
            if err == 'Singular matrix':
                print("Cut Hessian is singular. Cut Hessian:")
                print(hfun_cut)
                sys.exit(1)
            else:
                raise
        for i, _ in enumerate(hinv):
            try:
                param_err[i] = sqrt(2*hinv[i][i])
            except ValueError:
                print("***Warning***")
                print("Hessian inverse has negative diagonal entries,")
                print("leading to complex errors in some or all of the")
                print("fit parameters. failing entry:", 2*hinv[i][i])
                print("Giving up.")
                sys.exit(1)
    return param_err
