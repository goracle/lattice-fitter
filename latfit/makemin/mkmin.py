"""Minimizes chi^2 (t^2)"""
import sys
from collections import namedtuple
from itertools import product
from numdifftools import Jacobian # , Hessian
from scipy.optimize import minimize
# from scipy.optimize import curve_fit
# from iminuit import Minuit
from iminuit import minimize as minit
import numpy as np

from latfit.config import METHOD
# from latfit.mathfun.chi_sq import chi_sq
from latfit.config import START_PARAMS, KICK_DELTA
from latfit.config import BINDS, VERBOSE
from latfit.config import JACKKNIFE_FIT
# from latfit.config import MINTOL
from latfit.config import SYSTEMATIC_EST
from latfit.config import GRAD, EFF_MASS
from latfit.analysis.errorcodes import NegChisq
from latfit.analysis.errorcodes import EnergySortError, PrecisionLossError
import latfit.config
import latfit.mathfun.chi_sq as chi

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

KICK = False

def dealloc_chi():
    """Reset the precomputed ranges"""
    chi.RCORD = None
    chi.COUNT = None
    chi.SYMRANGE = None
    prealloc_chi.allocd = False

def prealloc_chi(covinv, coords):
    """Preallocate some variables for speedup of chi^2 eval
    perform checks
    """
    lcord = len(coords)
    chi.RCORD = np.arange(lcord)
    chi.SYMRANGE = sym_range(covinv, lcord)
    chi.COUNT = len(chi.SYMRANGE)
    chi.PRODRANGE = list(product(range(lcord), range(lcord)))
    if GRAD is not None:
        GRAD.PRODRANGE = chi.PRODRANGE
    covinv = np.asarray(covinv)
    assert covinv.shape[0] == covinv.shape[1], str(
        covinv.shape)+" "+str(coords)
    assert covinv.shape[0] == lcord, str(covinv.shape)+" "+str(coords)
    prealloc_chi.allocd = True
prealloc_chi.allocd = False

def sym_range(covinv, lcord):
    """Create iterable for symmetric i,j indices of covinv"""
    ret = []
    for i in range(lcord):
        for j in np.arange(i, lcord):
            if not np.any(covinv[i, j]):
                continue
            ret.append((i, j))
    return ret

def sym_norm(covinv):
    """Divide diagonal by 2 to prevent overcounting"""
    for i in chi.RCORD:
        covinv[i, i] /= 2
    return covinv

def check_covinv(covinv):
    """Check inverse covariance matrix"""
    for i in chi.RCORD:
        for j in chi.RCORD:
            if i <= j:
                continue
            comp1 = covinv[i][j]
            comp2 = np.transpose(covinv[j][i])
            try:
                assert np.allclose(comp1, comp2, rtol=1e-8)
            except AssertionError:
                err = str(covinv[i][j])+" "+str(covinv[j][i])
                print(i, j)
                print(err)
                raise PrecisionLossError


SPARAMS = list(START_PARAMS)
PARAMS = None

@PROFILE
def mkmin(covinv, coords, method=METHOD):
    """Minimization of chi^2 (t^2) section of fitter.
    Return minimized result.
    """
    status = 1
    status2 = 1
    kick = False
    kick = True if latfit.config.BOOTSTRAP else kick
    count = 15 if kick else 1 # try 10 times to get convergence (abitrary)
    kick = False
    while (status or status2) and count:
        assert count >= 0, str(count)
        res_min = mkmin_loop(covinv, coords, method, kick=kick)
        break
        status = res_min.status
        count -= 1
        if status:
            kick = True
            kick_params()
        elif count:
            try:
                getenergies(res_min)
                status2 = 0
            except EnergySortError:
                status2 = 1
                kick = True
                kick_params()
    return res_min

def kick_params(kick_delta=KICK_DELTA):
    """Try to give the start params some small kick
    in case we don't get convergence
    but we are only stuck in a local minimum
    kick delta determines the kick strength
    """
    skew = np.asarray(START_PARAMS) - np.asarray(SPARAMS)
    #print("kicking start params; currently:", SPARAMS)
    if not np.any(skew):
        skew = np.ones_like(START_PARAMS)
    for i, _ in enumerate(SPARAMS):
        noise = np.random.normal()
        assert None, "should not be kicking for reproduction"
        SPARAMS[i] += skew[i]*kick_delta*noise
    #print("after kick:", SPARAMS)

def getenergies(result_min):
    """Get energies.  Check if they are missorted"""
    PARAMS.energyind = 2 if PARAMS.energyind is None else PARAMS.energyind
    arr = np.asarray(result_min.x)
    if len(arr) != PARAMS.dimops and EFF_MASS:
        ret = arr[0::PARAMS.energyind][:-1]
    else:
        ret = arr
    for i, j in zip(sorted(list(ret)), ret):
        if i != j:
            if VERBOSE:
                print("mis-sorted energies:", ret)
            if not latfit.config.BOOTSTRAP:
                raise EnergySortError
    return ret


@PROFILE
def mkmin_loop(covinv, coords, method, kick=False):
    """Inner loop part
    """
    if not kick:
        covinv = sym_norm(covinv) # only do this once per mkmin call
    check_covinv(covinv)
    start_params = [*SPARAMS, *SPARAMS,
                    *SPARAMS] if SYSTEMATIC_EST else SPARAMS
    if method not in set(['L-BFGS-B', 'minuit']):
        if latfit.config.MINTOL:
            options = {'maxiter': 10000, 'maxfev': 10000,
                       'xatol': 0.00000001, 'fatol': 0.00000001}
        else:
            options = {}
        res_min = minimize(chi.chi_sq, start_params, (covinv, coords),
                           jac=GRAD,
                           method=method,
                           options=options)
        #else:
        #    res_min = minimize(chi.chi_sq, start_params, (covinv, coords),
        #                       method=METHOD)
        # options={'disp': True})
        #'maxiter': 10000,
        #'maxfev': 10000})
        # method = 'BFGS'
        # method = 'L-BFGS-B'
        # bounds = BINDS
        # options = {'disp': True}
    elif method in set(['L-BFGS-B']):
        if latfit.config.MINTOL:
            options = {}
        else:
            options = {}
        try:
            res_min = minimize(chi.chi_sq, start_params, (covinv, coords),
                               jac=GRAD,
                               method=method, bounds=BINDS,
                               options=options)
        except FloatingPointError:
            print('floating point error')
            print('covinv:')
            print(covinv)
            print('coords')
            print(coords)
            sys.exit(1)
        res_min = dict(res_min)
        # res_min['x'] = delta_add_energies(res_min['x'])
        res_min = convert_to_namedtuple(res_min)

    # print "minimized params = ", res_min.x
    elif 'minuit' in method:
        options = {}
        try:
            res_min = minit(chi.chi_sq, start_params, (covinv, coords),
                            method=method, bounds=BINDS,
                            jac=None,
                            options=options)
            status = res_min.minuit.get_fmin().is_valid
            status = 0 if res_min.success else 1
            res_min.status = status
        except RuntimeError:
            status = 1
            res_min = {}
            res_min['status'] = 1
        # res_min['x'] = delta_add_energies(res_min['x'])
        res_min = convert_to_namedtuple(res_min)
        # insert string here
    if not JACKKNIFE_FIT:
        print("number of iterations = ", res_min.nit)
        print("successfully minimized = ", res_min.success)
        print("status of optimizer = ", res_min.status)
        print("message of optimizer = ", res_min.message)
    # print "chi^2 minimized = ", res_min.fun
    # print "chi^2 minimized check = ", chi_sq(res_min.x, covinv, coords)
    # print covinv
    if (res_min.status and latfit.config.BOOTSTRAP) or False:
        def fun_der(xst, covinv, coords):
            return Jacobian(
                lambda xarg: chi.chi_sq(xarg, covinv, coords))(xst).ravel()
        print("boostrap debug")
        #print("covinv =", covinv)
        print("covinv.shape", covinv.shape)
        print("coords =", coords)
        print("start_params =", start_params)
        # covinv = np.ones_like(np.zeros(covinv.shape), dtype=np.float)
        #for i in range(len(coords)):
        #    coords[i][1] = np.ones_like(np.zeros(coords[i][1].shape), dtype=np.float)
        # start_params = np.ones_like(np.zeros(len(start_params)), dtype=np.float)
        print("chisq =", chi.chi_sq(start_params, covinv, coords))
        if GRAD is not None:
            print("grad =", GRAD(start_params, covinv, coords))
            print("num grad =", fun_der(start_params, covinv, coords))
        # sys.exit()
    if not res_min.status:
        if res_min.fun < 0:
            print("negative chi^2 found:", res_min.fun)
            print("result =", res_min.x)
            print("chi^2 (t^2; check) =",
                  chi.chi_sq(res_min.x, covinv, coords))
            print("covinv:", covinv)
            sys.exit(1)
            raise NegChisq
    # print "degrees of freedom = ", dimcov-len(start_params)
    # print "chi^2 reduced = ", res_min.fun/(dimcov-len(start_params))
    return prune_res_min(res_min)

def delta_add_energies(result_min):
    """If we restrict the energies to be the previous energy plus some
    positive delta, we end up with correctly sorted energies.  This
    function takes the initial energy and deltas
    and returns the proper energies"""
    tot = 0
    ret = []
    for i, delta in enumerate(result_min):
        if i % 2 or i == len(result_min)-1:
            continue
        tot += delta
        assert delta > 0, str(delta)+" "+str(result_min)+" "+str(tot)
        ret.append(tot)
        ret.append(result_min[i+1])
        if i:
            assert i > 1, str(i)
            assert ret[i] > ret[i-2], str(
                result_min)+" "+str(i)+" "+str(ret)
    ret.append(result_min[-1])
    assert len(ret) == len(result_min)
    ret = np.array(ret)
    return ret


@PROFILE
def convert_to_namedtuple(dictionary):
    """Convert dictionary to named tuple"""
    return namedtuple('min', dictionary.keys())(**dictionary)



if SYSTEMATIC_EST:
    @PROFILE
    def prune_res_min(res_min):
        """Get rid of systematic error information"""
        print([res_min.x[len(START_PARAMS):][2*i+1] for i in range(len(START_PARAMS))])
        res_min.x = np.array(res_min.x)[:len(START_PARAMS)]
        return res_min
else:
    @PROFILE
    def prune_res_min(res_min):
        """pass"""
        return res_min
