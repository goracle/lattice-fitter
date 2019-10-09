"""Various basic linear algebra operations on the GEVP matrices"""
import sys

import scipy
import scipy.linalg
import numpy as np
from accupy import kdot
from sympy import S
from sympy.matrices import Matrix

from latfit.utilities import exactmean as em
from latfit.config import GEVP_DEBUG, LOGFORM, DECREASE_VAR
from latfit.analysis.errorcodes import ImaginaryEigenvalue
from latfit.analysis.errorcodes import PrecisionLossError

def checkgteq0(eigfin):
    """Check to be sure all eigenvalues are greater than 0"""
    for i in eigfin:
        if not np.isnan(i):
            assert i >= 0, "negative eigenvalue found:"+str(eigfin)

def enforce_hermiticity(gevp_mat):
    """C->(C+C^dagger)/2"""
    gevp_mat = np.asarray(gevp_mat, dtype=np.complex128)
    return em.acsum([np.conj(gevp_mat).T, gevp_mat])/2


def finaleval_imag_check(eigvals):
    """At this point, there should be no imgaginary eigenvalues"""
    eigfin = np.zeros((len(eigvals)), dtype=np.float)
    for i, j in enumerate(eigvals):
        if abs(j.imag) < 1e-8 or np.isnan(j.imag):
            eigfin[i] = eigvals[i].real
        else:
            print("Eigenvalue=", j)
            raise ImaginaryEigenvalue
    return eigfin

def printevecs(c_lhs, c_rhs, eigvals, evecs):
    """Debug function, prints diagnostic info"""
    print("start solve")
    print("lhs=", c_lhs)
    print("rhs=", c_rhs)
    for i, j in enumerate(eigvals):
        print("eigval #", i, "=", j, "evec #", i, "=", evecs[:, i])
        assert np.all(evecs[:, i] == evecs.T[i])
    print("end solve")


def make_avg_zero(arr):
    """Subtract the average of the array to make the new average 0"""
    avg = em.acmean(arr, axis=0)
    ret = arr - avg
    return ret

# mostly useless (only a check)
def propagate_nans(blk):
    """Propagate nan's"""
    nandim = np.zeros(blk[0].shape)
    for _, sample in enumerate(blk):
        for i, val in enumerate(sample):
            if np.isnan(val):
                nandim[i] = np.nan
    for config, _ in enumerate(blk):
        blk[config] += nandim
    return blk

def jkerr(arr, axis=0):
    """Calculate the jackknife error in array arr"""
    ret = em.acstd(arr, axis=axis)*np.sqrt(len(arr)-1)
    return ret

def sterr(arr, axis=0, jack=True):
    """Calculate the standard error in array arr"""
    if jack:
        ret = jkerr(arr, axis=axis)
    else:
        ret = em.acstd(arr, axis=axis, ddof=1)/np.sqrt(len(arr))
    return ret

def checkherm(carr):
    """Check hermiticity of gevp matrix"""
    try:
        assert np.allclose(np.matrix(carr).H, carr, rtol=1e-12)
    except AssertionError:
        print("hermiticity enforcement failed.")
        print(carr)
        sys.exit(1)
    except TypeError:
        print("hermiticity enforcement failed.")
        print(carr)
        sys.exit(1)

def removerowcol(cmat, idx):
    """Delete the idx'th row and column"""
    return np.delete(np.delete(cmat, idx, axis=0), idx, axis=1)

def is_pos_semidef(cmat):
    """Check for positive semi-definiteness"""
    return np.all(np.linalg.eigvals(cmat) >= 0)

def defsign(cmat):
    """Check for definite sign (pos def or neg def)"""
    evals = np.linalg.eigvals(cmat)
    if np.all(evals > 0):
        ret = 1
    elif np.all(evals < 0):
        ret = -1
    elif np.all(np.real(evals) == 0) and np.all(np.imag(evals) == 0):
        ret = 0
    else:
        print("eigenvalues are not all the same sign:", str(evals))
        sys.exit(1)
        ret = False
    return ret


def log_matrix(cmat, check=False):
    """Take the log of the matrix"""
    assert None, "bad idea; introduces systematic error"
    if check:
        try:
            assert is_pos_semidef(cmat), \
                "input matrix is not positive semi-definite."
        except AssertionError:
            print(cmat)
            print("matrix is not positive semi-definite.")
            sys.exit(1)
    ret = scipy.linalg.logm(cmat)
    assert np.allclose(cmat, scipy.linalg.expm(ret), rtol=1e-8)
    return ret

def cmatdot(cmat, vec, transp=False):
    """Dot gevp matrix into vec on rhs if not transp"""
    cmat = np.asarray(cmat)
    cmat = cmat.T if transp else cmat
    vec = np.asarray(vec)
    ret = np.zeros(vec.shape)
    for i, row in enumerate(cmat):
        tosum = []
        for j, item in enumerate(row):
            tosum.append(item*vec[j])
        ret[i] = em.acsum(tosum)
    return ret

def bracket(evec, cmat):
    """ form v* . cmat . v """
    checkherm(cmat)
    cmat += np.eye(len(cmat))*1e-11
    right = cmatdot(cmat, evec)
    retsum = []
    for i, j in zip(np.conj(evec), right):
        retsum.append(i*j/2)
        retsum.append(np.conj(i*j)/2)
    ret = em.acsum(np.asarray(retsum, dtype=np.complex128))
    assert ret != 0
    return ret

def convtosmat(cmat):
    """Convert numpy matrix to sympy matrix
    for high precision calculation
    """
    ll1 = len(cmat)
    cmat += np.eye(ll1)*1e-6
    smat = [[S(str(cmat[i][j])) for i in range(ll1)] for j in range(ll1)]
    mmat = Matrix(smat)
    return mmat

def makeneg(val):
    """make a value negative"""
    val = drop0imag(val)
    if hasattr(val, '__iter__') and np.asarray(val).shape:
        if all(val < 0):
            ret = val
        elif any(val < 0):
            ret = val
        else:
            ret = np.asarray(val)*(-1)
        assert all(ret < 0), "bug"
    else:
        ret = val if val < 0 else -1*val
        assert ret <= 0, "val="+str(val)
    return ret

def drop0imag(val):
    """Get rid of complex type if the imaginary part is 0"""
    ret = val
    if isinstance(val, complex):
        if val.imag == 0:
            ret = val.real
    if hasattr(val, '__iter__') and np.asarray(val).shape:
        if np.all(np.imag(val) == 0):
            ret = np.real(val)
    return ret

def propnan(vals):
    """propagate a nan in a complex value"""
    if hasattr(vals, '__iter__') and np.asarray(vals).shape:
        for i, val in enumerate(vals):
            if np.isnan(val) and np.imag(val) != 0:
                vals[i] = np.nan+2j*np.nan
    else:
        if np.isnan(vals) and np.imag(vals) != 0:
            vals = np.nan+2j*np.nan
    return vals

def all0imag_ignorenan(vals):
    """check if all values
    have 0 imaginary piece or are nan
    """
    ret = True
    if hasattr(vals, '__iter__') and np.asarray(vals).shape:
        for _, val in enumerate(vals):
            if np.isnan(val):
                continue
            if abs(np.imag(val)) > 1e-8 and not np.isnan(np.imag(val)):
                ret = False
    else:
        val = vals
        if abs(np.imag(val)) > 1e-8 and not np.isnan(val):
            ret = False
    return ret

def inflate_with_nan(dimops, dimops_orig, eigvals, eliminated_operators):
    """inflate number of evals with nan's to match dimensions
    """
    for _ in range(dimops_orig-dimops):
        if dimops == 0:
            break
        eigvals = np.append(eigvals, np.nan)
    eigvals = propnan(eigvals)
    if eliminated_operators:
        #print(sorted(eliminated_operators))
        assert np.count_nonzero(np.isnan(eigvals)) >= len(
            eliminated_operators), "deletion mismatch."
    assert len(eigvals) == dimops_orig, "eigenvalue shape extension needed"
    return eigvals

def degenerate_subspace_check(evecs_mean_t):
    """If we are in a degenerate subspace,
    average norm of evecs should be far from 1
    """
    for evec in evecs_mean_t:
        evec = drop0imag(evec)
        dotprod = np.dot(np.conj(evec), evec)
        assert np.allclose(dotprod, 1.0, rtol=1e-8),\
            str(dotprod)
    if GEVP_DEBUG:
        print("evecs of avg gevp",
              np.real(evecs_mean_t))


def sortevals(evals):
    """Sort eigenvalues in order of increasing energy"""
    evals = list(evals)
    ret = evals
    ind = []
    for i, val in enumerate(evals):
        if val < 0 and LOGFORM:
            ret[i] += np.inf
            ind.append(i)
    ret = np.array(sorted(ret, reverse=False if LOGFORM else True))
    for i, _ in enumerate(ind):
        ret[-(i+1)] = -1
    return ret

def norms(evecs):
    """Get norms of evecs"""
    ret = []
    for i in evecs:
        ret.append(kdot(i, i))
    ret = np.asarray(ret)
    return ret

def variance_reduction(orig, avg, decrease_var=DECREASE_VAR):
    """
    apply y->(y_i-<y>)*decrease_var+<y>
    """
    assert np.asarray(avg).shape != (0,), str(avg)
    orig = np.asarray(orig)
    if hasattr(orig, '__iter__') and np.asarray(orig).shape:
        assert hasattr(avg, '__iter__') and np.asarray(avg).shape or len(
            orig.shape) == 1, "dimension mismatch"
        if len(orig.shape) == 1:
            for i, j in enumerate(orig):
                if np.isnan(j):
                    orig[i] = np.nan
                    if np.asarray(orig).shape == np.asarray(avg).shape:
                        avg[i] = np.nan
                    else:
                        avg = np.nan
    else:
        assert not np.isnan(orig+avg), "nan found"
    ret = (orig-avg)*decrease_var+avg
    check = (ret-avg)/decrease_var+avg
    try:
        assert np.allclose(check, orig, rtol=1e-8, equal_nan=True), \
            "precision loss detected:"+str(orig)+" "+\
            str(check)+" "+str(decrease_var)
    except AssertionError:
        print('avg =', avg)
        print('ret =', ret)
        print('check =', check)
        print('orig =', orig)
        print("precision loss detected, orig != check")
        raise PrecisionLossError
    return ret
