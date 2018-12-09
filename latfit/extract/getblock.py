""""Get the data block."""
import sys
from collections import deque
from math import fsum
from sympy import exp, N, S
from sympy.matrices import Matrix
import scipy
import scipy.linalg
from scipy import linalg
from scipy import stats
from matplotlib.mlab import PCA
import numpy as np
import h5py

from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.mathfun.binconf import binconf
from latfit.extract.proc_line import proc_line
from latfit.jackknife_fit import jack_mean_err

from latfit.config import EFF_MASS
from latfit.config import GEVP, DELETE_NEGATIVE_OPERATORS
from latfit.config import ELIM_JKCONF_LIST
from latfit.config import OPERATOR_NORMS, GEVP_DEBUG, USE_LATE_TIMES
from latfit.config import BINNUM, LOGFORM
from latfit.config import STYPE
from latfit.config import PIONRATIO, ADD_CONST_VEC
from latfit.config import MATRIX_SUBTRACTION
from latfit.config import DECREASE_VAR
from latfit.config import HINTS_ELIM
from latfit.config import REINFLATE_BEFORE_LOG

from mpi4py import MPI

NORMS = [[1 for _ in range(len(OPERATOR_NORMS))]
         for _ in range(len(OPERATOR_NORMS))]

for i, norm in enumerate(OPERATOR_NORMS):
    for j, norm2 in enumerate(OPERATOR_NORMS):
        NORMS[i][j] = norm*norm2

MPIRANK = MPI.COMM_WORLD.rank

if MATRIX_SUBTRACTION and GEVP:
    ADD_CONST_VEC = [0 for i in ADD_CONST_VEC]

XMAX = 999

if PIONRATIO and GEVP:
    PIONSTR = ['pioncorrChk_mom'+str(i)+'unit'+\
               ('s' if i != 1 else '') for i in range(2)]
    PION = []
    for istr in PIONSTR:
        print("using pion correlator:", istr)
        GN1 = h5py.File(istr+'.jkdat', 'r')
        PION.append(np.array(GN1[istr]))
    PION = np.array(PION)

#if STYPE == 'hdf5':
def getline_loc(filetup, num):
    """The file tup is actually already a numpy array.
    This function pretends to get the line from an ascii file
    """
    try:
        complex(filetup[num-1])
    except TypeError:
        print("***ERROR***")
        print("Expecting an array; in getblock")
        print(filetup[num-1], "should be array of floats")
        sys.exit(1)
    return filetup[num-1]
#else:
if 1 > 2:
    def getline_loc_bad(filetup, num):
        """This function does get the line from the ascii file
        it is a simple wrapper for linecache.getline
        proc_folder now turns this into a numpy array preemptively,
        so we don't need this function anymore
        (that code change makes things more uniform)
        """
        return filetup[num-1]
       # return getline(filetup, num)


def readin_gevp_matrices(file_tup, num_configs, decrease_var=DECREASE_VAR):
    """Read in the GEVP matrix
    """
    decrease_var = 0 if decrease_var is None else decrease_var
    dimops = len(file_tup)
    cmat = np.zeros((num_configs, dimops, dimops), dtype=float)
    for num in range(num_configs):
        for opa in range(dimops):
            for opb in range(dimops):
                cmat[num][opa][opb] = proc_line(
                    getline_loc(file_tup[opa][opb], num+1),
                    file_tup[opa][opb])*NORMS[opa][opb]
        cmat[num] = enforce_hermiticity(cmat[num])
        checkherm(cmat[num])
    mean = np.mean(cmat, axis=0)
    checkherm(mean)
    if decrease_var != 0:
        cmat = variance_reduction(cmat, mean)
    return np.asarray(cmat), np.asarray(mean)

def checkherm(carr):
    """Check hermiticity of gevp matrix"""
    assert np.allclose(np.matrix(carr).H, carr, rtol=1e-12),\
        "hermiticity enforcement failed."

def removerowcol(cmat, idx):
    """Delete the idx'th row and column"""
    return np.delete(np.delete(cmat, idx, axis=0), idx, axis=1)

def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def log_matrix(cmat, check=False):
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

# from here
# https://github.com/numpy/numpy/issues/8786
def kahan_sum(a, axis=0):
    a = np.asarray(a)
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/a/42817610/353337
        y = a[(slice(None), ) * axis + (i, )] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()
    return s

def cmatdot(cmat, vec, transp=False):
    cmat = np.asarray(cmat)
    cmat = cmat.T if transp else cmat
    vec = np.asarray(vec)
    ret = np.zeros(vec.shape)
    for i, row in enumerate(cmat):
        tosum = []
        for j, item in enumerate(row):
            tosum.append(item*vec[j])
        ret[i] = kahan_sum(tosum)
    return ret

def bracket(evec, cmat):
    """ form v* . cmat . v """
    checkherm(cmat)
    right = cmatdot(cmat, evec)
    retsum = []
    for i,j in zip(np.conj(evec), right):
        retsum.append(i*j/2)
        retsum.append(np.conj(i*j)/2)
    ret = kahan_sum(retsum)
    return ret

def convtosmat(cmat):
    """Convert numpy matrix to sympy matrix
    for high precision calculation
    """
    ll1 = len(cmat)
    smat=[[S(str(cmat[i][j])) for i in range(ll1)] for j in range(ll1)]
    mmat = Matrix(smat)
    return mmat

def sym_evals_gevp(c_lhs, c_rhs):
    """Do high precision solve of GEVP"""
    c_lhs = convtosmat(c_lhs)
    c_rhs = convtosmat(c_rhs)
    res = (c_rhs**-1*c_lhs).eigenvals()
    rres = []
    for i in res:
        rres.append(N(i))
    nums = []
    for num in rres:
        nums.append(np.complex(num))
    for i, num in enumerate(nums):
        if np.abs(np.imag(num)) < 1e-8:
            nums[i] = np.real(num)
    return nums

def calleig(c_lhs, c_rhs=None):
    """Actual call to scipy.linalg.eig"""
    flag = False
    if c_rhs is not None and LOGFORM:
        rhs = log_matrix(c_rhs)
        lhs = log_matrix(c_lhs)
        c_lhs_check = np.dot(linalg.inv(c_rhs), c_lhs)
        c_lhs = rhs-lhs
        try:
            assert np.allclose(linalg.eigvals(c_lhs_check),
                               linalg.eigvals(linalg.expm(-c_lhs)))
        except AssertionError:
            print(np.log(linalg.eigvals(c_lhs_check)))
            print(linalg.eigvals(-c_lhs))
            print(c_lhs_check)
            print(linalg.expm(c_lhs))
            sys.exit(1)
        assert np.allclose(c_lhs+lhs, rhs, rtol=1e-12)
        c_rhs = None
        flag = True
    elif LOGFORM:
        c_lhs = -1*log_matrix(c_lhs, check=True)
        flag = True
    if LOGFORM:
        assert is_pos_semidef(c_lhs), "not positive semi-definite."
        c_lhs = enforce_hermiticity(c_lhs)
        assert is_pos_semidef(c_lhs), "not positive semi-definite."
        eigenvals, evecs = scipy.linalg.eig(c_lhs, c_rhs,
                                            overwrite_a=False,
                                            overwrite_b=False,
                                            check_finite=True)
    else:
        checkherm(c_lhs)
        checkherm(c_rhs)
        eigenvals, evecs = scipy.linalg.eig(c_lhs, c_rhs,
                                            overwrite_a=False,
                                            overwrite_b=False,
                                            check_finite=True)
    for i, (eval1, evec) in enumerate(zip(eigenvals, evecs.T)):
        for j,k in zip(cmatdot(c_lhs, evec), cmatdot(c_rhs, evec)):
            try:
                assert np.allclose(eval1, j/k, rtol=1e-8)
            except AssertionError:
                print("GEVP solution does not solve GEVP")
                print(j,k,j/k,eval1)
                print("lhs, rhs")
                print(np.array2string(c_lhs, separator=','))
                print(np.array2string(c_rhs, separator=','))
                print('trying symbolic extended precision calc of eigenvalues')
                sevals = sym_evals_gevp(c_lhs, c_rhs)
                print(sevals)
                print(eigenvals)
                # assert np.allclose(sortevals(eigenvals), sevals, rtol=1e-8)
                sys.exit(1)
        eval_check = bracket(evec, c_lhs)/bracket(evec, c_rhs)
        try:
            assert np.allclose(eval_check, eval1, rtol=1e-10)
        except AssertionError:
            print("Eigenvalue consistency check failed."+\
                  "  ratio and eigenvalue not equal.")
            print("bracket lhs, bracket rhs, ratio, eval")
            print(bracket(evec,c_lhs), bracket(evec,c_rhs),
                  bracket(evec,c_lhs)/bracket(evec,c_rhs), eval1)
            sys.exit(1)
    if flag:
        if all(np.imag(eigenvals) < 1e-8):
            pass
        else:
            print("non-negligible imaginary eigenvalues found")
            print("eigenvals:")
            print(eigenvals)
            print("log lhs:")
            print(lhs)
            print("log rhs:")
            print(rhs)
            sys.exit(0)
        eigenvals = np.real(eigenvals)
    eigenvals = sortevals(eigenvals)
    return eigenvals, evecs

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

def makeneg(val):
    """make a value negative"""
    val = drop0imag(val)
    if hasattr(val, '__iter__'):
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
    ret = val
    if isinstance(val, complex):
        if val.imag == 0:
            ret = val.real
    if hasattr(val, '__iter__'):
        if isinstance(val[0], complex):
            if all(val.imag) == 0:
                ret = val.real
    return ret 
            
def propnan(vals):
    """propagate a nan in a complex value"""
    if hasattr(vals, '__iter__'):
        for i,val in enumerate(vals):
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
    if hasattr(vals, '__iter__'):
        for i, val in enumerate(vals):
            if np.isnan(val):
                continue
            if np.imag(val) != 0 and not np.isnan(np.imag(val)):
                ret = False
    else:
        val = vals
        if np.imag(val) != 0 and not np.isnan(val):
            ret = False
    return ret


def solve_gevp(c_lhs, c_rhs=None):
    """Solve the GEVP"""
    dimops_orig = len(c_lhs)
    dimops = len(c_lhs)
    dimremaining, toelim  = nexthint()
    eigvals, evecs = calleig(c_lhs, c_rhs)
    remaining_operator_indices = set(range(dimops))
    # make eval negative to eliminate it
    if dimops == dimremaining:
        eigvals[toelim] = makeneg(eigvals[toelim])
    if solve_gevp.mean is not None:
        assert 1/DECREASE_VAR > 1,\
            "variance is being reduced, but it should be increased here."
        eigvals = variance_reduction(eigvals, solve_gevp.mean,
                                     1/DECREASE_VAR)
        eigvals = sortevals(eigvals)

    eliminated_operators = set()
    #allowedeliminations(reset=True)
    while any(eigvals < 0):

        if not DELETE_NEGATIVE_OPERATORS:
            break

        # indexing updates
        assert isinstance(dimremaining, int), "bug"
        assert isinstance(toelim, int), "bug"
        dimops -= 1
        dimremaining, toelim = nexthint()
        if dimops == 0:
            #print("dimension reduction technique exhausted.")
            eigvals = np.array([np.nan]*dimops_orig)
            break
        count = 0
        dimdeldict = {}

        # try to eliminate different operators to remove negative eigenvalues
        loop = list(range(dimops+1))
        if len(loop) > 3:
            pass
            #loop[1], loop[3] = loop[3], loop[0]
        for dimdel in loop:
            if dimdel == 0 and toelim < 0: # heuristic, override with hint
                continue
            c_lhs_temp = removerowcol(c_lhs, dimdel)
            c_rhs_temp = removerowcol(
                c_rhs, dimdel) if c_rhs is not None else c_rhs
            eigvals, evecs = calleig(c_lhs_temp, c_rhs_temp)
            if solve_gevp.mean is not None:
                eigvals = variance_reduction(
                    eigvals, solve_gevp.mean[:dimops], 1/DECREASE_VAR)
                eigvals = sortevals(eigvals)

            if dimremaining == dimops:
                eigvals[toelim] = makeneg(eigvals[toelim])
            # count number of non-negative eigenvalues
            count = max(np.count_nonzero(eigvals > 0), count)
            # store dimension deletion leading to this count
            dimdeldict[count] = dimdel
        # find the maximum number of non-negative entries
        if dimdeldict:
            dimdel = dimdeldict[max([count for count in dimdeldict])]
            c_lhs = removerowcol(c_lhs, dimdel)
            c_rhs = removerowcol(c_rhs,
                                 dimdel) if c_rhs is not None else c_rhs
            orig_index = sorted(list(remaining_operator_indices))[dimdel]
            eliminated_operators.add(orig_index)
            remaining_operator_indices.remove(orig_index)
        # do final solve in truncated basis
        eigvals, evecs = calleig(c_lhs, c_rhs)
        if solve_gevp.mean is not None:
            eigvals = variance_reduction(eigvals, solve_gevp.mean[:dimops],
                                         1/DECREASE_VAR)
            eigvals = sortevals(eigvals)
        if dimremaining == dimops:
            eigvals[toelim] = makeneg(eigvals[toelim])
    if allowedeliminations() is not None:
        if not eliminated_operators.issubset(allowedeliminations()):
            allowedeliminations(eliminated_operators)
            eigvals[0] = -1
        #for dimdel in allowedeliminations():
        #    assert dimdel, "we should not be removing ground operator"
        #    c_lhs = removerowcol(c_lhs, dimdel)
        #    c_rhs = removerowcol(c_rhs, dimdel)
        #eigvals, evecs = calleig(c_lhs, c_rhs)
        #assert all(eigvals < 1), "negative energies found"
        #dimops = len(eigvals)
    else:
        allowedeliminations(eliminated_operators)
    try:
        assert len(eliminated_operators) == \
            dimops_orig-dimops or dimops == 0
    except AssertionError:
        print("deletion count is wrong.")
        print('dimops, dimops_orig, eliminated_operators:')
        print(dimops, dimops_orig, eliminated_operators)
        print("toelim", toelim, 'dimremaining', dimremaining)
        print('evals', eigvals)
        print('clhs', c_lhs)
        sys.exit(1)
    # inflate number of evals with nan's to match dimensions
    for _ in range(dimops_orig-dimops):
        if dimops == 0:
            break
        eigvals = np.append(eigvals, np.nan)
            #if nexthint.idx > 0:
            #    print(dimremaining, toelim)
            #    sys.exit(0)
    assert len(eigvals) == dimops_orig
    if eliminated_operators:
        #print(sorted(eliminated_operators))
        assert np.count_nonzero(np.isnan(eigvals)) >= len(
            eliminated_operators), "deletion mismatch."
    assert len(eigvals) == dimops_orig, "eigenvalue shape extension needed"
    nexthint(0)
    eigvals = propnan(eigvals)
    return eigvals, evecs
solve_gevp.mean = None
solve_gevp.hint = None

def allowedeliminations(newelim=None, reset=False):
    """The complete list of allowed eliminations"""
    if reset:
        allowedeliminations.elims = None
    else:
        allowedeliminations.elims = set(newelim) if isinstance(
            newelim, set) and newelim else allowedeliminations.elims
    return allowedeliminations.elims
allowedeliminations.elims = None
 

def nexthint(idx=None):
    ret = (-1, -2)
    if isinstance(solve_gevp.hint, tuple):
        ret = solve_gevp.hint
    elif isinstance(solve_gevp.hint, list):
        ret = solve_gevp.hint[nexthint.idx]
        if len(solve_gevp.hint) > nexthint.idx+1:
            nexthint.idx += 1
    else:
        assert solve_gevp.hint is None, "inconsistency in assigning variable"
    dimremaining, toelim = ret
    assert toelim < dimremaining, "index error"
    assert isinstance(dimremaining, int), "bug"
    assert isinstance(toelim, int), "bug"
    nexthint.idx = idx if idx is not None else nexthint.idx
    return ret
nexthint.idx = 0


def get_eigvals(c_lhs, c_rhs, overb=False, print_evecs=False,
                commnorm=False):
    """get the nth generalized eigenvalue from matrices of files
    file_tup_lhs, file_tup_rhs
    optionally, overwrite the rhs matrix we get
    if we don't need it anymore (overb=True)
    """
    checkherm(c_lhs)
    checkherm(c_rhs)
    overb = False # completely unnecessary and dangerous speedup
    print_evecs = False if not GEVP_DEBUG else print_evecs
    if not get_eigvals.sent and print_evecs:
        print("First solve, so printing norms "+\
              "which are multiplied onto GEVP entries.")
        print("e.g. C(t)_ij -> Norms[i][j]*C(t)_ij")
        print("Norms=", NORMS)
        get_eigvals.sent = True
    eigvals, evecs = solve_gevp(c_lhs, c_rhs)
    #checkgteq0(eigvals)
    dimops = len(c_lhs)
    late = False if all0imag_ignorenan(eigvals) else True
    assert not late, "imaginary eigenvalues found."
    skip_late = False
    try:
        c_rhs_inv = linalg.inv(c_rhs)
        # compute commutator divided by norm
        # to see how close rhs and lhs bases
        try:
            commutator_norms = (np.dot(c_rhs_inv, c_lhs)-np.dot(
                c_lhs, c_rhs_inv))
        except FloatingPointError:
            print("bad denominator:")
            print(np.linalg.norm(c_rhs_inv))
            print(np.linalg.norm(c_lhs))
            print(c_lhs)
            raise FloatingPointError
        assert np.allclose(np.dot(c_rhs_inv, c_rhs),
                           np.eye(dimops), rtol=1e-8),\
                           "Bad C_rhs inverse. Numerically unstable."
        assert np.allclose(np.matrix(c_rhs_inv).H, c_rhs_inv,
                           rtol=1e-8),\
                           "Inverse failed (result is not hermite)."
        c_lhs_new = (np.dot(c_rhs_inv, c_lhs)+np.dot(c_lhs, c_rhs_inv))/2
        commutator_norm = np.linalg.norm(commutator_norms)
        try:
            assert np.allclose(np.matrix(c_lhs_new).H, c_lhs_new, rtol=1e-8)
        except AssertionError:
            print("Correction to hermitian matrix failed.")
            print("commutator norm =", commutator_norm)
            print(c_lhs_new.T)
            print(c_lhs_new)
            print("printing difference in rows:")
            for l, m in zip(c_lhs_new.T, c_lhs_new):
                print(l-m)
            sys.exit(1)
    except np.linalg.linalg.LinAlgError:
        print("unable to symmetrize problem at late times")
        skip_late = True
        commutator_norms = 0
        commutator_norm = 0
    eigfin = np.zeros((len(eigvals)), dtype=np.float)
    for i, j in enumerate(eigvals):
        if j.imag == 0 or np.isnan(j.imag):
            eigfin[i] = eigvals[i].real
        else:
            if USE_LATE_TIMES and not skip_late and late:
                eigvals, evecs = solve_gevp(c_lhs_new)
                break
            else:
                print("late, skip_late, eigvals", late, skip_late, eigvals)
                sys.exit(1)
                raise ImaginaryEigenvalue

    for i, j in enumerate(eigvals):
        if j.imag == 0 or np.isnan(j.imag):
            eigfin[i] = eigvals[i].real
        else:
            print("Eigenvalue=", j)
            raise ImaginaryEigenvalue
    if print_evecs:
        print("start solve")
        print("lhs=", c_lhs)
        print("rhs=", c_rhs)
        for i, j in enumerate(eigvals):
            print("eigval #", i, "=", j, "evec #", i, "=", evecs[:, i])
        print("end solve")
    #checkgteq0(eigfin)
    ret = (eigfin, commutator_norm, commutator_norms) if commnorm else eigfin
    return ret
get_eigvals.sent = False

def checkgteq0(eigfin):
    """Check to be sure all eigenvalues are greater than 0"""
    for i in eigfin:
        if not np.isnan(i):
            assert i >= 0, "negative eigenvalue found:"+str(eigfin)

def enforce_hermiticity(gevp_mat):
    """C->(C+C^\dagger)/2"""
    return kahan_sum([np.conj(gevp_mat).T, gevp_mat])/2


class ImaginaryEigenvalue(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, expression='', message=''):
        print("***ERROR***")
        print('imaginary eigenvalue found')
        super(ImaginaryEigenvalue, self).__init__(message)
        self.expression = expression
        self.message = message

class XmaxError(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, problemx=None, message=''):
        print("***ERROR***")
        print('xmax likely too large, decreasing')
        super(XmaxError, self).__init__(message)
        self.problemx = problemx
        self.message = message


def variance_reduction(orig, avg, decrease_var=DECREASE_VAR):
    """
    apply y->(y_i-<y>)*decrease_var+<y>
    """
    nanindices = []
    if hasattr(orig, '__iter__'):
        assert hasattr(avg, '__iter__'), "dimension mismatch"
        if len(orig.shape) == 1:
            for i,j in enumerate(zip(orig, avg)):
                    if any(np.isnan(j)):
                        orig[i] = np.nan
                        avg[i] = np.nan
    else:
        assert not np.isnan(orig+avg), "nan found"
    ret = (orig-avg)*decrease_var+avg
    check = (ret-avg)/decrease_var+avg
    try:
        assert np.allclose(check, orig, rtol=1e-8, equal_nan=True),\
            "precision loss detected:"+str(orig)+" "+\
            str(check)+" "+str(decrease_var)
    except AssertionError:
        print('avg =', avg)
        print('ret =', ret)
        print('check =', check)
        print('orig =', orig)
        print("precision loss detected, orig != check")
        sys.exit(1)
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

def callprocmeff(eigvals, timeij, delta_t):
    """Call processing function for effective mass"""
    dimops = len(eigvals[0])
    toproc = 1/eigvals[0] if not LOGFORM else eigvals[0]/delta_t
    energies = np.array([proc_meff((toproc[op], 1, eigvals[1][op], eigvals[2][op]), index=op, time_arr=timeij) for op in range(dimops)])
    energies =  energies/delta_t if not LOGFORM else energies
    #avg_energies = eigvals_mean_t/delta_t
    return energies


if EFF_MASS:
    def getblock_gevp(file_tup, delta_t, timeij=None,
                      decrease_var=DECREASE_VAR):
        """Get a single rhs; loop to find the one with the least nan's"""
        blkdict = {}
        countdict = {}
        errdict = {}
        check_length = 0
        solve_gevp.hint = HINTS_ELIM[timeij] if timeij in HINTS_ELIM\
            else None
        assert len(file_tup) == 5, "bad file_tup length:"+str(len(file_tup))
        if hasattr(delta_t, '__iter__'):
            for i, dt1 in enumerate(delta_t):
                try:
                    assert len(file_tup[1]) == len(delta_t),\
                        "rhs times dimensions do not match:"+\
                        len(file_tup[1])+str(",")+len(delta_t)
                except TypeError:
                    print(file_tup)
                    print(delta_t)
                    assert None, "bug"
                argtup = (file_tup[0], file_tup[1][i], *file_tup[2:])
                for i,j in enumerate(argtup):
                    assert j is not None, "file_tup["+str(i)+"] is None"
                blkdict[dt1] = getblock_gevp_singlerhs(argtup, dt1, timeij,
                                                       decrease_var)
                print('dt1' , dt1, 'timeij', timeij, 'elim hint',
                      solve_gevp.hint,
                      "operator eliminations", allowedeliminations(),
                      'sample', blkdict[dt1][0])
                check_length = len(blkdict[dt1])
                relerr = np.abs(np.std(blkdict[dt1], ddof=1, axis=0)*(
                    len(blkdict[dt1])-1)/np.mean(blkdict[dt1], axis=0))
                errdict[dt1] = 0
                if not all(np.isnan(relerr)):
                    errdict[dt1] = np.nanargmax(relerr)
                count = 0
                for config, blk in enumerate(blkdict[dt1]):
                    count = max(np.count_nonzero(~np.isnan(blk)), count)
                    countdict[count] = dt1
            keymax = countdict[max([count for count in countdict])]
            print("final tlhs, trhs =", timeij, timeij-keymax, "next hint:(",
                  np.count_nonzero(~np.isnan(blkdict[keymax][0])),
                  ",", errdict[keymax], ")")
            for key in blkdict:
                assert len(blkdict[key]) == check_length,\
                    "number of configs is not consistent for different times"
            ret = blkdict[keymax]
        else:
            ret = getblock_gevp_singlerhs(
                file_tup, delta_t, timeij=timeij, decrease_var=decrease_var)
            relerr = np.abs(np.std(ret, ddof=1, axis=0)*(
                len(ret)-1)/np.mean(ret, axis=0))
            print('dt1' , delta_t, 'timeij', timeij, 'elim hint',
                  solve_gevp.hint,
                  "operator eliminations", allowedeliminations(),
                  'sample', ret[0])
            print("final tlhs, trhs =", timeij,
                  timeij-delta_t, "next hint:(",
                  np.count_nonzero(~np.isnan(ret[0])), ",",
                  np.nanargmax(relerr) if not all(np.isnan(relerr)) else 0,
                  ")")
        return ret


    def average_energies(mean_cmats_lhs, mean_crhs, delta_t, timeij):
        """get average energies"""
        dimops = len(mean_crhs)
        cmat_lhs_t_mean = mean_cmats_lhs[0]
        cmat_lhs_tp1_mean = mean_cmats_lhs[1]
        cmat_lhs_tp2_mean = mean_cmats_lhs[2]
        cmat_lhs_tp3_mean = mean_cmats_lhs[3]
        while 1<2:
            eigvals_mean_t = get_eigvals(cmat_lhs_t_mean, mean_crhs)
            try:
                if DELETE_NEGATIVE_OPERATORS:
                    checkgteq0(eigvals_mean_t)
                break
            except AssertionError:
                print("negative eigenvalues found")
                print('eigvals:', eigvals_mean_t)
                print("allowed operator eliminations:",
                      allowedeliminations())


        #eigvals_mean_tp1 = get_eigvals(cmat_lhs_tp1_mean, mean_crhs)
        eigvals_mean_tp1 = [np.nan]*len(eigvals_mean_t)
        #eigvals_mean_tp2 = get_eigvals(cmat_lhs_tp2_mean, mean_crhs)
        eigvals_mean_tp2 = [np.nan]*len(eigvals_mean_t)
        #eigvals_mean_tp3 = get_eigvals(cmat_lhs_tp3_mean, mean_crhs)
        eigvals_mean_tp3 = [np.nan]*len(eigvals_mean_t)
        if DELETE_NEGATIVE_OPERATORS:
            checkgteq0(eigvals_mean_tp1)
            checkgteq0(eigvals_mean_tp2)
            checkgteq0(eigvals_mean_tp3)

        avg_energies = callprocmeff([eigvals_mean_t, eigvals_mean_tp2,
                                     eigvals_mean_tp3], timeij, delta_t)

        return avg_energies, eigvals_mean_t

    def getlhsrhs(file_tup, num_configs):
        """Get lhs and rhs gevp matrices from file_tup
        (extract from index structure)
        """
        mean_cmats_lhs = []
        cmats_lhs = []
        assert len(file_tup) == 5, "bad length:"+str(len(file_tup))
        for idx in range(5):
            cmat , mean = readin_gevp_matrices(file_tup[idx], num_configs)
            if idx == 1:
                cmat_rhs, mean_crhs = cmat, mean
            else:
                cmats_lhs.append(cmat)
                mean_cmats_lhs.append(mean)
        return cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs


    def getblock_gevp_singlerhs(file_tup, delta_t, timeij=None,
                                decrease_var=DECREASE_VAR):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        files_tup[2] is the t+1 lhs
        files_tup[3] is the t+2 lhs
        C(t)v = Eigval*C(t_0)v
        """
        if timeij == 3.0:
            assert delta_t == 1, "bug:"+str(delta_t)
        assert delta_t is not None, "delta_t is None"
        if not delta_t:
            delta_t = 1.0
        dimops = len(file_tup[0])
        if STYPE == 'ascii':
            num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        elif STYPE == 'hdf5':
            num_configs = len(file_tup[0][0][0])
        if GEVP_DEBUG:
            print("Getting block for time slice=", timeij)

        # get the gevp matrices
        cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs = getlhsrhs(
            file_tup, num_configs)

        norm_comm = []
        norms_comm = []
        num = 0
        # reset the list of allowed operator eliminations at the
        # beginning of the loop
        allowedeliminations(reset=True)
        while num < num_configs:
            if GEVP_DEBUG:
                print("config #=", num)
            tprob = timeij
            try:
                if num == 0:
                    solve_gevp.mean = None
                    avg_energies, avg_eigvals = average_energies(
                        mean_cmats_lhs, mean_crhs, delta_t, timeij)
                    solve_gevp.mean = avg_eigvals if REINFLATE_BEFORE_LOG\
                        else None
                else:
                    pass
                    #assert solve_gevp.mean is not None, "bug"
                eigret = get_eigvals(cmats_lhs[0][num], cmat_rhs[num],
                                     print_evecs=True, commnorm=True)
                norm_comm.append(eigret[1])
                norms_comm.append(eigret[2])
                eigvals = np.array(eigret[0])
                #print('avg_energies', avg_energies)

                try:
                    if DELETE_NEGATIVE_OPERATORS:
                        checkgteq0(eigvals)
                except AssertionError:
                    print("negative eigenvalues found (non-avg)")
                    print('eigvals:', eigvals)
                    print("allowed operator eliminations:",
                          allowedeliminations())
                    num = 0
                    continue

                tprob = None if not EFF_MASS else tprob

                #eigvals2 = get_eigvals(cmat_lhs_tp1[num], cmat_rhs[num])
                eigvals2 = [np.nan]*len(eigvals)

                #eigvals3 = get_eigvals(cmat_lhs_tp2[num], cmat_rhs[num])
                eigvals3 = [np.nan]*len(eigvals)

                #eigvals4 = get_eigvals(cmat_lhs_tp3[num],
                # cmat_rhs[num], overb=True)
                eigvals4 = [np.nan]*len(eigvals)

                if PIONRATIO:
                    div = np.array([np.real(
                        PION[i][num][int(timeij)]**2-PION[i][num][int(
                        timeij)+1]**2) for i in range(dimops)])/1e10
                    eigvals /= div
                    eigvals2 /= div
                    eigvals3 /= div
                    eigvals4 /= div
                    for i, j in enumerate(ADD_CONST_VEC):
                        eigvals[i] -= eigvals2[i]*j
                        eigvals2[i] -= eigvals3[i]*j
            except ImaginaryEigenvalue:
                #print(num, file_tup)
                print('config_num:', num, 'time:', tprob)
                if tprob is not None:
                    raise XmaxError(problemx=tprob)
            
            # process the eigenvalues
            
            energies = callprocmeff([eigvals, eigvals3, eigvals4],
                                    timeij, delta_t)
            if solve_gevp.mean is None:
                result = variance_reduction(energies,
                                            avg_energies, 1/decrease_var)
            if num == 0:
                check_variance = []
                retblk = deque()
            check_variance.append(eigvals)
            retblk.append(result)
            num += 1
        assert len(retblk) == num_configs,\
            "number of configs should be the block length"
        #retblk = propagate_nans(retblk)
        if MPIRANK == 0:
            pass
            #chisq_bad, pval, dof = pval_commutator(norms_comm)
            #print("average commutator norm,
            #(t =", timeij, ") =", np.mean(norm_comm),
            #"chi^2/dof =", chisq_bad, "p-value =", pval, "dof =", dof)
        if GEVP_DEBUG:
            if not np.isnan(np.array(check_variance)).any():
                error_check = jack_mean_err(np.array(check_variance))
            else:
                error_check = None
            print("time, avg evals, variance of evals:",
                  timeij, error_check)
            if timeij == 16:
                sys.exit(0)
        return retblk

else:
    def getblock_gevp(file_tup, _, timeij=None):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        C(t)v = Eigval*C(t_0)v
        """
        retblk = deque()
        if timeij:
            pass
        if STYPE == 'ascii':
            num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        elif STYPE == 'hdf5':
            num_configs = len(file_tup[0][0][0])
        for num in range(num_configs):
            try:
                eigvals = get_eigvals(num, file_tup[0], file_tup[1])
            except ImaginaryEigenvalue:
                print(num, timeij)
                sys.exit(1)
            retblk.append(eigvals)
        return retblk

# obsolete; only works if GEVP matrix vector space shifts a lot from time slice to time slice
# weak consistency check
def pval_commutator(norms_comm):
    """Find the p-value for the test statistic defined by
    sum of squares of the commutator of the LHS and inverse RHS GEVP matrices
    we want to test this being consistent with 0
    """
    norms_comm = np.asarray(norms_comm)
    dimops = len(norms_comm[0][0])
    pca_list = []
    for i in norms_comm:
        reshape = np.reshape(i, dimops**2)
        pca_list.append(reshape)
    pca_list = np.asarray(pca_list)
    results_pca = PCA(pca_list, standardize=False)
    dof = len([i for i in results_pca.fracs if i > 0])
    #print("frac=", results_pca.fracs, "dof=", dof)
    #print("mean=", np.mean(results_pca.Y, axis=0))
    results_pca.Y = np.asarray(results_pca.Y)[:, dimops:dof]
    dof = results_pca.Y.shape[1]
    #print("original dimensions", dimops,
    #      "reduced dimensions:", results_pca.Y.shape)
    sample_stddev = np.std(
        results_pca.Y, ddof=0, axis=0)
    # assuming the population mean of our statistic is 0
    chisq_arr = []
    for i in results_pca.Y:
        chisq_arr.append(fsum([i[j]**2/sample_stddev[j]**2
                               for j in range(dof)]))
    chisq_arr = np.array(chisq_arr)
    pval_arr = 1-stats.chi2.cdf(chisq_arr, dof)
    pval = fsum(pval_arr)/len(chisq_arr)
    chisq = fsum(chisq_arr)/len(chisq_arr)
    #print("dev:", np.std(chisq_arr, ddof=1))
    #for i in sorted(list(chisq_arr)):
    #    print(i, ",")
    #print(np.sum(chisq_arr)/len(chisq_arr), fsum(chisq_arr)/len(chisq_arr))
    return chisq, pval, dof

if EFF_MASS:
    def getblock_simple(file_tup, reuse, timeij=None):
        """Given file,
        get block of effective masses, store in reuse[ij_str]
        """
        retblk = deque()
        if STYPE == 'ascii':
            zipfs = zip(open(file_tup[0], 'r'), open(file_tup[1], 'r'),
                        open(file_tup[2], 'r'), open(file_tup[3], 'r'))
        elif STYPE == 'hdf5':
            zipfs = zip(file_tup[0], file_tup[1], file_tup[2], file_tup[3])
        for line, line2, line3, line4 in zipfs:
            if line+line2+line3 not in reuse:
                reuse[str(line)+"@"+str(line2)+"@"+str(line3)] = proc_meff(
                    (line, line2, line3, line4),
                    files=file_tup, time_arr=timeij)
            if reuse[str(line)+'@'+str(line2)+'@'+str(line3)] == 0:
                raise Exception("Something has gone wrong.")
            retblk.append(reuse[str(line)+'@'+str(line2)+'@'+str(line3)])
        return retblk

else:
    def getblock_simple(ijfile, reuse, timeij=None):
        """Given file,
        get block, store in reuse[ij_str]
        """
        if reuse or timeij:
            pass
        retblk = deque()
        if STYPE == 'ascii':
            fn1 = open(ijfile)
        elif STYPE == 'hdf5':
            fn1 = ijfile
        for line in fn1:
            retblk.append(proc_line(line, ijfile))
        return retblk

# system stuff, do the subtraction of bad configs as well

if GEVP:

    def test_imagblk(blk):
        """test block for imaginary eigenvalues in gevp"""
        for test1 in blk:
            for test in test1:
                if test.imag != 0:
                    print("***ERROR***")
                    print("GEVP has negative eigenvalues.")
                    sys.exit(1)

    def getblock_plus(file_tup, reuse, timeij=None, delta_t=None):
        """get the block"""
        if reuse:
            pass
        retblk = getblock_gevp(file_tup, delta_t, timeij)
        test_imagblk(retblk)
        return retblk
else:

    def getblock_plus(file_tup, reuse, timeij=None, delta_t=None):
        """get the block"""
        if delta_t is None:
            pass
        return getblock_simple(file_tup, reuse, timeij)


def getblock(file_tup, reuse, timeij=None, delta_t=None):
    """get the block and subtract any bad configs"""
    retblk = np.array(getblock_plus(file_tup, reuse, timeij,
                                    delta_t=delta_t))
    if ELIM_JKCONF_LIST:
        retblk = elim_jkconfigs(retblk)
    if BINNUM != 1:
        retblk = binconf(retblk)
    return retblk
