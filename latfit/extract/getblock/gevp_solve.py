"""Functions to solve the generalized eigenvalue problem (GEVP)"""
import sys
from math import fsum

import scipy
from scipy import stats
from sympy import N
import numpy as np
from accupy import kdot
from matplotlib.mlab import PCA

from latfit.utilities import exactmean as em
from latfit.analysis.errorcodes import ImaginaryEigenvalue
from latfit.analysis.errorcodes import PrecisionLossError
from latfit.extract.getblock.gevp_linalg import variance_reduction
from latfit.extract.getblock.gevp_linalg import printevecs, convtosmat
from latfit.extract.getblock.gevp_linalg import finaleval_imag_check
from latfit.extract.getblock.gevp_linalg import all0imag_ignorenan, makeneg
from latfit.extract.getblock.gevp_linalg import checkherm, inflate_with_nan
from latfit.extract.getblock.gevp_linalg import removerowcol, check_solve
from latfit.extract.getblock.gevp_linalg import bracket, cmatdot, defsign
from latfit.extract.getblock.gevp_linalg import enforce_hermiticity
from latfit.extract.getblock.gevp_linalg import is_pos_semidef, check_bracket
from latfit.extract.getblock.gevp_linalg import log_matrix, drop0imag
import latfit.extract.getblock.disp_hacks as gdisp
import latfit.extract.getblock.gevp_linalg as glin

from latfit.config import UNCORR, USE_LATE_TIMES
from latfit.config import GEVP_DEBUG, DECREASE_VAR, DELETE_NEGATIVE_OPERATORS
from latfit.config import LOGFORM

MEAN = None
HINT = None



def calleig_logform(c_lhs, flag, c_rhs=None):
    """Log form of the ?GEVP?"""
    if c_rhs is not None:
        rhs = log_matrix(c_rhs)
        lhs = log_matrix(c_lhs)
        c_lhs_check = kdot(scipy.linalg.inv(c_rhs), c_lhs)
        c_lhs = rhs-lhs
        try:
            assert np.allclose(scipy.linalg.eigvals(c_lhs_check),
                               scipy.linalg.eigvals(
                                   scipy.linalg.expm(-c_lhs)))
        except AssertionError:
            print(np.log(scipy.linalg.eigvals(c_lhs_check)))
            print(scipy.linalg.eigvals(-c_lhs))
            print(c_lhs_check)
            print(scipy.linalg.expm(c_lhs))
            sys.exit(1)
        assert np.allclose(c_lhs+lhs, rhs, rtol=1e-12)
        c_rhs = None
        flag = True
    else:
        c_lhs = -1*log_matrix(c_lhs, check=True)
        flag = True
    assert is_pos_semidef(c_lhs), "not positive semi-definite."
    c_lhs = enforce_hermiticity(c_lhs)
    assert is_pos_semidef(c_lhs), "not positive semi-definite."
    eigenvals, evecs = scipy.linalg.eig(c_lhs, c_rhs,
                                        overwrite_a=False,
                                        overwrite_b=False,
                                        check_finite=True)
    return eigenvals, evecs, flag

def calleig(c_lhs, c_rhs=None):
    """Actual call to scipy.linalg.eig"""
    flag = False
    if LOGFORM:
        eigenvals, evecs, flag = calleig_logform(c_lhs, flag, c_rhs)
    else:
        checkherm(c_lhs)
        checkherm(c_rhs)
        signlhs = defsign(c_lhs)
        signrhs = defsign(c_rhs)
        assert signlhs == signrhs
        assert signrhs
        assert signlhs
        eigenvals, evecs = scipy.linalg.eig(c_lhs, c_rhs,
                                            overwrite_a=False,
                                            overwrite_b=False,
                                            check_finite=True)
    for i, (eval1, evec) in enumerate(zip(eigenvals, evecs.T)):
        flag_nosolve = check_solve(eval1, evec, c_lhs, c_rhs)
        eigenvals[i] = -1 if flag_nosolve else eigenvals[i]
        if not flag_nosolve:
            check_bracket(eval1, evec, c_lhs, c_rhs)
    if flag:
        if not np.all(np.imag(eigenvals) < 1e-8):
            print("non-negligible imaginary eigenvalues found")
            print("eigenvals:")
            print(eigenvals)
            sys.exit(1)
        eigenvals = np.real(eigenvals)
    eigenvals, evecs = glin.sortevals(eigenvals, evecs, c_lhs, c_rhs)
    return eigenvals, evecs

def solve_gevp(c_lhs, c_rhs=None):
    """Solve the GEVP"""
    dimops_orig = len(c_lhs)
    dimops = len(c_lhs)
    dimremaining, toelim = nexthint()
    eigvals, evecs = calleig(c_lhs, c_rhs)
    remaining_operator_indices = set(range(dimops))
    eigvals = elim_and_inflate(eigvals, evecs, toelim, dimops, dimremaining)
    eliminated_operators = set()
    while any(eigvals < 0):

        if not DELETE_NEGATIVE_OPERATORS:
            break

        dimops, dimremaining, toelim, eigvals, brk = indexing_update(
            dimops, dimops_orig, dimremaining, toelim, eigvals)
        if brk:
            break

        dimdeldict, eigvals = neg_op_trials(
            dimops, dimremaining, toelim, c_lhs, c_rhs)
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
        if MEAN is not None:
            eigvals = variance_reduction(eigvals, MEAN[:dimops],
                                         1/DECREASE_VAR)
            eigvals, evecs = glin.sortevals(eigvals, evecs, c_lhs, c_rhs)
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
    eigvals = inflate_with_nan(dimops,
                               dimops_orig,
                               eigvals, eliminated_operators)
    nexthint(0)
    return eigvals, evecs

def get_eigvals(c_lhs, c_rhs, print_evecs=False,
                commnorm=False):
    """get the nth generalized eigenvalue from matrices of files
    file_tup_lhs, file_tup_rhs
    optionally, overwrite the rhs matrix we get
    if we don't need it anymore (overb=True)
    """
    checkherm(c_lhs)
    checkherm(c_rhs)
    print_evecs = False if not GEVP_DEBUG else print_evecs
    if not get_eigvals.sent and print_evecs:
        print("First solve, so printing norms "+\
              "which are multiplied onto GEVP entries.")
        print("e.g. C(t)_ij -> Norms[i][j]*C(t)_ij")
        print("Norms=", gdisp.NORMS)
        get_eigvals.sent = True
    eigvals, evecs = solve_gevp(c_lhs, c_rhs)
    #checkgteq0(eigvals)
    late = False if all0imag_ignorenan(eigvals) else True
    try:
        assert not late
    except AssertionError:
        print("imaginary eigenvalues found.")
        print(c_lhs)
        print(c_rhs)
        print("determinants:")
        print(scipy.linalg.det(c_lhs), scipy.linalg.det(c_rhs))
        print("evals of lhs, rhs:")
        print(scipy.linalg.eigvals(c_lhs))
        print(scipy.linalg.eigvals(c_rhs))
        print("GEVP evals")
        print(eigvals)
        sys.exit(1)
    eigvals, commutator_norm, commutator_norms = comm_correct_evp(
        c_lhs, c_rhs, late, eigvals)
    eigfin = finaleval_imag_check(eigvals)
    if print_evecs:
        printevecs(c_lhs, c_rhs, eigvals, evecs)
    #checkgteq0(eigfin)
    evecs = evecs.T
    ret = (eigfin, evecs, commutator_norm, commutator_norms)\
        if commnorm else (eigfin, evecs)
    assert len(ret) == 2 or len(ret) == 4, str(ret)
    return ret
get_eigvals.sent = False

def sym_evals_gevp(c_lhs, c_rhs):
    """Do high precision solve of GEVP"""
    c_lhs = np.asarray(c_lhs)
    c_rhs = np.asarray(c_rhs)
    c_lhs = convtosmat(0.5*c_lhs)
    c_rhs = convtosmat(0.5*c_rhs)
    c_lhs = c_lhs+c_lhs.T
    c_rhs = c_rhs+c_rhs.T
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

#### Untested/Controversial/Wrong methods

## deletion of operators?

def nexthint(idx=None):
    """Get the next operator to delete"""
    ret = (-1, -2)
    if isinstance(HINT, tuple):
        ret = HINT
    elif isinstance(HINT, list):
        ret = HINT[nexthint.idx]
        if len(HINT) > nexthint.idx+1:
            nexthint.idx += 1
    else:
        assert HINT is None, \
            "inconsistency in assigning variable"
    dimremaining, toelim = ret
    assert toelim < dimremaining, "index error"
    assert isinstance(dimremaining, int), "bug"
    assert isinstance(toelim, int), "bug"
    nexthint.idx = idx if idx is not None else nexthint.idx
    return ret
nexthint.idx = 0

def comm_correct_evp(c_lhs, c_rhs, late, eigvals):
    """Try to eliminate eigenvalue imaginary piece via commutator"""
    dimops = len(c_lhs)
    skip_late = False
    c_lhs = drop0imag(c_lhs)
    c_rhs = drop0imag(c_rhs)
    try:
        c_rhs_inv = scipy.linalg.inv(c_rhs)
        # compute commutator divided by norm
        # to see how close rhs and lhs bases
        try:
            commutator_norms = (kdot(c_rhs_inv, c_lhs)-kdot(
                c_lhs, c_rhs_inv))
        except FloatingPointError:
            print("bad denominator:")
            print(np.linalg.norm(c_rhs_inv))
            print(np.linalg.norm(c_lhs))
            print(c_lhs)
            raise FloatingPointError
        assert np.allclose(kdot(c_rhs_inv, c_rhs),
                           np.eye(dimops), rtol=1e-8), \
                           "Bad C_rhs inverse. Numerically unstable."
        assert np.allclose(np.matrix(c_rhs_inv).H, c_rhs_inv,
                           rtol=1e-8), \
                           "Inverse failed (result is not hermite)."
        c_lhs_new = (kdot(c_rhs_inv, c_lhs)+kdot(c_lhs, c_rhs_inv))/2
        commutator_norm = np.linalg.norm(commutator_norms)
        try:
            assert np.allclose(np.matrix(c_lhs_new).H, c_lhs_new, rtol=1e-8)
        except AssertionError:
            print("Correction to hermitian matrix failed.")
            print("commutator norm =", commutator_norm)
            print(c_lhs_new.T)
            print(c_lhs_new)
            print("printing difference in rows:")
            for lf1, mf1 in zip(c_lhs_new.T, c_lhs_new):
                print(lf1-mf1)
            sys.exit(1)
    except np.linalg.linalg.LinAlgError:
        print("unable to symmetrize problem at late times")
        skip_late = True
        commutator_norms = 0
        commutator_norm = 0
    for _, j in enumerate(eigvals):
        if not (abs(j.imag) < 1e-8 or np.isnan(j.imag)):
            if USE_LATE_TIMES and not skip_late and late:
                eigvals, _ = solve_gevp(c_lhs_new)
                break
            else:
                print("late, skip_late, eigvals", late, skip_late, eigvals)
                sys.exit(1)
                raise ImaginaryEigenvalue
    return eigvals, commutator_norm, commutator_norms

def neg_op_trials(dimops, dimremaining, toelim, c_lhs, c_rhs=None):
    """ try to eliminate different operators
    to remove negative eigenvalues
    """
    count = 0
    dimdeldict = {}
    loop = list(range(dimops+1))
    for dimdel in loop:
        if dimdel == 0 and toelim < 0: # heuristic, override with hint
            continue
        c_lhs_temp = removerowcol(c_lhs, dimdel)
        c_rhs_temp = removerowcol(
            c_rhs, dimdel) if c_rhs is not None else c_rhs
        eigvals, evecs = calleig(c_lhs_temp, c_rhs_temp)
        if MEAN is not None:
            eigvals = variance_reduction(
                eigvals, MEAN[:dimops], 1/DECREASE_VAR)
            eigvals, evecs = glin.sortevals(eigvals, evecs, c_lhs, c_rhs)

        if dimremaining == dimops:
            eigvals[toelim] = makeneg(eigvals[toelim])
        # count number of non-negative eigenvalues
        count = max(np.count_nonzero(eigvals > 0), count)
        # store dimension deletion leading to this count
        dimdeldict[count] = dimdel
    return dimdeldict, eigvals


def allowedeliminations(newelim=None, reset=False):
    """The complete list of allowed eliminations"""
    if reset:
        allowedeliminations.elims = None
    else:
        allowedeliminations.elims = set(newelim) if isinstance(
            newelim, set) and newelim else allowedeliminations.elims
    return allowedeliminations.elims
allowedeliminations.elims = None


def elim_and_inflate(eigvals, evecs, toelim, dimops, dimremaining):
    """make eval negative to eliminate it
    also, if we are reinflating before the log, do that too
    """
    if dimops == dimremaining:
        eigvals[toelim] = makeneg(eigvals[toelim])
    if MEAN is not None:
        assert 1/DECREASE_VAR > 1, \
            "variance is being reduced, but it should be increased here."
        eigvals = variance_reduction(eigvals, MEAN,
                                     1/DECREASE_VAR)
        eigvals, evecs = glin.sortevals(eigvals, evecs)

    return eigvals
    #allowedeliminations(reset=True)

def indexing_update(dimops, dimops_orig, dimremaining, toelim, eigvals):
    """Update indexing for deletion while loop"""
    # indexing updates
    assert isinstance(dimremaining, int), "bug"
    assert isinstance(toelim, int), "bug"
    dimops -= 1
    dimremaining, toelim = nexthint()
    brk = False
    if dimops == 0:
        #print("dimension reduction technique exhausted.")
        eigvals = np.array([np.nan]*dimops_orig)
        brk = True
    return dimops, dimremaining, toelim, eigvals, brk

# obsolete; only works if GEVP matrix vector space
# shifts a lot from time slice to time slice
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
    #print("mean=", em.acmean(results_pca.Y, axis=0))
    results_pca.Y = np.asarray(results_pca.Y)[:, dimops:dof]
    dof = results_pca.Y.shape[1]
    #print("original dimensions", dimops,
    #      "reduced dimensions:", results_pca.Y.shape)
    sample_stddev = em.acstd(
        results_pca.Y, ddof=0, axis=0)
    # assuming the population mean of our statistic is 0
    chisq_arr = []
    for i in results_pca.Y:
        chisq_arr.append(fsum([i[j]**2/sample_stddev[j]**2
                               for j in range(dof)]))
    chisq_arr = np.array(chisq_arr)
    assert None, "pvalue code not checked below this line"
    if UNCORR:
        pval_arr = 1- stats.chi2.cdf(chisq_arr, dof)
    else:
        pval_arr = stats.f.sf(chisq_arr*(len(
            chisq_arr)-dof)/(len(chisq_arr)-1)/dof, dof, len(chisq_arr)-dof)
    pval = fsum(pval_arr)/len(chisq_arr)
    chisq = fsum(chisq_arr)/len(chisq_arr)
    #print("dev:", em.acstd(chisq_arr, ddof=1))
    #for i in sorted(list(chisq_arr)):
    #    print(i, ", ")
    #print(em.acsum(chisq_arr)/len(chisq_arr),
    # fsum(chisq_arr)/len(chisq_arr))
    return chisq, pval, dof
