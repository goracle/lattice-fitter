"""Various basic linear algebra operations on the GEVP matrices"""
import sys
from math import exp, log

import scipy
import scipy.linalg
from scipy.linalg import inv, det
import numpy as np
from accupy import kdot
from sympy import S
from sympy.matrices import Matrix
import gvar

from latfit.utilities import exactmean as em
from latfit.config import GEVP_DEBUG, LOGFORM, DECREASE_VAR, PSEUDO_SORT
from latfit.config import VERBOSE
from latfit.analysis.errorcodes import ImaginaryEigenvalue
from latfit.analysis.errorcodes import NegativeEigenvalue
from latfit.analysis.errorcodes import PrecisionLossError
from latfit.analysis.errorcodes import EigenvalueSignInconsistency

def checkgteq0(eigfin):
    """Check to be sure all eigenvalues are greater than 0"""
    for i in eigfin:
        if not np.isnan(i):
            try:
                assert i >= 0
            except AssertionError:
                print("negative eigenvalue found:"+str(eigfin))
                raise NegativeEigenvalue


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

def check_solve(eigval, evec, c_lhs, c_rhs):
    """Check the solution of the GEVP.
    Does it actually solve the GEVP?"""
    for j, k in zip(cmatdot(c_lhs, evec), cmatdot(c_rhs, evec)):
        try:
            if j and k:
                assert np.allclose(eigval, j/k, rtol=1e-8)
            else:
                assert j == k
            flag_nosolve = False
        except FloatingPointError:
            print("invalid GEVP values found")
            print("lhs vec, rhs vec, eval")
            print(j, k, eigval)
            sys.exit(1)
        except AssertionError:
            flag_nosolve = True
    return flag_nosolve

def check_bracket(eigval, evec, c_lhs, c_rhs):
    """Check that the ratio of brackets gives the eigenvalue"""
    eval_check = bracket(evec, c_lhs)/bracket(evec, c_rhs)
    try:
        assert np.allclose(eval_check, eigval, rtol=1e-10)
    except AssertionError:
        print("Eigenvalue consistency check failed."+\
            "  ratio and eigenvalue not equal.")
        print("bracket lhs, bracket rhs, ratio, eval")
        print(bracket(evec, c_lhs), bracket(evec, c_rhs),
              bracket(evec, c_lhs)/bracket(evec, c_rhs), eigval)
        raise PrecisionLossError

def printevecs(c_lhs, c_rhs, eigvals, evecs):
    """Debug function, prints diagnostic info"""
    print("start solve")
    print("lhs=", np.array2string(c_lhs, separator=', '))
    print("rhs=", np.array2string(c_rhs, separator=', '))
    for i, j in enumerate(eigvals):
        try:
            assert not check_solve(j, evecs[:, i], c_lhs, c_rhs)
            check_bracket(j, evecs[:, i], c_lhs, c_rhs)
        except PrecisionLossError:
            assert None, "unexpected precision loss"
        print("eigval #", i, "=", j, "evec #", i, "=",
              np.array2string(evecs[:, i], separator=', '))
        assert np.all(evecs[:, i] == evecs.T[i])
    print("a=", np.array2string(evecs[:, 0], separator=', '))
    print("b=", np.array2string(evecs[:, 1], separator=', '))
    print("c=", np.array2string(evecs[:, 2], separator=', '))
    print("end solve")
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

def posdef_diag_check(cmat):
    """If any of the GEVP diagonal terms is zero within errors,
    throw an error"""
    for i, _ in enumerate(cmat[0]):
        mat = cmat[:, i, i]
        posdef_check(mat, idx1=i, idx2=i)

def posdef_check(mat, idx1=None, idx2=None, time=None):
    """Check for positive definiteness"""
    val = em.acmean(mat, axis=0)
    assert not hasattr(val, '__iter__'), val
    sdev = em.acstd(mat, axis=0)*np.sqrt(len(mat)-1)
    if sdev*1.5 >= np.abs(val):
        print("Signal loss for correlator src index", idx1,
              "to sink index", idx2)
        print('val', gvar.gvar(val, sdev))
        if time is not None:
            print("op from time:", time)
        raise PrecisionLossError

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
        if VERBOSE:
            print("eigenvalues are not all the same sign:", str(evals))
            print(cmat)
        raise EigenvalueSignInconsistency
        #ret = False
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

def cmatdot(cmat, vec, transp=False, sloppy=False):
    """Dot gevp matrix into vec on rhs if not transp"""
    cmat = np.asarray(cmat)
    cmat = cmat.T if transp else cmat
    vec = np.asarray(vec)
    assert len(vec) == len(cmat), str(vec)+" "+str(cmat)
    ret = np.zeros(vec.shape, np.complex)
    for i, row in enumerate(cmat):
        tosum = []
        for j, item in enumerate(row):
            tosum.append(item*vec[j])
        if sloppy:
            ret[i] = np.sum(tosum, axis=0)
        else:
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
        try:
            assert np.allclose(dotprod, 1.0, rtol=1e-8),\
                str(dotprod)
        except AssertionError:
            print("degenerate subspace failure; dotprod:", dotprod)
            print("dotprod should be 1.0")
            raise PrecisionLossError
    if GEVP_DEBUG:
        print("evecs of avg gevp",
              np.real(evecs_mean_t))

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
        assert np.allclose(
            check, orig, rtol=1e-8, equal_nan=True), \
            "precision loss detected:"+str(decrease_var)
    except AssertionError:
        #print('avg =', avg)
        #print('ret =', ret)
        #print('check =', check)
        #print('orig =', orig)
        print("precision loss detected, orig != check")
        raise PrecisionLossError
    return ret


### PSEUDO SORT -- BEGIN

def ratio_evals(evecs, c_lhs, c_rhs):
    """Get the ratio evals from the evecs and C(t), C(t_0)"""
    ret = []
    for i in evecs:
        assert len(i) == len(c_lhs), str(c_lhs)+" "+str(i)
        assert len(i) == len(c_rhs), str(c_rhs)+" "+str(i)
        num = bracket(i, c_lhs)
        denom = bracket(i, c_rhs)
        ret.append(num/denom)
    ret = np.asarray(ret)
    return ret

def index_pm1(idx, refl):
    """Find plus and minus index unless we are the border
    then flip; use reference length refl for length of vector
    to be indexed
    """
    assert refl > 1, refl
    if idx:
        idxm1 = idx-1
    else:
        idxm1 = 1
    idxp1 = idx+1
    if idxp1 >= refl:
        idxp1 = idx - 1
        assert idxp1 >= 0, (idx, refl)
    return idxp1, idxm1

def score(eval_to_score, ref_evals, idx, func='gaussian'):
    """Return an absolute similarity score for a pseudo
    eigenvalue (eval_to_score) to a set of reference eigenvalues
    using func as the base scoring function
    with idx as the best match index
    """
    zero = 1e-8
    idxp1, idxm1 = index_pm1(idx, len(ref_evals))
    if func == 'gaussian':
        diff = ref_evals[idx] - eval_to_score
        assert np.abs(np.imag(diff)) < 1e-8, diff
        if diff < 0:
            #idxp1 = (idx+1) % len(ref_evals)
            widp1 = (ref_evals[idx]-ref_evals[idxp1])**2/log(1/zero)
            assert np.abs(np.imag(widp1)) < 1e-8, widp1
            ret = exp(-(diff)**2/widp1)
        else:
            #idxm1 = (idx-1) % len(ref_evals)
            widm1 = (ref_evals[idx]-ref_evals[idxm1])**2/log(1/zero)
            assert np.abs(np.imag(widm1)) < 1e-8, widm1
            ret = exp(-(diff)**2/widm1)
    else:
        assert None, "other functions not supported at this time"
    return ret

def indicator(pseudo_evals, ref_evals, idx):
    """ legacy wrapper function; should be factored out

    Check the nearest neighbor alternative matches
    return the max score, normalized to the base score"""

    # the init_sort of the pseudo eigenvalues may leave them
    # not in ascending (or descending) order
    # thus, we may not find nearest (numerically) neighbors just by
    # looking at the init_sort'd vector of pseudo eigenvalues

    base_score = score(pseudo_evals[idx], ref_evals, idx)
    ret = base_score
    return ret


def most_similar_pair(pseudos, evals, evalsi):
    """Find most similar pair of entries
    
    evals = evalsi
    """
    mscr = 0
    for _, item1 in enumerate(pseudos):
        for _, item2 in enumerate(evals):
            idx = list(evalsi).index(item2)
            scr = score(item1, evalsi, idx)
            mscr = max(scr, mscr)
            if scr == mscr:
                ret = (item1, item2)
    return ret

def init_sort(pseudosi, evalsi):
    """Initial sort of pseudo eigenvalues"""
    mapi = {}

    # make list copies
    evalsi = list(evalsi)
    pseudosi = list(pseudosi)
    evals = list(evalsi)
    pseudos = list(pseudosi)

    # init ret
    ret = list(evalsi)

    while pseudos:
        valp, vale = most_similar_pair(pseudos, evals, evalsi)
        idxp, idxe = pseudos.index(valp), evals.index(vale)
        mapi[vale] = valp
        pseudos = list(np.delete(pseudos, idxp, axis=0))
        evals = list(np.delete(evals, idxe, axis=0))
    for i in mapi:
        ret[evalsi.index(i)] = mapi[i]
    return ret


def map_evals(evals_from, evals_to):
    """Get similarity mapping"""

    leval = len(evals_from)

    # length check
    assert leval == len(evals_to), str(
        evals_from)+" "+str(evals_to)
    # initial sort check
    assert list(evals_from) == list(fallback_sort(evals_from))

    # sort evals based on pseudo-evals
    evals_to_sorted = init_sort(evals_to, evals_from)

    # derive map from sort
    assert len(evals_to), evals_to
    ret = make_id(leval) # ret is map
    evals_to = list(evals_to)
    for _, (i, j) in enumerate(zip(evals_to_sorted, evals_to)):
        fidx = evals_to.index(i)
        tidx = evals_to.index(j)
        ret[fidx] = tidx
    assert not check_map_dups(ret), str(ret)+" "+str(test_arr)
    assert len(ret) == leval, str(ret)
    assert ret, ret

    # wrapper to get inverse scores
    rel_diff = [indicator(
        evals_to_sorted, evals_from, idx) for idx,
                _ in enumerate(evals_from)]
    rel_diff = np.array(rel_diff)
    # rel_diff = 1/(scores*sum(scores))
    # sum(scores) attempts to give more weight to time slices which have good
    # agreement across all comparisons
    # (or, rather, tries to filter out coincidental agreement)
    # also, attempt to allow for inf rel diff's
    try:
        rel_diff = [1/i if i else np.inf for i in np.sum(rel_diff)*rel_diff]
    except FloatingPointError:
        print('rel diff', rel_diff)
        raise
    rel_diff = list(rel_diff)
    rel_diff = invert_reldiff_map(rel_diff, ret)

    return ret, rel_diff

def invert_reldiff_map(rel_diff, dot_map):
    """Invert the ordering of relative diff vector
    (the later analysis assumes score is property of source, not sink)"""
    ret = []
    for idx in range(len(rel_diff)):
        ret.append(rel_diff[dot_map[idx]])
    return ret


def collision_check(smap):
    """Check for collision"""
    ret = len(set(smap)) != len(smap)
    if not ret:
        used = set()
        for i in smap:
            if smap[i] in used:
                ret = True
            else:
                used.add(smap[i])
    return ret


def sortevals(evals, evecs=None, c_lhs=None, c_rhs=None):
    """Sort eigenvalues in order of increasing energy, 
    then pseudo sort them.
    
    PSEUDO SORT MAIN
    """
    evals = list(evals)
    ret = fallback_sort(evals, evecs)
    # unpack
    if evecs is not None:
        evals = ret[0]
        assert evecs is not None, evecs
        evecs = ret[1]
        evals = list(evals)

    # now, pseudo-sort
    dot_map = make_id(len(evals))
    if sortevals.last_time is not None and c_lhs is not None\
       and c_rhs is not None and not np.any(np.isnan(evals)) and PSEUDO_SORT:

        count = 5 # max number of time slices to compare
        timeij = sortevals.last_time # most recent sorted time slice

        votes = []
        while timeij in sortevals.sorted_evecs and len(
                sortevals.sorted_evecs.keys()) > 1:

            # do not use first time (naive sort the first two time slices)
            if timeij == min(sortevals.sorted_evecs.keys()):
                break

            # loop increment
            count -= 1 # 3, 2, 1

            evecs_past = sortevals.sorted_evecs[timeij][sortevals.config]
            evals_past = sortevals.sorted_evals[timeij]
            # reinflate the error
            evals_past = gvar.gvar(np.mean(evals_past, axis=0),
                                   np.std(evals_past, axis=0)*np.sqrt(
                                       len(evals_past)-1)*1/DECREASE_VAR)

            # length checks
            assert len(evecs_past) == len(evals), str(evecs_past)
            assert len(evecs_past[0]) == len(evals), str(evecs_past[0])
            assert len(evecs_past[0]) == len(c_lhs), str(
                evecs_past[0])+" "+str(c_lhs)

            evals_from = np.copy(evals)
            evals_to = ratio_evals(evecs_past, c_lhs, c_rhs) # pseudo-evals
            vote_map, rel_diff = map_evals(
                evals_from, evals_to)

            # unambiguous different mapping; accumulate two identical votes
            # to resort the eigenvalues
            # ambiguous sorting, throw out this map

            assert vote_map, vote_map
            votes.append((vote_map, rel_diff, timeij))

            timeij -= 1 # t-1, t-2, t-3

        # if any votes are cast, modify the map (which is set to identity map)
        if votes:
            dot_map = votes_to_map(votes)

    assert len(dot_map) == len(evals), (evals, dot_map)
    assert not collision_check(dot_map) # check that map is injective (one-to-one)
    # sort the eigenvalues using the map
    ret = dot_map_to_evals_final(dot_map, evals, evecs)
    return ret
sortevals.sorted_evecs = {}
sortevals.sorted_evals = {}
sortevals.last_time = None
sortevals.config = None

def dot_map_to_evals_final(dot_map, evals, evecs):
    """Get final eval sort from dot map"""
    ret = (np.array(evals), np.array(evecs))
    if not isid(dot_map):
        sevals = np.zeros(len(evals))
        sevecs = np.zeros(np.asarray(evecs).T.shape)
        for i in dot_map:
            sevals[i] = drop0imag(evals[dot_map[i]])
            sevecs[i] = drop0imag(evecs.T[dot_map[i]])
        ret = (sevals, sevecs.T)
    return ret


def votes_to_map(votes, stop=np.inf):
    """Get sorting map based on previous time slice votes
    for what each one thinks is the right ordering"""
    ret = votes[0] # initial consensus from most recently sorted time slice
    stop = len(votes) if stop >= len(votes) else stop
    votes = votes[:stop]
    for mapi, scorei, timei in votes:
        #imap = filter_dict(i, idxsi)
        for mapj, scorej, timej in votes:
            if timej <= timei: # speed-up
                continue
            assert list(idxsj), votes
            #jmap = filter_dict(j, idxsj)
            ret1 = partial_compare_dicts((mapi, scorei, timei),
                                         (mapj, scorej, timej))
            ret = partial_compare_dicts(ret, ret1)
    return ret[0]

def partial_compare_dicts(ainfo, binfo):
    """Compare common entries in two dictionaries"""
    adict, arel, timea = ainfo
    bdict, brel, timeb = binfo

    # check to be sure score lists aren't empty
    assert list(arel), ainfo
    assert list(brel), binfo

    arel = del_maxrel(arel)
    brel = del_maxrel(brel)
    assert len(brel) == len(arel), (arel, brel)

    aset = set(adict)
    bset = set(bdict)
    inter = aset.intersection(bset)
    assert len(inter) == len(arel), (arel, inter)

    rrel = {}
    used = {}
    ret = {}
    passed = False # have we done one iter of the while loop below?

    # loop over map disagreements, using score to resolve disagreements
    while collision_check(ret) or len(ret) < len(inter):
        for i in sorted(list(inter)):

            # the score for the source is minimized
            # by second pass
            if i in ret:
                continue
            # the score for the target is minimized
            # by second pass
            if adict[i] in used and passed:
                aarel = np.inf
            else:
                aarel = arel[i]
            if bdict[i] in used and passed:
                bbrel = np.inf
            else:
                bbrel = brel[i]

            if adict[i] != bdict[i]:
                if aarel < bbrel:
                    toadd = adict[i]
                    mrel = aarel
                else:
                    toadd = bdict[i]
                    mrel = bbrel
            else:
                toadd = adict[i]
                mrel = min(aarel, bbrel)

            # check to see if already mapped
            if toadd in used:
                if mrel < used[toadd]:
                    # unmap prev src
                    # remap target
                    used[toadd] = mrel
                    # map new src
                    rrel[i] = mrel
                    ret[i] = toadd
            else:
                # mapt target
                used[toadd] = mrel
                # map src
                rrel[i] = mrel
                ret[i] = toadd
        if passed: # branch into this after one complete loop over inter
            for i in inter:
                if i not in rrel:
                    print("no vote gives necessary pairing:", ret)
                    raise PrecisionLossError
        passed = True
    assert not collision_check(ret), ret
    assert rrel, (ret, used, rrel)
    rrel = conv_dict_to_list(rrel)
    assert rrel
    return ret, rrel, None

def conv_dict_to_list(rrel):
    """Convert dict to list"""
    ret = []
    for i in sorted(list(rrel)):
        ret.append(rrel[i])
    return ret

def del_maxrel(rel):
    """We can sort by process of elimination
    delete the max relative difference
    replace with second highest
    """
    rel = list(rel)
    mmax = max(rel)
    idx = rel.index(mmax)
    rel2 = list(np.delete(rel, idx))
    mmax2 = max(rel2)
    rel[idx] = mmax2
    return rel

def partial_id_check(rdict, keys):
    """All the keys in rdict are mapped to themselves
    (identity map over keys)"""
    ret = True
    for i in keys:
        if i in rdict:
            if rdict[i] != i:
                ret = False
    return ret

#def filter_dict(sdict, good_keys):
#    """Get rid of all keys which are not in the good key list"""
#    ret = {}
#    for i in sdict:
#        if i in good_keys:
#            ret[i] = sdict[i]
#    return ret

def make_id(mlen):
    """Get the identity mapping"""
    ret = {}
    if hasattr(mlen, '__iter__'):
        mlen = len(mlen)
    for i in range(mlen):
        ret[i] = i
    assert isid(ret), str(ret)
    return ret

def isid(dot_map):
    """Check if map is identity"""
    ret = True
    for i in dot_map:
        if dot_map[i] != i:
            ret = False
    return ret

def reset_sortevals():
    """Reset function variables"""
    sortevals.sorted_evecs = {}
    sortevals.sorted_evals = {}
    sortevals.last_time = None
    sortevals.config = None

def update_sorted_evecs(newe, timeij, config_num):
    """Update sortevals.sorted_evecs; perform stability check
    (evecs should be stable from time slice to time slice
    otherwise sorting by dot product is not reliable)"""
    if timeij not in sortevals.sorted_evecs:
        sortevals.sorted_evecs[timeij] = []
        assert sortevals.last_time == timeij - 1, (
            timeij, sortevals.last_time)
    sortevals.sorted_evecs[timeij].append(np.copy(newe))
    assert np.all(sortevals.sorted_evecs[timeij][config_num] == newe),\
        (newe, config_num, sortevals.sorted_evecs[timeij][config_num])

def update_sorted_evals(newe, timeij, config_num):
    """Update sortevals.sorted_evals; perform stability check
    (evecs should be stable from time slice to time slice
    otherwise sorting by dot product is not reliable)"""
    if timeij not in sortevals.sorted_evals:
        sortevals.sorted_evals[timeij] = []
        assert sortevals.last_time == timeij - 1, (
            timeij, sortevals.last_time)
    sortevals.sorted_evals[timeij].append(np.copy(newe))
    assert np.all(sortevals.sorted_evals[timeij][config_num] == newe),\
        (newe, config_num, sortevals.sorted_evals[timeij][config_num])


def select_sorted_evecs(config_num, timeij):
    """Select evecs to use"""
    sortevals.config = config_num
    sortevals.last_time = timeij - 1

def fallback_sort(evals, evecs=None, reverse=None):
    """The usual sorting procedure for the eigenvalues"""
    evals = list(evals)

    ind = []
    for i, val in enumerate(evals):
        if val < 0 and LOGFORM:
            #ret[i] += np.inf
            ind.append(i)

    if reverse is None:
        sortrev = not LOGFORM
    else:
        sortrev = reverse

    if evecs is not None:
        evecs = [x for y, x in sorted(zip(evals, evecs.T), reverse=sortrev)]
        evecs = np.array(evecs).T
        evals = [y for y, x in sorted(zip(evals, evecs.T), reverse=sortrev)]
    else:
        evals = sorted(evals, reverse=sortrev)
    for i, _ in enumerate(ind):
        evals[-(i+1)] = -1
    evals = np.asarray(evals)
    ret = evals if evecs is None else (evals, evecs)
    return ret

def check_map_dups(dot_map):
    """Check for duplicates in evec dot map"""
    ret = False
    for i in dot_map:
        for j in dot_map:
            if i != j and dot_map[i] == dot_map[j]:
                ret = True
    return ret
