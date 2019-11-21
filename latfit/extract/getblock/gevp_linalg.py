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
        print(cmat)
        raise EigenvalueSignInconsistency
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

def cmatdot(cmat, vec, transp=False, sloppy=False):
    """Dot gevp matrix into vec on rhs if not transp"""
    cmat = np.asarray(cmat)
    cmat = cmat.T if transp else cmat
    vec = np.asarray(vec)
    assert len(vec) == len(cmat), str(vec)+" "+str(cmat)
    ret = np.zeros(vec.shape)
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
        assert np.allclose(dotprod, 1.0, rtol=1e-8),\
            str(dotprod)
    if GEVP_DEBUG:
        print("evecs of avg gevp",
              np.real(evecs_mean_t))

def update_sort_evecs(evecs, presort_evals, postsort_evals):
    """Update the sort of the eigenvectors"""
    assert None, "dones't work"
    for i, eval1 in enumerate(presort_evals):
        for j, eval2 in enumerate(postsort_evals):
            if j <= i:
                continue
            if eval1 == eval2:
                pre = list(np.copy(evecs[i]))
                post = list(np.copy(evecs[j]))
                evecs[j] = pre
                evecs[i] = post
    evecs = np.asarray(evecs)
    return evecs

def ratio_evals(evecs, c_lhs, c_rhs):
    """Get the ratio evals from the evecs and C(t), C(t_0)"""
    ret = []
    for i in evecs:
        assert len(i) == len(c_lhs), str(c_lhs)+" "+str(i)
        assert len(i) == len(c_rhs), str(c_rhs)+" "+str(i)
        ret.append(bracket(i, c_lhs)/bracket(i, c_rhs))
    ret = np.asarray(ret)
    return ret

def map_evals(evals_from, evals_to, debug=False):
    """Get similarity mapping"""
    ret = {}
    used = set()
    leval = len(evals_from)
    assert leval == len(evals_to), str(
        evals_from)+" "+str(evals_to)
    assert list(evals_from) == list(fallback_sort(evals_from))
    evals_to_sorted = fallback_sort(evals_to)
    eleft = np.roll(evals_to_sorted, -1)
    eright = np.roll(evals_to_sorted, 1)
    rel_diff = np.abs(evals_from - evals_to_sorted)/evals_from
    rdleft = np.abs(evals_from - eleft)/evals_from
    rdright = np.abs(evals_from - eright)/evals_from

    fallback = False # unambiguous mapping
    test_arr = [1 if i > 0.01 else 0 for i in rel_diff]
    for idx, item in enumerate(test_arr):
        lcomp = np.real(rdleft[idx]/rel_diff[idx])
        rcomp = np.real(rdright[idx]/rel_diff[idx])
        if not idx and debug and False:
            print(lcomp, rcomp)
            print('evals_to_sorted', evals_to_sorted)
            print('rel_diff', rel_diff, 'idx', idx)
            print('rdleft', rdleft)
            print('rdright', rdright)
        if item: # compare to neighbors to see if there's a better candidate
            if lcomp > 3 and rcomp > 3: # neighbors are 3x worse
                if debug and False:
                    print(lcomp, rcomp)
                    print('evals_to_sorted', evals_to_sorted)
                    print('rel_diff', rel_diff, 'idx', idx)
                    print('rdleft', rdleft)
                    print('rdright', rdright)
                    print("flipped to", False)
                test_arr[idx] = 0
        else:
            # comparison may be good, but neighbor also matches
            if lcomp <= 3 or rcomp <= 3:
                if debug and False:
                    print(lcomp, rcomp)
                    print('evals_to_sorted', evals_to_sorted)
                    print('rel_diff', rel_diff, 'idx', idx)
                    print('rdleft', rdleft)
                    print('rdright', rdright)
                    print("flipped to", True)
                test_arr[idx] = 1
    # > 1 since getting one eigenvalue wrong still gives a good sort
    test = np.sum(test_arr) == len(evals_to)
    test2 = np.sum(test_arr) > 1
    test3 = np.sum(test_arr) <= 1
    ret = make_id(leval)
    if debug:
        print('test_arr', test_arr)
    evals_to = list(evals_to)
    fallback_level = -1
    unambig_indices = set()
    if test:
        fallback_level = 2 # ambiguous mapping
    elif test2:
        fallback_level = 1
        mapped_from = set()
        mapped_to = set()
        for idx, (i, j) in enumerate(zip(evals_to_sorted, evals_to)):
            fidx = evals_to.index(i)
            if not test_arr[idx]:
                tidx = evals_to.index(j)
                mapped_from.add(fidx)
                mapped_to.add(tidx)
                ret[fidx] = tidx
                unambig_indices.add(fidx)
        assert len(mapped_from) == len(mapped_to)
        tot = set(range(len(evals_to)))
        for i, j in zip(sorted(list(tot-mapped_from)),
                        sorted(list(tot-mapped_to))):
            ret[i] = j
    elif test3:
        fallback_level = 0
        unambig_indices = set(range(len(evals_to)))
        if debug and False:
            print('fr', evals_to_sorted)
            print('to', evals_to)
        for _, (i, j) in enumerate(zip(evals_to_sorted, evals_to)):
            fidx = evals_to.index(i)
            tidx = evals_to.index(j)
            ret[fidx] = tidx
    assert fallback_level >= 0, str(test_arr)
    assert not check_map_dups(ret), str(ret)+" "+str(test_arr)
    assert len(ret) == leval, str(ret)
    return ret, fallback_level, unambig_indices

def make_id(mlen):
    """Get the identity mapping"""
    ret = {}
    for i in range(mlen):
        ret[i] = i
    assert isid(ret), str(ret)
    return ret


def sortevals(evals, evecs=None, c_lhs=None, c_rhs=None):
    """Sort eigenvalues in order of increasing energy"""
    evals = list(evals)
    ret = fallback_sort(evals, evecs)
    if evecs is not None:
        evals = ret[0]
        if evecs is None:
            raise
        evecs = ret[1]
        evals = list(evals)
    dot_map = make_id(len(evals))
    debug = False
    timeij_start = None
    if sortevals.last_time is not None and c_lhs is not None\
       and c_rhs is not None and not np.any(np.isnan(evals)):
        count = 5
        timeij_start = sortevals.last_time
        timeij = sortevals.last_time
        if timeij + 1 == 13:
            debug = False
        #if debug or timeij + 1 == 13:
        #    print("\nevals init", evals)
        fallback = True
        votes = []
        soft_votes = []
        while timeij in sortevals.sorted_evecs:

            # loop increment
            count -= 1 # 3, 2, 1
            if debug and False:
                print('count', count)

            evecs_past = sortevals.sorted_evecs[timeij][sortevals.config]
            if debug and False:
                print("evecs(", timeij, ") =", evecs_past) 
            assert len(evecs_past) == len(evals), str(evecs_past)
            assert len(evecs_past[0]) == len(evals), str(evecs_past[0])
            assert len(evecs_past[0]) == len(c_lhs), str(
                evecs_past[0])+" "+str(c_lhs)
            timeij -= 1 # t-1, t-2, t-3
            #if debug:
                #assert timeij in sortevals.sorted_evecs
            evals_from = np.copy(evals)
            evals_to = ratio_evals(evecs_past, c_lhs, c_rhs)
            if debug:
                #print("evals from", evals_from)
                print("evals to", evals_to)
            vote_map, fallback_level, unambig_indices = map_evals(
                evals_from, evals_to, debug=debug)
            if debug and False:
                print('vote map', vote_map)
                print('fallback', fallback_level)

            # unambiguous different mapping; accumulate two identical votes
            # to resort the eigenvalues
            # ambiguous sorting, throw out this map
            if fallback_level == 2:
                continue
            elif fallback_level == 1:
                soft_votes.append((vote_map, unambig_indices))
            else:
                assert not fallback_level
                votes.append((vote_map, unambig_indices))
            if debug:
                print('votes', votes)
                print('soft_votes', soft_votes)

            if len(votes)+np.floor(len(soft_votes)/2) == 3:
                break

            #if len(votes) == 2: # two unambiguous votes is arbitrary
                #if debug:
                    #print("good votes")
                #break
        if len(votes) >= 3:
            dot_map = votes_to_map(votes, debug=debug)
        elif len(votes) + np.floor(len(soft_votes)/2) >= 3:
            stop_votes_len = len(votes)+np.floor(len(soft_votes)/2)
            stop_votes_len = int(stop_votes_len)
            if debug:
                print("votes", votes, "soft_votes", soft_votes)
            votes.extend(soft_votes)
            dot_map = votes_to_map(votes, stop=stop_votes_len, debug=debug)
        elif len(votes) >= 2:
            dot_map = votes_to_map(votes, debug=debug)
        if debug:
            print("final votes", votes)
            print("final dot map:", dot_map)
    exitp = False
    if not isid(dot_map):
        exitp = True
        sevals = np.zeros(len(evals))
        sevecs = np.zeros(np.asarray(evecs).T.shape)
        for i in dot_map:
            sevals[i] = drop0imag(evals[dot_map[i]])
            sevecs[i] = drop0imag(evecs.T[dot_map[i]])
        ret = (sevals, sevecs.T)
    if evecs is not None and not np.any(np.isnan(evals)):
        if debug:
            print("evals final", ret[0])
    elif not np.any(np.isnan(evals)):
        if debug:
            print("evals final", ret)
    if debug and False:
        sys.exit()
    return ret
sortevals.sorted_evecs = {}
sortevals.last_time = None
sortevals.config = None

def votes_to_map(votes, stop=np.inf, debug=False):
    """Get sorting map based on previous time slice votes
    for what each one thinks is the right ordering"""
    ret, _ = votes[0]
    stop = len(votes) if stop >= len(votes) else stop
    votes = votes[:stop]
    for i, idxsi in votes:
        i = filter_dict(i, idxsi)
        for j, idxsj in votes:
            j = filter_dict(j, idxsj)
            disagree = partial_compare_dicts(i, j)
            if disagree:
                agree = set(idxsi).intersection(idxsj)
                agree = agree-set(disagree)
                allid = partial_id_check(ret, agree)
                if allid:
                    ret = make_id(ret)
    return ret

def partial_id_check(rdict, keys):
    """All the keys in rdict are mapped to themselves
    (identity map over keys)"""
    ret = True
    for i in keys:
        if i in rdict:
            if rdict[i] != i:
                ret = False
    return ret

def partial_compare_dicts(adict, bdict):
    """Compare common entries in two dictionaries"""
    aset = set(adict)
    bset = set(bdict)
    inter = aset.intersection(bset)
    ret = []
    for i in sorted(list(inter)):
        if adict[i] != bdict[i]:
            ret.append(i)
    return ret

def filter_dict(sdict, good_keys):
    """Get rid of all keys which are not in the good key list"""
    ret = {}
    for i in sdict:
        if i in good_keys:
            ret[i] = sdict[i]
    return ret

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

def get_evec_map(evs1, evs2):
    """Get similarity mapping between sets of eigenvectors"""
    map1 = evec_max_op_dimension(evs1)
    map2 = evec_max_op_dimension(evs2)
    dot_map = combine_maps(map1, map2)
    if not isid(dot_map):
        print(dot_map)
        print(evs1)
        print(evs2)
    return dot_map

def combine_maps(frommap, tommap):
    """Compose one-to-one maps:  roughly, from(i,j)*to(j, i)"""
    ret = {}
    for i in frommap:
        val = frommap[i]
        for j in tommap:
            if tommap[j] == val:
                ret[i] = j
    return ret


def evec_max_op_dimension(evs):
    """Get evec/eval->operator mapping based on relative overlaps"""
    opmap = {}
    evs = np.asarray(evs)
    for i in range(evs.shape[0]):
        dim = evs[:, i]
        idx = list(np.abs(dim)).index(max(np.abs(dim)))
        opmap[i] = idx
    # assert map is one-to-one (injective)
    assert not check_map_dups(opmap), str(opmap)+" "+str(evs)
    return opmap

def delete_max(evec):
    """Delete the max element of the eigenvector"""
    maxe = 0
    ind = None
    for i, elem in enumerate(evec):
        maxe = max(np.abs(maxe), np.abs(elem))
        if np.abs(elem) == maxe:
            ind = i
    ret = np.delete(evec, ind)
    return ret


def max_sign(evec):
    """Get the sign of the eigenvector's largest element"""
    maxe = 0
    for i in evec:
        maxe = max(np.abs(maxe), np.abs(i))
        if np.abs(i) == maxe:
            ret = np.sign(i)
    return ret

def reset_sortevals():
    """Reset function variables"""
    sortevals.sorted_evecs = {}
    sortevals.last_time = None

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

def select_sorted_evecs(config_num, timeij):
    """Select evecs to use"""
    sortevals.config = config_num
    sortevals.last_time = timeij - 1

def fallback_sort(evals, evecs=None):
    """The usual sorting procedure for the eigenvalues"""
    evals = list(evals)

    ind = []
    for i, val in enumerate(evals):
        if val < 0 and LOGFORM:
            ret[i] += np.inf
            ind.append(i)

    sortrev = False if LOGFORM else True

    if evecs is not None:
        evecs = [x for y,x in sorted(zip(evals, evecs.T), reverse=sortrev)]
        evecs = np.array(evecs).T
        evals = [y for y,x in sorted(zip(evals, evecs.T), reverse=sortrev)]
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
            "precision loss detected:"+str(decrease_var)
    except AssertionError:
        #print('avg =', avg)
        #print('ret =', ret)
        #print('check =', check)
        #print('orig =', orig)
        print("precision loss detected, orig != check")
        raise PrecisionLossError
    return ret
