""""Get the data block."""
import sys
from collections import deque
from math import fsum
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
from latfit.config import GEVP
from latfit.config import ELIM_JKCONF_LIST
from latfit.config import NORMS, GEVP_DEBUG, USE_LATE_TIMES
from latfit.config import BINNUM
from latfit.config import STYPE
from latfit.config import PIONRATIO, ADD_CONST_VEC
from latfit.config import MATRIX_SUBTRACTION
from latfit.config import DECREASE_VAR
from latfit.config import HINTS_ELIM

from mpi4py import MPI

MPIRANK = MPI.COMM_WORLD.rank

if MATRIX_SUBTRACTION and GEVP:
    ADD_CONST_VEC = [0 for i in ADD_CONST_VEC]

XMAX = 999

if PIONRATIO and GEVP:
    PIONSTR = ['pioncorrChk_mom'+str(i)+'unit'+('s' if i != 1 else '') for i in range(2)]
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
    assert np.allclose(np.matrix(carr).H, carr, rtol=1e-8), "hermiticity enforcement failed."

def removerowcol(cmat, idx):
    """Delete the idx'th row and column"""
    return np.delete(np.delete(cmat, idx, axis=0), idx, axis=1)

def solve_gevp(c_lhs, c_rhs=None):
    dimops_orig = len(c_lhs)
    dimops = len(c_lhs)
    if solve_gevp.hint is not None:
        dimremaining, toelim  = solve_gevp.hint
    else:
        dimremaining, toelim  = (-1, -2)
    assert toelim < dimremaining, "index error"

    eigvals, evecs = scipy.linalg.eig(c_lhs, c_rhs, overwrite_a=False,
                                      overwrite_b=False, check_finite=True)
    sorted(eigvals, reverse=True)
    if dimops == dimremaining:
        assert eigvals[toelim] > 0, "already negative"
        eigvals[toelim] *= -1
    while any(eigvals < 0):
        dimops -= 1
        assert dimops > 0, "dimension reduction technique exhausted."
        count = 0
        dimdeldict = {}
        for dimdel in range(dimops+1):
            if dimdel == 0 and toelim < 0: # heuristic
                continue
            c_lhs_temp = removerowcol(c_lhs, dimdel)
            c_rhs_temp = removerowcol(c_rhs, dimdel) if c_rhs is not None else c_rhs
            eigvals, evecs = scipy.linalg.eig(c_lhs_temp, c_rhs_temp, overwrite_a=False,
                                              overwrite_b=False, check_finite=True)
            if dimremaining == dimops:
                assert eigvals[toelim] > 0, "already negative"
                eigvals[toelim] *= -1
            count = max(np.count_nonzero(eigvals > 0), count)
            #assert count not in dimdeldict, "degeneracy of deletions:"+str(count)+","+str(dimdel)+","+str(dimdeldict[count])
            dimdeldict[count] = dimdel
        if dimdeldict:
            dimdel = dimdeldict[max([count for count in dimdeldict])]
            c_lhs = removerowcol(c_lhs, dimdel)
            c_rhs = removerowcol(c_rhs, dimdel) if c_rhs is not None else c_rhs
        eigvals, evecs = scipy.linalg.eig(c_lhs, c_rhs, overwrite_a=False,
                                          overwrite_b=False, check_finite=True)
        if dimremaining == dimops:
            assert eigvals[toelim] > 0, "already negative"
            eigvals[toelim] *= -1
        for _ in range(dimops_orig-dimops):
            eigvals = np.append(eigvals, np.nan)
    assert len(eigvals) == dimops_orig, "eigenvalue shape extension needed"
    return eigvals, evecs
solve_gevp.hint = None


def get_eigvals(c_lhs, c_rhs, overb=False, print_evecs=False, commnorm=False):
    """get the nth generalized eigenvalue from matrices of files
    file_tup_lhs, file_tup_rhs
    optionally, overwrite the rhs matrix we get if we don't need it anymore (overb=True)
    """
    checkherm(c_lhs)
    checkherm(c_rhs)
    overb = False # completely unnecessary and dangerous speedup
    print_evecs = False if not GEVP_DEBUG else print_evecs
    if not get_eigvals.sent and print_evecs:
        print("First solve, so printing norms which are multiplied onto GEVP entries.")
        print("e.g. C(t)_ij -> Norms[i][j]*C(t)_ij")
        print("Norms=", NORMS)
        get_eigvals.sent = True
    eigvals, evecs = solve_gevp(c_lhs, c_rhs)
    checkgteq0(eigvals)
    dimops = len(c_lhs)
    late = False if all(np.imag(eigvals) == 0) else True
    skip_late = False
    try:
        c_rhs_inv = linalg.inv(c_rhs)
        # compute commutator divided by norm to see how close rhs and lhs bases
        try:
            commutator_norms = (np.dot(c_rhs_inv, c_lhs)-np.dot(c_lhs, c_rhs_inv))
        except FloatingPointError:
            print("bad denominator:")
            print(np.linalg.norm(c_rhs_inv))
            print(np.linalg.norm(c_lhs))
            print(c_lhs)
            raise FloatingPointError
        assert np.allclose(np.dot(c_rhs_inv, c_rhs), np.eye(dimops), rtol=1e-8), "Bad C_rhs inverse. Numerically unstable."
        assert np.allclose(np.matrix(c_rhs_inv).H, c_rhs_inv, rtol=1e-8), "Inverse failed (result is not hermite)."
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
    eigfin = np.zeros((len(eigvals)), dtype=np.float)
    for i, j in enumerate(eigvals):
        if j.imag == 0:
            eigfin[i] = eigvals[i].real
        else:
            if USE_LATE_TIMES and not skip_late and late:
                eigvals, evecs = solve_gevp(c_lhs_new)
                break
            else:
                raise ImaginaryEigenvalue

    for i, j in enumerate(eigvals):
        if j.imag == 0:
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
    checkgteq0(eigfin)
    ret = (eigfin, commutator_norm, commutator_norms) if commnorm else eigfin
    return ret
get_eigvals.sent = False

def checkgteq0(eigfin):
    for i in eigfin:
        if not np.isnan(i):
            assert i >= 0, "non-positive eigenvalue found:"+str(eigfin)

def enforce_hermiticity(gevp_mat):
    """C->(C+C^\dagger)/2"""
    return (np.conj(gevp_mat).T+gevp_mat)/2


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
            "precision loss detected:"+str(orig)+" "+str(check)+" "+str(decrease_var)
    except AssertionError:
        print("precision loss detected: (check, orig, avg, ret):")
        print(check)
        print(orig)
        print(avg)
        print(ret)
        sys.exit(1)
    return ret


if EFF_MASS:
    def getblock_gevp(file_tup, delta_t, timeij=None, decrease_var=DECREASE_VAR):
        """Get a single rhs; loop to find the one with the least nan's"""
        blkdict = {}
        countdict = {}
        errdict = {}
        check_length = 0
        if hasattr(delta_t, '__iter__'):
            for i, dt1 in enumerate(delta_t):
                solve_gevp.hint = HINTS_ELIM[timeij] if timeij in HINTS_ELIM else None
                print('dt1' , dt1, 'timeij', timeij)
                print('hint=', solve_gevp.hint)
                try:
                    assert len(file_tup[1]) == len(delta_t), "rhs times dimensions do not match:"+\
                        len(file_tup[1])+str(",")+len(delta_t)
                except TypeError:
                    print(file_tup)
                    print(delta_t)
                    assert None, "bug"
                argtup = (file_tup[0], file_tup[1][i], file_tup[2], file_tup[3])
                for i,j in enumerate(argtup):
                    assert j is not None, "file_tup["+str(i)+"] is None"
                blkdict[dt1] = getblock_gevp_singlerhs(argtup, dt1, timeij, decrease_var)
                check_length = len(blkdict[dt1])
                relerr = np.abs(np.std(blkdict[dt1], ddof=1, axis=0)*(len(blkdict[dt1])-1)/np.mean(blkdict[dt1], axis=0))
                errdict[dt1] = np.nanargmax(relerr)
                count = 0
                for config, blk in enumerate(blkdict[dt1]):
                    count = max(np.count_nonzero(~np.isnan(blk)), count)
                    countdict[count] = dt1
            keymax = countdict[max([count for count in countdict])]
            print("final tlhs, trhs =", timeij, timeij-keymax, "next hint:(",
                  np.count_nonzero(~np.isnan(blkdict[keymax][0])), ",", errdict[keymax], ")")
            for key in blkdict:
                assert len(blkdict[key]) == check_length, "number of configs is not consistent for different times"
            ret = blkdict[keymax]
        else:
            print("final tlhs, trhs =", timeij, timeij-delta_t)
            solve_gevp.hint = None
            ret = getblock_gevp_singlerhs(
                file_tup, delta_t, timeij=timeij, decrease_var=decrease_var)
        return ret


    def getblock_gevp_singlerhs(file_tup, delta_t, timeij=None, decrease_var=DECREASE_VAR):
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
        retblk = deque()
        dimops = len(file_tup[0])
        if STYPE == 'ascii':
            num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        elif STYPE == 'hdf5':
            num_configs = len(file_tup[0][0][0])
        if GEVP_DEBUG:
            print("Getting block for time slice=", timeij)
        check_variance = []

        cmat_lhs_t, cmat_lhs_t_mean = readin_gevp_matrices(file_tup[0], num_configs)
        cmat_rhs, cmat_rhs_mean = readin_gevp_matrices(file_tup[1], num_configs)
        cmat_lhs_tp1, cmat_lhs_tp1_mean = readin_gevp_matrices(file_tup[2], num_configs)
        cmat_lhs_tp2, cmat_lhs_tp2_mean = readin_gevp_matrices(file_tup[3], num_configs)
        cmat_lhs_tp3, cmat_lhs_tp3_mean = readin_gevp_matrices(file_tup[3], num_configs)

        eigvals_mean_t = sorted(get_eigvals(
            cmat_lhs_t_mean, cmat_rhs_mean), reverse=True)
        eigvals_mean_tp1 = sorted(get_eigvals(
            cmat_lhs_tp1_mean, cmat_rhs_mean), reverse=True)
        eigvals_mean_tp2 = sorted(get_eigvals(
            cmat_lhs_tp2_mean, cmat_rhs_mean), reverse=True)
        eigvals_mean_tp3 = sorted(get_eigvals(
            cmat_lhs_tp3_mean, cmat_rhs_mean), reverse=True)

        avg_energies = np.array([proc_meff(
            (1/eigvals_mean_t[op], 1, eigvals_mean_tp2[op],
             eigvals_mean_tp3[op]), index=op, time_arr=timeij)
                         for op in range(dimops)])/delta_t

        norm_comm = []
        norms_comm = []
        for num in range(num_configs):
            if GEVP_DEBUG:
                print("config #=", num)
            tprob = timeij
            try:
                eigret = get_eigvals(cmat_lhs_t[num], cmat_rhs[num], print_evecs=True, commnorm=True)
                norm_comm.append(eigret[1])
                norms_comm.append(eigret[2])
                eigvals = np.array(sorted(eigret[0], reverse=True))

                tprob = None if not EFF_MASS else tprob

                eigvals2 = np.array(sorted(get_eigvals(
                    cmat_lhs_tp1[num], cmat_rhs[num]), reverse=True))

                eigvals3 = np.array(sorted(get_eigvals(
                    cmat_lhs_tp2[num], cmat_rhs[num]), reverse=True))

                eigvals4 = np.array(sorted(get_eigvals(
                    cmat_lhs_tp3[num], cmat_rhs[num],
                    overb=True), reverse=True))

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
            check_variance.append(eigvals)
            result = variance_reduction(np.array([proc_meff(
                (1/eigvals[op], 1, eigvals3[op], eigvals4[op]),
                index=op, time_arr=timeij) for op in range(dimops)])/delta_t, avg_energies, 1/decrease_var)
            retblk.append(result)
        if MPIRANK == 0:
            pass
            #chisq_bad, pval, dof = pval_commutator(norms_comm)
            #print("average commutator norm, (t =", timeij, ") =", np.mean(norm_comm), "chi^2/dof =", chisq_bad, "p-value =", pval, "dof =", dof)
        if GEVP_DEBUG:
            print("time, avg evals, variance of evals:",
                  timeij, jack_mean_err(np.array(check_variance)))
            if timeij == 10:
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
        chisq_arr.append(fsum([i[j]**2/sample_stddev[j]**2 for j in range(dof)]))
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

    def getblock_plus(file_tup, reuse, timeij=None):
        """get the block"""
        return getblock_simple(file_tup, reuse, timeij)


def getblock(file_tup, reuse, timeij=None, delta_t=None):
    """get the block and subtract any bad configs"""
    retblk = np.array(getblock_plus(file_tup, reuse, timeij, delta_t=delta_t))
    if ELIM_JKCONF_LIST:
        retblk = elim_jkconfigs(retblk)
    if BINNUM != 1:
        retblk = binconf(retblk)
    return retblk
