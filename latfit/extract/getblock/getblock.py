""""Get the data block, process eigenvalues into energies.
all getblock module energy related functionality is contained here
"""
import sys
from os import path
from collections import deque
import pickle
import h5py
import numpy as np

from latfit.utilities import exactmean as em
from latfit.mathfun.proc_meff import proc_meff
#from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.mathfun.binconf import binconf
from latfit.extract.proc_line import proc_line
from latfit.analysis.errorcodes import XmaxError, NegativeEigenvalue
from latfit.analysis.errorcodes import PrecisionLossError, NegativeEnergy
from latfit.analysis.errorcodes import ImaginaryEigenvalue, XminError
from latfit.analysis.errorcodes import EigenvalueSignInconsistency
from latfit.analysis.superjack import jack_mean_err

from latfit.extract.getblock.gevp_linalg import sterr, checkgteq0
from latfit.extract.getblock.gevp_linalg import degenerate_subspace_check
from latfit.extract.getblock.gevp_linalg import enforce_hermiticity, norms
from latfit.extract.getblock.gevp_linalg import checkherm, posdef_diag_check
from latfit.extract.getblock.gevp_solve import allowedeliminations
from latfit.extract.getblock.gevp_linalg import variance_reduction
from latfit.extract.getblock.gevp_pionratio import atwsub_cmats, modenergies
import latfit.extract.getblock.disp_hacks as gdisp
import latfit.extract.getblock.gevp_solve as gsolve
import latfit.extract.getblock.gevp_linalg as glin

from latfit.config import EFF_MASS, PIONRATIO, MULT
from latfit.config import GEVP, DELETE_NEGATIVE_OPERATORS
from latfit.config import GEVP_DEBUG, GEVP_DERIV, STYPE
from latfit.config import DECREASE_VAR, ISOSPIN
from latfit.config import HINTS_ELIM, VERBOSE
from latfit.config import REINFLATE_BEFORE_LOG

#MPIRANK = MPI.COMM_WORLD.rank

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
    return np.complex(filetup[num-1])
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


if GEVP_DEBUG:
    def final_gevp_debug_print(timeij, num_configs):
        """Print final info for this particular timeij"""
        check_variance = final_gevp_debug_print.check_variance
        avg_evecs = final_gevp_debug_print.avg_evecs[timeij]
        check_variance = np.asarray(check_variance)
        print("average evecs:")
        for i in list(zip(avg_evecs/num_configs,
                          norms(avg_evecs/num_configs))):
            print(i)
        if not np.any(np.isnan(check_variance)):
            error_check = jack_mean_err(check_variance)
        else:
            error_check = None
        print("time, avg evals, variance of evals:",
              timeij, error_check)
    final_gevp_debug_print.check_variance = None
    final_gevp_debug_print.avg_evecs = {}

else:
    def final_gevp_debug_print(*xargs):
        """print nothing"""
        if xargs:
            pass
    final_gevp_debug_print.check_variance = None
    final_gevp_debug_print.avg_evecs = {}


def readin_gevp_matrices(file_tup, num_configs, decrease_var=DECREASE_VAR, docheck=False):
    """Read in the GEVP matrix
    """
    decrease_var = 0 if decrease_var is None else decrease_var
    dimops = len(file_tup)
    cmat = np.zeros((num_configs, dimops, dimops), dtype=complex)
    for num in range(num_configs):
        for opa in range(dimops):
            for opb in range(dimops):
                corr = getline_loc(
                    file_tup[opa][opb], num+1)*gdisp.NORMS[opa][opb]
                assert isinstance(corr, (np.complex, str))
                # takes the real
                cmat[num][opa][opb] = proc_line(corr, file_tup[opa][opb])
                if opa == opb and opa == 1 and not num:
                    pass
                    #print(cmat[num][opa][opb], corr)
                if opa != opb and ISOSPIN != 0:
                    pass
                    #assert cmat[num][opa][opb] > 0\
                    #    or 'sigma' in GEVP_DIRS[opa][opb], \
                    #    str(corr)+str((opa, opb))
        #print(num, cmat[num][1, 0], cmat[num][0, 1])
        cmat[num] = enforce_hermiticity(cmat[num])
        #print(num, cmat[num][1, 0], cmat[num][0, 1])
        #sys.exit(0)
        #if ISOSPIN != 0 and not MATRIX_SUBTRACTION:
        #    pass
            #assert np.all(cmat[num] > 0), str(cmat[num])
        checkherm(cmat[num])
    if docheck:
        posdef_diag_check(cmat)
    mean = em.acmean(cmat, axis=0)
    checkherm(mean)
    if decrease_var != 1:
        cmat = variance_reduction(cmat, mean)
    return np.asarray(cmat), np.asarray(mean)


def gevp_block_checks(dt1, blkdict, errdict, countdict):
    """Perform some checks on the new gevp block"""
    check_length = len(blkdict[dt1])
    relerr = np.abs(em.acstd(blkdict[dt1], ddof=1, axis=0)*(
        len(blkdict[dt1])-1)/em.acmean(blkdict[dt1], axis=0))
    errdict[dt1] = 0
    if not all(np.isnan(relerr)):
        errdict[dt1] = np.nanargmax(relerr)
    count = 0
    for _, blk in enumerate(blkdict[dt1]):
        count = max(np.count_nonzero(~np.isnan(blk)), count)
        countdict[count] = dt1
    return errdict, countdict, check_length

def blkdict_dt1(index_info, file_tup, timeinfo,
                decrease_var=DECREASE_VAR):
    """get blkdict[dt1]"""
    idx, dt1 = index_info
    delta_t, timeij = timeinfo
    try:
        assert len(file_tup[1]) == len(delta_t), \
            "rhs times dimensions do not match:"+\
            len(file_tup[1])+str(", ")+len(delta_t)
    except TypeError:
        print(file_tup)
        print(delta_t)
        assert None, "bug"
    argtup = (file_tup[0], file_tup[1][idx], *file_tup[2:])
    for idx2, j in enumerate(argtup):
        assert j is not None, "file_tup["+str(idx2)+"] is None"
    ret = getblock_gevp_singlerhs(argtup, dt1, timeij,
                                  decrease_var)
    return ret


if EFF_MASS:
    def getblock_gevp(file_tup, delta_t, timeij=None,
                      decrease_var=DECREASE_VAR):
        """Get a single rhs; loop to find the one with the least nan's"""
        blkdict = {}
        countdict = {}
        errdict = {}
        check_length = 0
        assert timeij-delta_t >= 0, str((timeij, delta_t))
        gsolve.HINT = HINTS_ELIM[timeij] if timeij in HINTS_ELIM\
            else None
        assert len(file_tup) == 5, "bad file_tup length:"+str(len(file_tup))
        if hasattr(delta_t, '__iter__') and np.asarray(delta_t).shape:
            for i, dt1 in enumerate(delta_t):
                blkdict[dt1] = blkdict_dt1((i, dt1), file_tup,
                                           (delta_t, timeij),
                                           decrease_var)
                if VERBOSE:
                    print('dt1', dt1, 'timeij', timeij, 'elim hint',
                          gsolve.HINT,
                          "operator eliminations", allowedeliminations(),
                          'sample', blkdict[dt1][0])
                    for zdx, zamp in enumerate(blkdict[dt1]):
                        print('dt1', dt1, 'timeij', timeij,
                              "idx, examp:", zdx, zamp)
                errdict, countdict, check_length = gevp_block_checks(
                    dt1, blkdict, errdict, countdict)
            keymax = countdict[max(countdict)]
            print("final tlhs, trhs =", timeij, timeij-keymax, "next hint:(",
                  np.count_nonzero(~np.isnan(blkdict[keymax][0])),
                  ", ", errdict[keymax], ")")
            for key in blkdict:
                assert len(blkdict[key]) == check_length, \
                    "number of configs is not consistent"+\
                    " for different times"
            ret = blkdict[keymax]
        else:
            ret = getblock_gevp_singlerhs(
                file_tup, delta_t, timeij=timeij, decrease_var=decrease_var)
            relerr = np.abs(em.acstd(ret, ddof=1, axis=0)*(
                len(ret)-1)/em.acmean(ret, axis=0))
            if VERBOSE:
                print('dt1', delta_t, 'timeij', timeij, 'elim hint',
                      gsolve.HINT,
                      "operator eliminations", allowedeliminations(),
                      'sample', ret[0])
                for zdx, zamp in enumerate(ret):
                    print('dt1', delta_t, 'timeij', timeij, "idx, examp:",
                          zdx, zamp)
                print("final tlhs, trhs =", timeij,
                      timeij-delta_t, "next hint:(",
                      np.count_nonzero(~np.isnan(ret[0])), ", ",
                      np.nanargmax(relerr) if not all(np.isnan(relerr)) else 0,
                      ")")
        return ret


    def average_energies(mean_cmats_lhs, mean_crhs, delta_t, timeij):
        """get average energies"""
        cmat_lhs_t_mean = mean_cmats_lhs[0]
        cmat_lhs_tp1_mean = mean_cmats_lhs[1]
        # cmat_lhs_tp2_mean = mean_cmats_lhs[2]
        # cmat_lhs_tp3_mean = mean_cmats_lhs[3]
        while 1 < 2:
            eigvals_mean_t, evecs_mean_t = gsolve.get_eigvals(
                cmat_lhs_t_mean, mean_crhs)
            try:
                if DELETE_NEGATIVE_OPERATORS:
                    checkgteq0(eigvals_mean_t)
                break
            except AssertionError:
                assert None, "no longer supported"
                print("negative eigenvalues found")
                print('eigvals:', eigvals_mean_t)
                print("allowed operator eliminations:",
                      allowedeliminations())


        if GEVP_DERIV:
            eigvals_mean_tp1, _ = gsolve.get_eigvals(
                cmat_lhs_tp1_mean, mean_crhs)
            for i, eva1 in enumerate(eigvals_mean_tp1):
                if eva1 < 0:
                    eigvals_mean_tp1[i] = np.nan
            checkgteq0(eigvals_mean_tp1)
        else:
            eigvals_mean_tp1 = [np.nan]*len(eigvals_mean_t)
        #eigvals_mean_tp2 = gsolve.get_eigvals(cmat_lhs_tp2_mean, mean_crhs)
        eigvals_mean_tp2 = [np.nan]*len(eigvals_mean_t)
        #eigvals_mean_tp3 = gsolve.get_eigvals(cmat_lhs_tp3_mean, mean_crhs)
        eigvals_mean_tp3 = [np.nan]*len(eigvals_mean_t)
        if DELETE_NEGATIVE_OPERATORS:
            checkgteq0(eigvals_mean_tp1)
            checkgteq0(eigvals_mean_tp2)
            checkgteq0(eigvals_mean_tp3)

        degenerate_subspace_check(evecs_mean_t)
        avg_energies = gdisp.callprocmeff([eigvals_mean_t, eigvals_mean_tp1,
                                           eigvals_mean_tp2,
                                           eigvals_mean_tp3], timeij,
                                          delta_t, id_sort=False)
        for i, eng in enumerate(avg_energies):
            print('avg energy', i, ':', eng)

        return avg_energies, eigvals_mean_t, evecs_mean_t

    def getlhsrhs(file_tup, num_configs, decrease_var=DECREASE_VAR):
        """Get lhs and rhs gevp matrices from file_tup
        (extract from index structure)
        """
        mean_cmats_lhs = []
        cmats_lhs = []
        assert len(file_tup) == 5, "bad length:"+str(len(file_tup))
        for idx in range(5):
            cmat, mean = readin_gevp_matrices(file_tup[idx], num_configs, docheck=(
                not idx or idx==1), decrease_var=decrease_var)
            if idx == 1:
                cmat_rhs, mean_crhs = cmat, mean
            else:
                cmats_lhs.append(cmat)
                mean_cmats_lhs.append(mean)
        return cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs

    def setup_getblock_gevp_singlerhs(timeinfo, file_tup):
        """Perform time checks for getblock_gevp_singlerhs"""
        delta_t, timeij = timeinfo
        assert delta_t is not None, "delta_t is None"
        if not delta_t:
            assert None, "when does this still occur?"
            delta_t = 1.0
        if STYPE == 'ascii':
            num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        elif STYPE == 'hdf5':
            num_configs = len(file_tup[0][0][0])
        if GEVP_DEBUG:
            print("Getting block for time slice=", timeij)
        timeij = None if not EFF_MASS else timeij
        return num_configs, (delta_t, timeij)

    def dump_cmats(*cmats):
        """Dump GEVP matrices as pickle/hdf5 files"""
        lencheck1 = None # check on the number of configs
        lencheck2 = None # check on the column length
        lencheck3 = None # check on the row length
        done = set()
        for time, cmat_set in cmats:

            assert int(time) == float(time), time
            time = str(int(time))

            if time in done:
                print("GEVP matrix C("+str(time)+") already written to disk; skipping.")
                continue
            done.add(time)

            # setup
            start_str = 'cmat.t'+str(time)
            if isinstance(cmat_set[0][0][0], np.complex):
                noimag = True
                for i in cmat_set[0]:
                    if not noimag:
                        break
                    for j in i:
                        if np.imag(j) != 0.0:
                            noimag = False
                            break
                if noimag:
                    dtype = np.float64
                else:
                    dtype = np.complex128
            else:
                assert isinstance(cmat_set[0][0][0], np.float), cmat[0]
                dtype = np.float64
            print('dtype', dtype)
            arr = np.array(cmat_set, dtype=dtype)

            # checks
            lencheck1 = len(arr) if lencheck1 is None else lencheck1
            assert len(arr) == lencheck1, (lencheck1, len(arr))
            lencheck2 = len(arr[0]) if lencheck2 is None else lencheck2
            assert len(arr[0]) == lencheck2, (lencheck2, len(arr[0]))
            lencheck3 = len(arr[0].T) if lencheck3 is None else lencheck3
            assert len(arr[0].T) == lencheck3, (lencheck3, len(arr[0].T))

            # hdf5 write
            if path.exists(start_str+'.hdf5'):
                print("skipping write of file", start_str+'.hdf5', "since it already exists.")
            else:
                print("writing", start_str+'.hdf5') 
                fn1 = h5py.File(start_str+'.hdf5', 'w')
                fn1[time] = arr
                fn1.close()

            # pickle file write
            if path.exists(start_str+'.p'):
                print("skipping write of file", start_str+'.p', "since it already exists.")
            else:
                print("writing", start_str+'.p')
                gn1 = open(start_str+'.p', 'wb')
                pickle.dump(arr, gn1)
                gn1.close()

        print("finished writing GEVP matrices of times:", sorted([int(i) for i in done]))

    def getblock_gevp_singlerhs(file_tup, delta_t, timeij=None,
                                decrease_var=DECREASE_VAR):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        files_tup[2] is the t+1 lhs
        files_tup[3] is the t+2 lhs
        C(t)v = Eigval*C(t_0)v
        """
        # do the setup
        num_configs, (delta_t, timeij) = setup_getblock_gevp_singlerhs(
            (delta_t, timeij), file_tup)

        # get the gevp matrices
        ## variance decreased
        cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs = getlhsrhs(
            file_tup, num_configs)
        ## varianced not decreased
        cmats_lhs1, mean_cmats_lhs1, cmat_rhs1, mean_crhs1 = getlhsrhs(
            file_tup, num_configs, decrease_var=1.0)

        # do the around the world subtraction
        assert timeij-delta_t >= 0, str((timeij, delta_t))
        ## variance decreased
        cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs = atwsub_cmats(
            (delta_t, timeij),
            cmats_lhs, mean_cmats_lhs,
            cmat_rhs, mean_crhs)
        ## varianced not decreased
        cmats_lhs1, mean_cmats_lhs1, cmat_rhs1, mean_crhs1 = atwsub_cmats(
            (delta_t, timeij),
            cmats_lhs1, mean_cmats_lhs1,
            cmat_rhs1, mean_crhs1)

        # assumes delta_t is not None (i.e. t-t0 = const)
        # dump (jackknifed, non-variance-decreased) GEVP matrices to disk
        dump_cmats((timeij, cmats_lhs1[0]), (timeij-delta_t, cmat_rhs1))

        num = 0
        # reset the list of allowed operator eliminations at the
        # beginning of the loop

        brackets = []

        allowedeliminations(reset=True)
        #glin.reset_sortevals()
        while num < num_configs:
            if GEVP_DEBUG:
                print("config #=", num)
            try:
                glin.select_sorted_evecs(num, timeij)

                eigret = gsolve.get_eigvals(cmats_lhs[0][num], cmat_rhs[num],
                                            print_evecs=True, commnorm=True)

                print("n", num, cmats_lhs[0][num], cmat_rhs[num])
                if MULT > 1:
                    brackets.append([glin.bracket(eigret[1][i], cmats_lhs[0][num]) for i in range(MULT)])
                else:
                    brackets.append(glin.bracket(eigret[1][0], cmats_lhs[0][num]))
                eigret = np.asarray(eigret)
                if VERBOSE:
                    print("timeij, num, evals:", timeij, num, eigret[0])
                    #print("timeij, num, evecs:", timeij, num, eigret[1])

                glin.update_sorted_evecs(eigret[1], timeij, num)
                glin.update_sorted_evals(eigret[0], timeij, num)

                if num == 0:

                    # pickle the GEVP matrices, for debug purposes.
                    gn1 = open('timeij_'+str(timeij)+'.p', 'wb')
                    pickle.dump([cmats_lhs[0], cmat_rhs], gn1)
                    gn1.close()

                    gsolve.MEAN = None
                    avg_en_eig = average_energies(
                        mean_cmats_lhs, mean_crhs, delta_t, timeij)
                    gsolve.MEAN = avg_en_eig[
                        1] if REINFLATE_BEFORE_LOG else None
                    final_gevp_debug_print.avg_evecs[timeij] = np.zeros(
                        eigret[1].shape)
                    final_gevp_debug_print.check_variance = []
                    retblk = deque()

                # accumulate diagnostic information
                final_gevp_debug_print.check_variance.append(eigret[0])
                final_gevp_debug_print.avg_evecs[timeij] += np.real(
                    eigret[1])

                if continue_neg(eigret[0]):
                    num = 0
                    continue

                # get t+t, t_0 eigenvalues (for GEVP derivative wrt t)
                eigvals2 = eigvals_tplus_one(len(eigret[0]), num,
                                             cmats_lhs, cmat_rhs)

            except ImaginaryEigenvalue:
                print('config_num:', num, 'time:', timeij)
                if timeij is not None:
                    raise XmaxError(problemx=timeij)

            # process the eigenvalues into energies
            # don't sort the eigenvalues, as they are assumed sorted
            energies = gdisp.callprocmeff([eigret[0], eigvals2,
                                           [np.nan]*len(eigret[0]),
                                           [np.nan]*len(eigret[0])],
                                          timeij, delta_t, id_sort=False)
            if not REINFLATE_BEFORE_LOG:
                energies = variance_reduction(energies,
                                              avg_en_eig[0],
                                              1/decrease_var)
            if VERBOSE:
                print("timeij, num, energies:", timeij, num, energies)
            retblk.append(energies)
            num += 1

        # pion ratio
        retblk = modenergies(retblk, timeij, delta_t)
        retblk = deque(retblk)
        assert len(retblk) == num_configs, \
            "number of configs should be the block length"
        final_gevp_debug_print(timeij, num_configs)
        #glin.update_sorted_evecs(avg_en_eig[2], timeij)
        test_bracket_signal(brackets)
        gn1 = open('bracket_timeij_'+str(timeij)+'.p', 'wb')
        pickle.dump(brackets, gn1)
        gn1.close()
        test_ground_increase(retblk, timeij)
        return retblk

    if GEVP_DERIV:
        def eigvals_tplus_one(dimops, num, cmats_lhs, cmat_rhs):
            """get eigvals for GEVP derivative (t+1, t_0)"""
            eigvals2, _ = gsolve.get_eigvals(cmats_lhs[1][num],
                                             cmat_rhs[num])
            assert len(eigvals2) == dimops
            for i, eva1 in enumerate(eigvals2):
                if eva1 < 0:
                    eigvals2[i] = np.nan
            checkgteq0(eigvals2)
            return eigvals2
    else:
        def eigvals_tplus_one(dimops, *xargs):
            """Just return nan's.  We don't need it."""
            if xargs:
                pass
            eigvals2 = [np.nan]*dimops
            return eigvals2

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
                eigvals, _ = gsolve.get_eigvals(num,
                                                file_tup[0], file_tup[1])
            except ImaginaryEigenvalue:
                print(num, timeij)
                sys.exit(1)
            retblk.append(eigvals)
        return retblk

def test_ground_increase(retblk, timeij):
    """Test for increasing energy in the ground state.
    Throw an xmin error if it increases.
    """
    mean = em.acmean(retblk, axis=0)[0]
    err = em.acstd(retblk, axis=0)[0]*np.sqrt(len(retblk)-1)
    prev = test_ground_increase.mean
    if prev is not None and prev != mean:
        if not np.isnan(test_ground_increase.err):
            merr = max(err, test_ground_increase.err)
        else:
            merr = err
        test_ground_increase.mean = mean
        test_ground_increase.err = err
        statistically_significant = np.abs(
            mean-prev) > 1.5*merr
        if not np.isnan(prev):
            if prev < mean and statistically_significant:
                assert PIONRATIO
                if VERBOSE:
                    print("ground state is still increasing:")
                    print("prev, mean:", prev, mean)
                    print("increasing tmin to decay region")
                raise XminError(problemx=timeij-1)
            test_ground_increase.mean = None
test_ground_increase.mean = np.nan
test_ground_increase.err = np.nan


def grd_inc_reset():
    """Reset cache when beginning to reprocess"""
    test_ground_increase.mean = np.nan
    test_ground_increase.err = np.nan

def test_bracket_signal(brackets_in, decrease_var=DECREASE_VAR):
    """Test the bracket signal"""
    brackets = [i[1] for i in brackets_in]
    mean = em.acmean(np.real(brackets))
    brackets = np.real(brackets)
    assert np.all(brackets > 0) or np.all(brackets < 0), brackets
    brackets = variance_reduction(
        np.real(brackets), mean, 1/decrease_var)
    err = em.acstd(brackets)*np.sqrt(len(brackets)-1)
    if 1.5 * err >= mean:
        pass
        #print("Ground state bracket is statistically zero.")
        #print(brackets)
        #raise PrecisionLossError



def continue_neg(eigvals):
    """Restart config loop if we find
    a negative eigenvalue"""
    ret = False
    try:
        if DELETE_NEGATIVE_OPERATORS:
            checkgteq0(eigvals)
    except AssertionError:
        assert None, "no longer supported"
        print("negative eigenvalues found (non-avg)")
        print('eigvals:', eigvals)
        print("allowed operator eliminations:",
              allowedeliminations())
        ret = True
    return ret


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
                line = np.real(line)
                line2 = np.real(line2)
                line3 = np.real(line3)
                line4 = np.real(line4)
                toapp = proc_meff((line, line2, line3, line4),
                                  files=file_tup, time_arr=timeij)
                reuse[str(line)+"@"+str(line2)+"@"+str(line3)] = toapp
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
                    assert None, "no longer supported"
                    print("***ERROR***")
                    print("GEVP has negative eigenvalues.")
                    sys.exit(1)

    def getblock_plus(file_tup, reuse, timeij=None, delta_t=None):
        """get the block"""
        if timeij is not None and delta_t is not None:
            assert timeij-delta_t >= 0, str((timeij, delta_t))
        if reuse:
            pass
        try:
            retblk = getblock_gevp(file_tup, delta_t, timeij)
        except (ImaginaryEigenvalue, NegativeEigenvalue, NegativeEnergy,
                PrecisionLossError, EigenvalueSignInconsistency):
            glin.reset_sortevals()
            raise XmaxError(problemx=timeij)
        test_imagblk(retblk)
        return retblk
else:

    def getblock_plus(file_tup, reuse, timeij=None, delta_t=None):
        """get the block"""
        if delta_t is None:
            pass
        ret = getblock_simple(file_tup, reuse, timeij)
        if timeij == 7.0 and False:
            ret = np.asarray(ret)
            print("error in eff mass at t=7:", sterr(ret))
            print("error in eff mass at t=7 up to 70:",
                  sterr(ret[16:70+16]))
            print("error in eff mass at t=7 70 to 140:",
                  sterr(ret[70+16:140+16]))
            sys.exit(0)
        return ret


def getblock(file_tup, reuse, timeij=None, delta_t=None):
    """get the block and subtract any bad configs"""
    if timeij is not None and delta_t is not None:
        assert timeij-delta_t >= 0, str((timeij, delta_t))
    retblk = np.array(getblock_plus(file_tup, reuse, timeij,
                                    delta_t=delta_t))
    retblk = binconf(retblk)
    return retblk
