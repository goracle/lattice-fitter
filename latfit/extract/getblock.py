""""Get the data block."""
import sys
from collections import deque
import scipy
import scipy.linalg
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
from latfit.config import NORMS, GEVP_DEBUG
from latfit.config import BINNUM
from latfit.config import STYPE
from latfit.config import PIONRATIO, ADD_CONST_VEC
from latfit.config import MATRIX_SUBTRACTION

if MATRIX_SUBTRACTION and GEVP:
    ADD_CONST_VEC = [0 for i in ADD_CONST_VEC]

XMAX = 999

if PIONRATIO:
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


def get_eigvals(num, file_tup_lhs, file_tup_rhs, overb=False, print_evecs=False):
    """get the nth generalized eigenvalue from matrices of files
    file_tup_lhs, file_tup_rhs
    optionally, overwrite the rhs matrix we get if we don't need it anymore.
    """
    print_evecs = False if not GEVP_DEBUG else print_evecs
    if not get_eigvals.sent and print_evecs:
        print("First solve, so printing norms which are multiplied onto GEVP entries.")
        print("e.g. C(t)_ij -> Norms[i][j]*C(t)_ij")
        print("Norms=", NORMS)
        get_eigvals.sent = True
    dimops = len(file_tup_lhs)
    c_lhs = np.zeros((dimops, dimops), dtype=float)
    c_rhs = np.zeros((dimops, dimops), dtype=float)
    for opa in range(dimops):
        for opb in range(dimops):
            c_lhs[opa][opb] = proc_line(
                getline_loc(file_tup_lhs[opa][opb], num+1),
                file_tup_lhs[opa][opb])*NORMS[opa][opb]
            c_rhs[opa][opb] = proc_line(
                getline_loc(file_tup_rhs[opa][opb], num+1),
                file_tup_rhs[opa][opb])*NORMS[opa][opb]
    eigvals, evecs = scipy.linalg.eig(c_lhs, c_rhs, overwrite_a=True,
                                      overwrite_b=overb, check_finite=False)
    eigfin = np.zeros((len(eigvals)), dtype=np.float)
    for i, j in enumerate(eigvals):
        if j.imag == 0:
            eigfin[i] = eigvals[i].real
        else:
            print("Eigenvalue=", j)
            print("Manually enforcing Hermiticity of GEVP.")
            if not np.allclose(c_lhs[opa][opb], np.conj(c_lhs[opb][opa]), rtol=1e-8):
                c_lhs = (c_lhs+np.conj(c_lhs))/2
            if not np.allclose(c_rhs[opa][opb], np.conj(c_rhs[opb][opa]), rtol=1e-8):
                c_rhs = (c_rhs+np.conj(c_rhs))/2
            eigvals, evecs = scipy.linalg.eig(c_lhs, c_rhs, overwrite_a=True,
                                              overwrite_b=overb, check_finite=False)
    if print_evecs:
        print("start solve")
        print("lhs=", c_lhs)
        print("rhs=", c_rhs)
        for i, j in enumerate(eigvals):
            print("eigval #", i, "=", j, "evec #", i, "=", evecs[:, i])
        print("end solve")
    for i, j in enumerate(eigvals):
        if j.imag == 0:
            eigfin[i] = eigvals[i].real
        else:
            print("Eigenvalue=", j)
            raise ImaginaryEigenvalue
    return eigfin
get_eigvals.sent = False


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



if EFF_MASS:
    def getblock_gevp(file_tup, timeij=None):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        files_tup[2] is the t+1 lhs
        files_tup[3] is the t+2 lhs
        C(t)v = Eigval*C(t_0)v
        """
        retblk = deque()
        dimops = len(file_tup[0])
        if STYPE == 'ascii':
            num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        elif STYPE == 'hdf5':
            num_configs = len(file_tup[0][0][0])
        if GEVP_DEBUG:
            print("Getting block for time slice=", timeij)
        check_variance = []
        for num in range(num_configs):
            if GEVP_DEBUG:
                print("config #=", num)
            try:
                eigvals = sorted(get_eigvals(num, file_tup[0], file_tup[1],
                                             print_evecs=True), reverse=True)
                eigvals2 = sorted(get_eigvals(num, file_tup[2],
                                              file_tup[1]), reverse=True)
                eigvals3 = sorted(get_eigvals(num, file_tup[3],
                                              file_tup[1]), reverse=True)
                eigvals4 = sorted(get_eigvals(num, file_tup[4], file_tup[1],
                                              overb=True), reverse=True)

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
                print('config_num:', num, 'time:', timeij)
                raise XmaxError(problemx=timeij)
            check_variance.append(eigvals)
            retblk.append(np.array([proc_meff(
                (eigvals[op], eigvals2[op], eigvals3[op],
                 eigvals4[op]), index=op, time_arr=timeij)
                                    for op in range(dimops)]))
        if GEVP_DEBUG:
            print("time, avg evals, variance of evals:",
                  timeij, jack_mean_err(np.array(check_variance)))
            if timeij == 10:
                sys.exit(0)
        return retblk

else:
    def getblock_gevp(file_tup, timeij=None):
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

    def getblock_plus(file_tup, reuse, timeij=None):
        """get the block"""
        if reuse:
            pass
        retblk = getblock_gevp(file_tup, timeij)
        test_imagblk(retblk)
        return retblk
else:

    def getblock_plus(file_tup, reuse, timeij=None):
        """get the block"""
        return getblock_simple(file_tup, reuse, timeij)


def getblock(file_tup, reuse, timeij=None):
    """get the block and subtract any bad configs"""
    retblk = np.array(getblock_plus(file_tup, reuse, timeij))
    if ELIM_JKCONF_LIST:
        retblk = elim_jkconfigs(retblk)
    if BINNUM != 1:
        retblk = binconf(retblk)
    return retblk
