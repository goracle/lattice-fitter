"""Get the data block."""
import sys
from collections import deque
from linecache import getline
from scipy.linalg import eig
import numpy as np

from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS
from latfit.config import GEVP
from latfit.config import START_PARAMS
from latfit.config import ELIM_JKCONF_LIST
from latfit.config import NORMS

#todo, check for neg/imag eigenvals


def get_eigvals(num, file_tup_lhs, file_tup_rhs, overb=False):
    """get the nth generalized eigenvalue from matrices of files
    file_tup_lhs, file_tup_rhs
    optionally, overwrite the rhs matrix we get if we don't need it anymore.
    """
    dimops = len(file_tup_lhs)
    c_lhs = np.zeros((dimops, dimops), dtype=float)
    c_rhs = np.zeros((dimops, dimops), dtype=float)
    for opa in range(dimops):
        for opb in range(dimops):
            c_lhs[opa][opb] = proc_line(
                getline(file_tup_lhs[opa][opb], num+1),
                file_tup_lhs[opa][opb])*NORMS[opa][opb]
            c_rhs[opa][opb] = proc_line(
                getline(file_tup_rhs[opa][opb], num+1),
                file_tup_rhs[opa][opb])*NORMS[opa][opb]
    eigvals, _ = eig(c_lhs, c_rhs, overwrite_a=True,
                     overwrite_b=overb, check_finite=False)
    return eigvals


if EFF_MASS:
    def getblock_gevp(file_tup):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        files_tup[2] is the t+1 lhs
        files_tup[3] is the t+2 lhs
        C(t)v = Eigval*C(t_0)v
        """
        retblk = deque()
        dimops = len(file_tup[0])
        num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        for num in range(num_configs):
            eigvals = get_eigvals(num, file_tup[0], file_tup[1])
            eigvals2 = get_eigvals(num, file_tup[2], file_tup[1])
            eigvals3 = get_eigvals(num, file_tup[3], file_tup[1], overb=True)
            retblk.append(np.array([proc_meff(
                eigvals[op], eigvals2[op], eigvals3[op])
                                    for op in range(dimops)]))
        return retblk

else:
    def getblock_gevp(file_tup):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        C(t)v = Eigval*C(t_0)v
        """
        retblk = deque()
        num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        for num in range(num_configs):
            eigvals = get_eigvals(num, file_tup[0], file_tup[1])
            retblk.append(eigvals)
        return retblk

if EFF_MASS:
    def getblock_simple(file_tup, reuse):
        """Given file,
        get block of effective masses, store in reuse[ij_str]
        """
        retblk = deque()
        for line, line2, line3 in zip(
                open(file_tup[0], 'r'),
                open(file_tup[1], 'r'),
                open(file_tup[2], 'r')):
            if not line+line2+line3 in reuse:
                reuse[str(line)+" "+str(line2)+" "+str(line3)] = proc_meff(
                    line, line2, line3, file_tup)
            if reuse[line+line2+line3] == 0:
                reuse[line+line2+line3] = START_PARAMS[1]
            retblk.append(reuse[line+line2+line3])
        return retblk

else:
    def getblock_simple(ijfile, reuse):
        """Given file,
        get block, store in reuse[ij_str]
        """
        if reuse:
            pass
        retblk = deque()
        for line in open(ijfile):
            retblk.append(proc_line(line, ijfile))
        return retblk


###system stuff, do the subtraction of bad configs as well

if GEVP:
    def test_imagblk(blk):
        """test block for imaginary eigenvalues in gevp"""
        for test1 in blk:
            for test in test1:
                if test.imag != 0:
                    print("***ERROR***")
                    print("GEVP has negative eigenvalues.")
                    sys.exit(1)
    def getblock_plus(file_tup, reuse):
        """get the block"""
        if reuse:
            pass
        retblk = getblock_gevp(file_tup)
        test_imagblk(retblk)
        return retblk
else:
    def getblock_plus(file_tup, reuse):
        """get the block"""
        return getblock_simple(file_tup, reuse)

def getblock(file_tup, reuse):
    """get the block and subtract any bad configs"""
    retblk = np.array(getblock_plus(file_tup, reuse))
    if ELIM_JKCONF_LIST:
        return elim_jkconfigs(retblk)
