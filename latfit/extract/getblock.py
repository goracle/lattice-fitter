"""Get the data block."""
from linecache import getline
import numpy as np
from scipy.linalg import eig

from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.extract.proc_line import proc_line

from latfit.config import EFF_MASS
from latfit.config import GEVP
from latfit.config import START_PARAMS
from latfit.config import elim_jkconf_list

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
                file_tup_lhs[opa][opb])
            c_rhs[opa][opb] = proc_line(
                getline(file_tup_rhs[opa][opb], num+1),
                file_tup_rhs[opa][opb])
    eigvals, _ = eig(c_lhs, c_rhs,
                     overwrite_a=True, overwrite_b=overb, check_finite=False)
    return eigvals


if EFF_MASS:
    def getblock_gevp(file_tup, reuse, ij_str):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        files_tup[2] is the t+1 lhs
        files_tup[3] is the t+2 lhs
        C(t)v = Eigval*C(t_0)v
        """
        dimops = len(file_tup[0])
        num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        for num in num_configs:
            eigvals = get_eigvals(num, file_tup[0], file_tup[1])
            eigvals2 = get_eigvals(num, file_tup[2], file_tup[1])
            eigvals3 = get_eigvals(num, file_tup[3], file_tup[1], overb=True)
            reuse[ij_str].append(np.array([proc_meff(
                eigvals[op].real,
                eigvals2[op].real,
                eigvals3[op].real) for op in range(dimops)]))
        if elim_jkconf_list:
            reuse[ij_str] = elim_jkconfigs(reuse[ij_str])

else:
    def getblock_gevp(file_tup, reuse, ij_str):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        C(t)v = Eigval*C(t_0)v
        """
        num_configs = sum(1 for _ in open(file_tup[0][0][0]))
        for num in range(num_configs):
            eigvals = get_eigvals(num, file_tup[0], file_tup[1])
            reuse[ij_str].append(eigvals)
        if elim_jkconf_list:
            reuse[ij_str] = elim_jkconfigs(reuse[ij_str])

if EFF_MASS:
    def getblock_simple(file_tup, reuse, ij_str):
        """Given file,
        get block of effective masses, store in reuse[ij_str]
        """
        for line, line2, line3 in zip(
                open(file_tup[0], 'r'),
                open(file_tup[1], 'r'),
                open(file_tup[2], 'r')):
            if not line+line2+line3 in reuse:
                reuse[line+line2+line3] = proc_meff(
                    line, line2, line3, file_tup)
            if reuse[line+line2+line3] == 0:
                reuse[line+line2+line3] = START_PARAMS[1]
            reuse[ij_str].append(reuse[line+line2+line3])
        if elim_jkconf_list:
            reuse[ij_str] = elim_jkconfigs(reuse[ij_str])

else:
    def getblock_simple(ijfile, reuse, ij_str):
        """Given file,
        get block, store in reuse[ij_str]
        """
        for line in open(ijfile):
            reuse[ij_str].append(proc_line(line, ijfile))


###system stuff, do the subtraction of bad configs as well

if GEVP:
    def getblock_plus(file_tup, reuse, ij_str):
        """get the block"""
        getblock_gevp(file_tup, reuse, ij_str)
else:
    def getblock_plus(file_tup, reuse, ij_str):
        """get the block"""
        getblock_simple(file_tup, reuse, ij_str)

def getblock(file_tup, reuse, ij_str):
    """get the block and subtract any bad configs"""
    getblock_plus(file_tup, reuse, ij_str)
    if elim_jkconf_list:
        reuse[ij_str] = elim_jkconfigs(reuse[ij_str])
