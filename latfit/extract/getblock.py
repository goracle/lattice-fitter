"""Get the data block."""
import numpy as np

from scipy.linalg import eig
from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.extract.proc_line import proc_line

from latfit.config import UNCORR
from latfit.config import EFF_MASS
from latfit.config import elim_jkconf_list

#todo, check for neg/imag eigenvals
if EFF_MASS:
    def getblock_gevp(file_tup, reuse, ij_str):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        """
        dimops=len(file_tup[0])
        #C(t)v=Eigval*C(t_0)v
        C_LHS=np.zeros((dimops,dimops),dtype=float)
        C_RHS=np.zeros((dimops,dimops),dtype=float)
        C2_LHS=np.zeros((dimops,dimops),dtype=float)
        C3_LHS=np.zeros((dimops,dimops),dtype=float)
        for line_lhs, line_rhs, line2_lhs, line3_lhs in zip(
                open(file_tup[0]),
                open(files_tup[1]),
                open(files_tup[2]),
                open(files_tup[3])):
            for opa in range(dimops):
                for opb in range(dimops):
                    C_LHS[opa][opb]=proc_line(line_lhs,
                                           file_tup[0][opa][opb])
                    C_RHS[opa][opb]=proc_line(line_rhs,
                                           file_tup[1][opa][opb])
                    C2_LHS[opa][opb]=proc_line(line_lhs2,file_tup[2][opa][opb])
                    C3_LHS[opa][opb]=proc_line(line_lhs3,file_tup[3][opa][opb])
                eigvals,eigvecs=eig(C_LHS,C_RHS,
                                    overwrite_a=True,check_finite=False)
                eigvals2,eigvecs2=eig(C2_LHS,C_RHS,
                                      overwrite_a=True,check_finite=False)
                eigvals3,eigvecs3=eig(C3_LHS,C_RHS,
                    overwrite_a=True,overwrite_b=True,check_finite=False)
                reuse[ij_str].append(np.array([proc_meff(
                    eigvals[op].real,eigvals2[op].real,
                    eigvals3[op].real) for op in range(dimops)]))
        if elim_jkconf_list:
            reuse[ij_str]=elim_jkconfigs(reuse[ij_str])

else:
    def getblock_gevp(file_tup, reuse, ij_str):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        files_tup[0] is the LHS of the GEVP, files_tup[1] is the RHS
        """
        dimops=len(file_tup[0])
        C_LHS=np.zeros((dimops,dimops),dtype=float)
        C_RHS=np.zeros((dimops,dimops),dtype=float)
        for opa in range(dimops):
            for opb in range(dimops):
                for line_lhs, line_rhs in zip(open(file_tup[0][opa][opb]),
                                              open(files_tup[1][opa][opb])):
                    C_LHS[opa][opb]=proc_line(line_lhs,
                                               file_tup[0][opa][opb])
                    C_RHS[opa][opb]=proc_line(line_rhs,
                                               file_tup[1][opa][opb])
            eigvals,eigvecsI=eig(C_LHS,C_RHS,overwrite_a=True,
                overwrite_b=True,check_finite=False)
            reuse[ij_str].append(eigvals)
        if elim_jkconf_list:
            reuse[ij_str]=elim_jkconfigs(reuse[ij_str])

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
