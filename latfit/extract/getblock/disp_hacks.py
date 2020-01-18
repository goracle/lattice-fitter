"""Dispersive energies are needed in processing the data from getblock
These are hacks to deal with dynamic array structure introduced by
jackknifing and binning.
"""
import numpy as np
from latfit.config import MULT, GEVP, GEVP_DIRS, DISP_ENERGIES, OPERATOR_NORMS
from latfit.config import LOGFORM, GEVP_DERIV
import latfit.analysis.misc as misc
from latfit.mathfun.proc_meff import proc_meff
import latfit.extract.getblock.gevp_linalg as glin

NORMS = [[(1+0j) for _ in range(len(OPERATOR_NORMS))]
         for _ in range(len(OPERATOR_NORMS))]

for IDX, NORM1 in enumerate(OPERATOR_NORMS):
    for IDX2, NORM2 in enumerate(OPERATOR_NORMS):
        NORMS[IDX][IDX2] = NORM1*np.conj(NORM2)
print("NORMS_ij =", NORMS)


def mod_disp(dispe):
    """Modify imported dispersion energies"""
    if len(dispe) != MULT and GEVP:
        for i, dur in enumerate(GEVP_DIRS):
            if 'rho' in dur[i] or 'sigma' in dur[i]:
                dispe = list(dispe)
                if hasattr(dispe[0], '__iter__')\
                and np.asarray(dispe[0]).shape:
                    assert i, "rho/sigma should not be first operator."
                    dispe.insert(
                        i, np.zeros(len(dispe[0]), dtype=np.complex))
                else:
                    dispe.insert(i, 0)

    if dispe:
        if hasattr(dispe[0], '__iter__')\
        and np.asarray(dispe[0]).shape:
            dispe = np.swapaxes(dispe, 0, 1)
    return dispe

DISP_ENERGIES = mod_disp(DISP_ENERGIES)

def disp():
    """Return the dispersion relation energies = Sqrt(m^2+p^2)"""
    assert not disp.binned or disp.origl != len(DISP_ENERGIES),\
        ("remove this check after a runtime encounter. ",
         DISP_ENERGIES, disp.origl)
    if not disp.binned and disp.origl == len(DISP_ENERGIES):
        #disp.energies = latfit.config.update_disp()
        #disp.energies = mod_disp(disp.energies)
        disp.energies = binhalf_e(disp.energies)
        disp.binned = True
    return disp.energies
disp.energies = np.asarray(DISP_ENERGIES)
disp.binned = False
disp.origl = len(DISP_ENERGIES)

def binhalf_e(ear):
    """Update energies for binning and elimination
    horrible global/local hack
    """
    new_disp = []
    ldisp = len(np.asarray(misc.massfunc()).shape)
    for i in range(MULT):
        if MULT > 1 and ldisp > 1:
            new_disp.append(misc.select_subset(ear[:, i]))
        elif ldisp > 1:
            ear = misc.select_subset(ear)
    if MULT > 1 and ldisp > 1:
        ear = np.swapaxes(new_disp, 0, 1)
    return ear

def callprocmeff(eigvals, timeij, delta_t, sort=False):
    """Call processing function for effective mass"""
    dimops = len(eigvals[0])
    if len(eigvals) == 2:
        eigvals = list(eigvals)
        eigvals.append(np.zeros(dimops)*np.nan)
        eigvals.append(np.zeros(dimops)*np.nan)
        assert len(eigvals) == 4
    if sort:
        for i in range(4):
            eigvals[i] = glin.sortevals(eigvals[i])
    toproc = 1/eigvals[0] if not LOGFORM else eigvals[0]/delta_t
    if GEVP_DERIV:
        energies = np.array([proc_meff((eigvals[0][op], eigvals[1][op],
                                        eigvals[1][op], eigvals[2][op]),
                                       index=op, time_arr=timeij)
                             for op in range(dimops)])
    else:
        energies = np.array([proc_meff((toproc[op], 1, eigvals[1][op],
                                        eigvals[2][op]), index=op,
                                       time_arr=timeij)
                             for op in range(dimops)])
    return energies
