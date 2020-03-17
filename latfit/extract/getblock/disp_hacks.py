"""Dispersive energies are needed in processing the data from getblock
These are hacks to deal with dynamic array structure introduced by
jackknifing and binning.
"""
import numpy as np
from latfit.config import MULT, GEVP, GEVP_DIRS, DISP_ENERGIES, OPERATOR_NORMS
from latfit.config import LOGFORM, GEVP_DERIV, VERBOSE
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.config import DELTA_E2_AROUND_THE_WORLD
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
    assert disp.binned or disp.origl == len(DISP_ENERGIES),\
        (DISP_ENERGIES.shape, disp.origl)
    if not disp.binned:
        #disp.energies = latfit.config.update_disp()
        #disp.energies = mod_disp(disp.energies)
        disp.energies = binhalf_e(disp.energies)
        disp.binned = True
        if DELTA_E_AROUND_THE_WORLD is not None:
            delt = np.asarray(DELTA_E_AROUND_THE_WORLD)
            delt = match_delt(delt)
            assert np.asarray(
                disp.energies).shape == delt.shape, (disp.energies.shape)
            disp.energies = np.asarray(disp.energies) - delt
        if DELTA_E2_AROUND_THE_WORLD is not None:
            delt = np.asarray(DELTA_E2_AROUND_THE_WORLD)
            delt = match_delt(delt)
            assert np.asarray(
                disp.energies).shape == delt.shape, (disp.energies.shape)
            disp.energies = disp.energies - delt
    return disp.energies
disp.energies = np.asarray(DISP_ENERGIES)
disp.binned = False
disp.origl = len(DISP_ENERGIES)

def match_delt(delt):
    """Duplicate delta energy in matrix subtraction
    to give it the same array structure as disp energies.
    """
    ret = []
    sigveck = True if not np.all(DISP_ENERGIES[0]) else False
    mult = int(MULT)
    if sigveck:
        mult -= 1
    mult = range(mult)
    for i in delt:
        toapp = []
        for _ in mult:
            toapp.append(i)
        toapp = mod_disp(toapp)
        ret.append(toapp)
    ret = np.asarray(ret)
    return ret

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

def identity_sort_meff(tosort, timeij, dimops):
    """Perform a check:
    Sort the eigenvalues; throw an error if the sort
    is not the identity"""
    ret = list(tosort)
    if dimops == 1:
        begin = tosort[0]
    else:
        begin = list(tosort)
    if len(tosort) > 1:
        ret = glin.sortevals(tosort)[0]
        if dimops == 1:
            # assert map is identity
            assert np.all(begin == ret[0]), (ret, begin)
        else:
            try:
                assert np.all(begin == ret) or np.all(np.isnan(ret)),\
                    (ret, begin)
            except AssertionError:
                if VERBOSE:
                    print("non-identity map found")
                    print(ret)
                    print(begin)
                raise XminError(problemx=timeij)
    return ret

def inv_arr(arr):
    """1/array without throwing error"""
    ret = []
    for i in arr:
        if not i:
            ret.append(np.inf)
        else:
            ret.append(1/i)
    return ret

def callprocmeff(eigvals, timeij, delta_t, id_sort=False, dimops=None):
    """Call processing function for effective mass"""
    dimops = len(eigvals[0]) if dimops is None else dimops
    assert delta_t, delta_t
    if len(eigvals) == 2:
        eigvals = list(eigvals)
        eigvals.append(np.zeros(dimops)*np.nan)
        eigvals.append(np.zeros(dimops)*np.nan)
        assert len(eigvals) == 4
    if id_sort:
        for i in range(4):
            tosort = eigvals[i]
            eigvals[i] = identity_sort_meff(
                tosort, timeij, dimops)
    todiv = eigvals[0]
    try:
        toproc = inv_arr(todiv[:dimops]) if not LOGFORM else todiv/delta_t
    except FloatingPointError:
        print(dimops)
        print(eigvals[0])
        print(delta_t)
        raise
    except TypeError:
        print(eigvals)
        print(dimops)
        print(delta_t)
        raise
    if dimops > 1:
        assert not np.any(np.isnan(toproc)[:dimops]), (toproc, eigvals, delta_t)
    else:
        assert not np.isnan(toproc[0]), (toproc, eigvals, delta_t)
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
    if len(eigvals[0]) != dimops:
        energies = list(energies)
        for _ in range(len(eigvals[0])-dimops):
            energies.append(np.nan)
    return np.asarray(energies)
