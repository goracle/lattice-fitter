"""Add a quantity which is zero on average to the energies,
but is correlated to reduce the statistical error.
Technique is called 'pion ratio', and is due primarily
to X. Feng, C. Lehner.

Also, use non-interacting pi-pi correlators to subtract non-interacting
around the world terms
"""
import sys
import re
import math
import copy

import numpy as np
from scipy.stats import pearsonr

from latfit.extract.getblock.gevp_linalg import sterr, checkgteq0
from latfit.extract.getblock.gevp_linalg import make_avg_zero
from latfit.extract.getblock.gevp_linalg import variance_reduction

from latfit.extract.proc_folder import proc_folder
from latfit.utilities import exactmean as em
from latfit.analysis.errorcodes import XminError, XmaxError
from latfit.config import PIONRATIO, GEVP_DIRS, GEVP_DERIV
from latfit.config import DECREASE_VAR, NOATWSUB, MATRIX_SUBTRACTION
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.config import DELTA_E2_AROUND_THE_WORLD, VERBOSE
from latfit.config import ISOSPIN
from latfit.config import GEVP_DIRS_PLUS_ONE, FULLDIM
from latfit.config import PR_GROUND_ONLY, VERBOSE
import latfit.config
import latfit.extract.getblock.disp_hacks as gdisp
if PIONRATIO:
    from latfit.config import MINIMIZE_STAT_ERROR_PR
else:
    MINIMIZE_STAT_ERROR_PR = False

### Pion ratio
def aroundworld_energies():
    """Add around the delta world energies"""
    assert None, "this is not needed."
    if MATRIX_SUBTRACTION and not NOATWSUB:
        exp = list(DELTA_E_AROUND_THE_WORLD)
        exp2 = list(DELTA_E2_AROUND_THE_WORLD)
        ret = exp2-exp if exp2 is not None else exp
    else:
        ret = 0
    return ret

def aroundtheworld_pionratio(diag_name, timeij):
    """Do around the world subtraction for the 1x1 pion ratio GEVP"""
    name = diag_name
    ret = proc_folder(name, timeij)
    if MATRIX_SUBTRACTION and not NOATWSUB:
        exp = list(DELTA_E_AROUND_THE_WORLD)
        exp2 = list(DELTA_E2_AROUND_THE_WORLD)
        if exp is not None:
            sub = proc_folder(name, timeij-latfit.config.DELTA_T_MATRIX_SUBTRACTION)
            if hasattr(exp, '__iter__') and np.asarray(exp).shape:
                for i, _ in enumerate(exp):
                    ret[i] *= math.exp(exp[i]*timeij)
                    sub[i] *= math.exp(exp[i]*(
                        timeij-latfit.config.DELTA_T_MATRIX_SUBTRACTION))
            else:
                ret *= math.exp(exp*timeij)
                sub *= math.exp(exp*(timeij-latfit.config.DELTA_T_MATRIX_SUBTRACTION))
            ret -= sub
        if exp2 is not None:
            ret *= math.exp(exp2*timeij)
            time2 = timeij-latfit.config.DELTA_T2_MATRIX_SUBTRACTION
            sub2 = proc_folder(name, time2)
            time3 = timeij-\
                latfit.config.DELTA_T2_MATRIX_SUBTRACTION-latfit.config.DELTA_T_MATRIX_SUBTRACTION
            sub3 = proc_folder(name, time3)
            ret -= sub2*math.exp((
                exp+exp2)*time2)-sub3*math.exp((exp+exp2)*time3)
    return ret

def evals_pionratio(timeij, delta_t, switch=False):
    """Get the non-interacting eigenvalues"""
    ret = []
    skip_next = False
    for i, diag in enumerate(GEVP_DIRS_PLUS_ONE):
        zeroit = False
        if skip_next:
            skip_next = False
            continue
        if 'rho' in diag[i] or 'sigma' in diag[i]:
            diag = GEVP_DIRS_PLUS_ONE[i+1] if not FULLDIM else GEVP_DIRS[i-1]
            zeroit = bool(FULLDIM)
            name = re.sub(r'.jkdat', r'_pisq.jkdat',
                          diag[i+(1 if not FULLDIM else -1)])
            skip_next = not FULLDIM
        else:
            skip_next = False
            name = re.sub(r'.jkdat', r'_pisq.jkdat', diag[i])
        assert 'rho' not in name
        assert 'sigma' not in name
        app = aroundtheworld_pionratio(name, timeij)
        app = np.zeros(len(app), dtype=np.complex) if zeroit else app
        assert not any(np.isnan(app))
        ret.append(app)
    ret = np.swapaxes(ret, 0, 1)
    ret = np.real(ret)
    ret = variance_reduction(ret, em.acmean(ret, axis=0))
    if not MATRIX_SUBTRACTION and not NOATWSUB:
        ret = atwsub(ret, timeij, delta_t, reverseatw=switch)
    return np.asarray(ret)

def energies_pionratio(timeij, delta_t):
    """Find non-interacting energies"""
    key = getkey(timeij, delta_t)
    assert key not in energies_pionratio.store
    lhs = evals_pionratio(timeij, delta_t)
    lhs_p1 = np.zeros(lhs.shape)*np.nan
    try:
        lhs_p1 = evals_pionratio(timeij+1, delta_t)
    except XmaxError:
        if GEVP_DERIV:
            if VERBOSE:
                print("raising from pion ratio")
            raise
    rhs = evals_pionratio(timeij-delta_t, delta_t, switch=True)
    avglhs = np.asarray(em.acmean(lhs, axis=0))
    avglhs_p1 = np.asarray(em.acmean(lhs_p1, axis=0))
    avgrhs = np.asarray(em.acmean(rhs, axis=0))
    exclsave = [list(i) for i in latfit.config.FIT_EXCL]
    try:
        pass
        #assert all(abs(rhs[0]/rhs[0]) > 1)
    except AssertionError:
        print("(abs) lhs not greater than rhs in pion ratio")
        print("example config value of lhs, rhs:")
        print(lhs[10], rhs[10])
        sys.exit(1)
    avg_energies = gdisp.callprocmeff([
        (avglhs/avgrhs), (avglhs_p1/avgrhs)], timeij, delta_t, sort=True)
    energies_pionratio.store[key] = proc_meff_pionratio(
        lhs, lhs_p1, rhs, avg_energies, (timeij, delta_t))
    # so we don't lose operators due to rho/sigma nan's
    latfit.config.FIT_EXCL = exclsave
    return energies_pionratio.store[key]
energies_pionratio.store = {}

def proc_meff_pionratio(lhs, lhs_p1, rhs, avg_energies, timedata):
    """wrapper for callprocmeff
    (non avg version)
    """
    timeij, delta_t = timedata
    np.seterr(divide='ignore', invalid='ignore')
    arg1 = np.asarray(lhs/rhs)
    arg2 = np.asarray(lhs_p1/rhs)
    energies = []
    # config loop
    for i in range(len(lhs)):
        checkgteq0(arg1[i])
        checkgteq0(arg2[i])
        energies.append(gdisp.callprocmeff([arg1[i], arg2[i]],
                                           timeij, delta_t, sort=True))
    np.seterr(divide='warn', invalid='warn')
    energies = variance_reduction(energies, avg_energies, 1/DECREASE_VAR)
    assert all(energies[0] != energies[1]), "same energy found."
    energies = np.asarray(energies)
    for i, dim in enumerate(energies):
        for j, en1 in enumerate(dim):
            if np.isnan(en1):
                energies[i][j] = np.nan
    np.seterr(divide='raise', invalid='raise')
    return energies

def finsum_dev(i, j, addzero, eint):
    """Test new final sum of interacting energy and additive zero
    also, calculate the standard error
    """
    # proposal for final sum (to test)
    finsum = addzero[:, j]+eint[i]
    assert not np.any(np.isnan(finsum)), str(
        addzero[0, j])+" "+str(eint[i][0])+" "+str(i)+" "+str(j)

    # std error in finsum
    dev = em.acstd(finsum)*np.sqrt(len(finsum)-1)
    return dev

def sort_addzero(addzero, enint, timeij, sortbydist=True):
    """Introducing rho/sigma operator introduces ambiguity
    in energy sort:  where to sort the extra 0 entry
    in the non-interacting energies introduced by these operators?
    Well, we are free to choose (since we are adding 0),
    so sort by the min distance
    between interacting energies and dispersion relation energies"""
    assert sortbydist or not MINIMIZE_STAT_ERROR_PR
    if MINIMIZE_STAT_ERROR_PR:
        assert None, "needs consistency between all lattice spacing"
    mapi = []
    ret = np.zeros(addzero.shape, np.float)
    dispf = np.asarray(gdisp.disp())
    if not isinstance(gdisp.disp()[0], float):
        dispf = em.acmean(gdisp.disp(), axis=0)
        # eint = np.swapaxes(enint, 0, 1)
    #else:
    #    eint = em.acmean(enint, axis=0)
    assert addzero.shape[1] == len(dispf),\
        "array mismatch:"+str(dispf)+" "+str(addzero[0])
    for i, mean in enumerate(em.acmean(enint, axis=0)):

        if np.isnan(mean):
            continue

        # reset comparisons
        mindist = np.inf
        #mindev = np.inf
        mindx = np.nan
        #mindx2 = np.nan

        for j, edisp in enumerate(dispf):

            # calculate metrics
            dist = np.abs(mean-edisp)
            # rho/sigma index is 1
            if i == 1 and not np.any(edisp):
                assert FULLDIM,\
                    "rho/sig disp energy is 0 (skipped),"+\
                    " but we have leftover disp energy"+\
                    " which is not being used"
                dist = 0
            # dev = finsum_dev(i, j, addzero, eint)

            # which additive zero
            # has a corresponding disp energy
            # closest to the energy level in question?
            mindist = min(dist, mindist)

            # same thing but min std error
            # mindev = min(dev, mindev)

            # store result
            if mindist == dist and sortbydist:
                mindx = j
            #elif mindev == dev:
                #mindx = j
        # check
        if not np.isnan(mindx):
            # print(mindist, mindx, i)
            mapi.append((mindx, i))
    check_map(mapi, timeij)
    for i, mapel in enumerate(mapi):
        # fromj, toi = mapel
        #assert toi != 1, \
        #    "index bug, rho/sigma should not get a correlated 0"
        # print("add zero mean (", j, "):", em.acmean(addzero[:, fromj]))
        ret[:, mapel[1]] = copy.deepcopy(addzero[:, mapel[0]])
    if not mapi:
        assert None, "bug"
        ret = addzero
    # we can always add/subtract the average,
    # which should be 0 in the large stat. limit.
    # assuming pion ratio method doesn't correct discr. error.
    for i in range(addzero.shape[1]):
        # check to see if this raises the energy
        # assuming excited states are the main sys. err,
        # additive zero should always be 0 or < 0
        if em.acmean(ret[:, i]) > 0 and MINIMIZE_STAT_ERROR_PR:
            print("correcting index", i, "of add zero avg")
            assert None, "hold"
            ret[:, i] = make_avg_zero(ret[:, i])
    return ret

def check_map(mapi, timeij):
    """check to make sure the map doesn't change
    from the trivial identity map
    otherwise, our dispersive energy mapping
    to interacting energy is unstable
    and can't be used when doing a continuum extrap."""
    for item in mapi:
        i, j = item
        if i != j:
            if VERBOSE:
                print(str(mapi)+" "+str(i)+" "+str(j))
            raise XminError(problemx=timeij)

def getkey(timeij, delta_t):
    """Get cache key"""
    dt1 = latfit.config.DELTA_T_MATRIX_SUBTRACTION
    dt2 = latfit.config.DELTA_T2_MATRIX_SUBTRACTION
    ret = (timeij, delta_t, dt1, dt2)
    return ret

if PIONRATIO:
    def modenergies(energies_interacting, timeij, delta_t):
        """modify energies for pion ratio
        noise cancellation"""
        key = getkey(timeij, delta_t)
        if key not in energies_pionratio.store:
            energies_noninteracting = energies_pionratio(timeij, delta_t)
        else:
            energies_noninteracting = energies_pionratio.store[key]
        enint = np.asarray(energies_interacting)
        ennon = np.asarray(energies_noninteracting)
        for i, _ in enumerate(ennon[:, 0]):
            pass
            #print(i, en, min(ennon[:, 0]), max(ennon[:, 0]))
        if timeij == 7.0 and False:
            print("original ground non interacting energies")
            show_original_data(ennon[:, 0], enint[:, 0])
        if VERBOSE:
            print(timeij, 'pearson r:', pearsonr(enint[:, 0], ennon[:, 0]))
        if timeij == 7.0 and False:
            sys.exit(0)
        if not np.all(energies_noninteracting.shape == np.asarray(
                gdisp.disp()).shape):
            energies_noninteracting = gdisp.binhalf_e(
                energies_noninteracting)
        # this fails if the binning didn't fix the broadcast incompatibility
        addzero = -1*energies_noninteracting+np.asarray(gdisp.disp())
        for i, energy in enumerate(addzero[0]):
            if np.isnan(energy):
                assert 'rho' in GEVP_DIRS_PLUS_ONE[
                    i][i] or 'sigma' in GEVP_DIRS_PLUS_ONE[i][i]
        addzero = np.nan_to_num(addzero)
        # sanity check; all additive zeros should be small (magic number = 0.1)
        # chosen since usually 0.1 is O(100) MeV
        # sanity check
        try:
            assert np.all(np.asarray(addzero) < 0.1), str(addzero)
        except AssertionError:
            raise XminError(problemx=timeij)
        if PR_GROUND_ONLY:
            for i in range(addzero.shape[1]):
                if i:
                    addzero[:, i] = 0.0
        elif not FULLDIM:
            pass
        else:
            addzero = sort_addzero(addzero, enint, timeij)
        ret = energies_interacting + addzero
        for i, _ in enumerate(addzero):
            try:
                assert not any(np.isnan(addzero[i]))
            except AssertionError:
                print("nan found in pion ratio energies:")
                print(addzero[i])
                sys.exit(1)
        ret = np.asarray(ret)
        print(timeij, "before - after (diff):",
              em.acstd(enint[:, 0])-em.acstd(ret[:, 0]))
        return ret
else:
    def modenergies(energies, *unused):
        """pass"""
        if unused:
            pass
        return energies


def show_original_data(jkarr, jkarr2):
    """Show original (unjackknifed) data for diagnostic purposes"""
    orig = np.zeros(np.asarray(jkarr).shape)
    orig2 = np.zeros(np.asarray(jkarr2).shape)
    jsum = em.acsum(jkarr, axis=0)
    jsum2 = em.acsum(jkarr2, axis=0)
    diffarr = []
    diffarr_total = []
    diffarr_small = []
    diffarr_early = []
    diffarr_late = []
    for i, _ in enumerate(jkarr):
        orig[i] = jkarr[i]
        orig2[i] = jkarr2[i]
        assert len(jkarr) == len(jkarr2), "array mismatch"
        orig[i] = jsum-jkarr[i]*(len(jkarr)-1)
        orig2[i] = jsum2-jkarr2[i]*(len(jkarr2)-1)
        diff = orig2[i]-orig[i]
        diff = jkarr2[i]-jkarr[i]
        diffarr_total.append(diff)
        if i > 70:
            diffarr_late.append(diff)
        else:
            diffarr_early.append(diff)
        if np.abs(diff) > (0.03+jsum-jsum2)/(len(jkarr)-1):
            diffarr.append(diff)
            print(i, orig[i], orig2[i], diff)
        else:
            print(i, orig[i], orig2[i])
            diffarr_small.append(diff)
    print(em.acstd(orig)/np.sqrt(len(orig)))
    print("early diff avg:", em.acmean(diffarr_early),
          "error:", sterr(diffarr_early))
    print("late diff avg:", em.acmean(diffarr_late),
          "error:", sterr(diffarr_late))
    if diffarr_small:
        print("small diff avg:",
              em.acmean(diffarr_small), "error:", sterr(diffarr_small))
    if diffarr:
        print("large diff avg:", em.acmean(diffarr),
              "error:", sterr(diffarr))
    print("early errors (up to 70) (int, nonint):",
          sterr(jkarr2[:70]), sterr(jkarr[:70]))
    print("later errors (70 to 140) (int, nonint):",
          sterr(jkarr2[70:140]), sterr(jkarr[70:140]))
    print("total diff avg:",
          em.acmean(diffarr_total), "error:", sterr(diffarr_total))
    print("running average-total avg, interacting energy, block of 10:")
    for i in running_avg(jkarr2, 10):
        print(i-em.acmean(jkarr2))
    #for i in reversed(diffarr_total):
    #    print(i-em.acmean(diffarr_total))
    return orig

def running_avg(arr, blocksize):
    """Get a running average of the array"""
    arr = np.asarray(arr)
    ret = []
    for i in range(len(arr)-blocksize):
        ret.append(em.acmean(arr[i:blocksize+i]))
    ret = np.array(ret)
    return ret

### around the world subtraction

def atwsub(cmat_arg, timeij, delta_t, reverseatw=False):
    """Subtract the atw vacuum saturation single pion correlators
    (non-interacting around the world term, single pion correlator squared)
    """
    assert timeij >= 0, str(timeij)
    cmat = np.array(copy.deepcopy(np.array(cmat_arg)))
    origshape = cmat.shape
    if not MATRIX_SUBTRACTION and ISOSPIN != 1 and not NOATWSUB:
        suffix = r'_pisq_atwR' if reverseatw else r'_pisq_atw'
        suffix = suffix + '_dt' + str(int(delta_t))+'.jkdat'
        for i, diag in enumerate(GEVP_DIRS):
            zeroit = False
            if 'rho' in diag[i] or 'sigma' in diag[i]:
                # copy of code from pion ratio section
                diag = GEVP_DIRS[i-1]
                zeroit = True
                name = re.sub(r'.jkdat', r'_pisq.jkdat',
                              diag[i-1])
            else:
                #skip_next = False
                name = re.sub(r'.jkdat', suffix, diag[i])
            #print(diag, name)
            assert 'rho' not in name
            assert 'sigma' not in name
            tosub = proc_folder(name, timeij)
            tosub = variance_reduction(tosub,
                                       em.acmean(tosub, axis=0))
            if zeroit:
                tosub = np.real(tosub)*0
            else:
                tosub = np.real(tosub)
            if len(cmat.shape) == 3:
                assert len(cmat) == len(tosub), \
                    "number of configs mismatch:"+str(len(cmat))
                for item in tosub:
                    try:
                        assert (item or zeroit) and not np.isnan(item)
                    except AssertionError:
                        if VERBOSE:
                            print(item)
                            print("nan error in (name, time slice):",
                                  name, timeij)
                        raise XmaxError(problemx=timeij)
                cmat[:, i, i] = cmat[:, i, i]-tosub*np.abs(gdisp.NORMS[i][i])
                assert cmat[:, i, i].shape == tosub.shape
            elif len(cmat.shape) == 2 and len(cmat) != len(cmat[0]):
                for item in tosub:
                    try:
                        assert (item or zeroit) and not np.isnan(item)
                    except AssertionError:
                        if VERBOSE:
                            print(item)
                            print("nan error in (name, time slice):",
                                  name, timeij)
                        raise XmaxError(problemx=timeij)
                cmat[:, i] = cmat[:, i]-tosub*np.abs(gdisp.NORMS[i][i])
            else:
                cmat[i, i] -= em.acmean(tosub,
                                        axis=0)*np.abs(gdisp.NORMS[i][i])
                #if not reverseatw:
                    #print(i, em.acmean(tosub, axis=0)/ cmat[i, i])
                assert not em.acmean(tosub, axis=0).shape
                #print(cmat)
    assert cmat.shape == origshape
    return cmat

def atwsub_cmats(timeinfo, cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs):
    """Wrapper for around the world subtraction"""
    # subtract the non-interacting around the world piece
    delta_t, timeij = timeinfo
    assert timeij-delta_t >= 0, str(timeinfo)
    for i, mean in enumerate(mean_cmats_lhs):
        assert mean_cmats_lhs[i].shape == mean.shape
        try:
            mean_cmats_lhs[i] = atwsub(mean, timeij+i, delta_t)
            cmats_lhs[i] = atwsub(cmats_lhs[i], timeij+i, delta_t)
        except XmaxError:
            if (GEVP_DERIV and i) or not i:
                raise
    mean_crhs = atwsub(mean_crhs, timeij-delta_t,
                       delta_t, reverseatw=True)
    cmat_rhs = atwsub(np.array(copy.deepcopy(np.array(cmat_rhs))),
                      timeij-delta_t, delta_t, reverseatw=True)
    return cmats_lhs, mean_cmats_lhs, cmat_rhs, mean_crhs
