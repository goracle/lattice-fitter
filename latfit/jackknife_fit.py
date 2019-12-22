"""Fit under a jackknife"""
import sys
import copy
import os
import mpi4py
from mpi4py import MPI
from collections import namedtuple
import pickle
import numpy as np
from numpy import ma
from numpy import swapaxes as swap
from numpy.linalg import inv, tensorinv
from scipy import stats
from scipy.optimize import fsolve
from accupy import kdot

from latfit.extract.inverse_jk import inverse_jk
from latfit.makemin.mkmin import mkmin
from latfit.mathfun.block_ensemble import delblock, block_ensemble
from latfit.mathfun.block_ensemble import bootstrap_ensemble

from latfit.analysis.superjack import jack_mean_err
from latfit.config import START_PARAMS, RANDOMIZE_ENERGIES
from latfit.config import JACKKNIFE_FIT
from latfit.config import CORRMATRIX, EFF_MASS
from latfit.config import GEVP, FIT_SPACING_CORRECTION
from latfit.config import JACKKNIFE_BLOCK_SIZE, NOLOOP
from latfit.config import UNCORR, UNCORR_OP, SYS_ENERGY_GUESS
from latfit.config import PVALUE_MIN, NOATWSUB, PIONRATIO
from latfit.config import PICKLE, MATRIX_SUBTRACTION
from latfit.config import CALC_PHASE_SHIFT, PION_MASS
from latfit.config import SUPERJACK_CUTOFF, SLOPPYONLY
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.config import DELTA_E2_AROUND_THE_WORLD
from latfit.config import ISOSPIN, VERBOSE
from latfit.config import SKIP_OVERFIT
from latfit.utilities.zeta.zeta import zeta
import latfit.finalout.mkplot
import latfit.config
import latfit.analysis.misc as misc
import latfit.makemin.mkmin as mkmin
from latfit.analysis.errorcodes import NoConvergence, TooManyBadFitsError
from latfit.analysis.errorcodes import BadChisq, BadJackknifeDist
from latfit.analysis.errorcodes import EnergySortError, ZetaError
from latfit.analysis.errorcodes import PrecisionLossError
from latfit.analysis.result_min import ResultMin
from latfit.utilities import exactmean as em
from latfit.utilities.actensordot import actensordot

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

SUPERJACK_CUTOFF = 0 if SLOPPYONLY else SUPERJACK_CUTOFF

CONST_SHIFT = 0

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def torchi():
    """Are we calculating Hotelling's t^2 statistic or a true chi^2?
    return the corresponding string.
    """
    if UNCORR:
        ret = 'chisq/dof='
    else:
        ret = 't^2/dof='
    return ret

def convert_coord_dict(shift, coords):
    """Convert coord dict to array"""
    ret = shift
    allowedtimes = set()
    for i in coords:
        allowedtimes.add(i[0])
    if shift:
        ret = []
        keys = []
        for i in shift:
            if i in allowedtimes:
                keys.append(i)
        keys = sorted(keys)
        nextkey = keys[0]
        for i in keys:
            assert i == nextkey
            nextkey += 1
            ret.append(shift[i])
        ret = np.array(ret)
    return ret

def apply_shift(coords_jack_reuse, coords_jack):
    """apply the bootstrap constant shift to the averages"""
    coords_jack_reuse = copy.deepcopy(np.array(coords_jack_reuse))
    shift = 0 if not CONST_SHIFT else CONST_SHIFT
    shift_arr = convert_coord_dict(shift, coords_jack)
    sh1 = np.asarray(shift_arr).shape
    sh2 = coords_jack_reuse.shape
    check = copy.deepcopy(np.array(coords_jack_reuse))
    if not GEVP or np.all(sh2[1:] == sh1) or not np.array(shift).shape:
        shift = collapse_shift(shift_arr)
        try:
            coords_jack_reuse = coords_jack_reuse + shift
            assert np.allclose(check + shift, coords_jack_reuse,
                               rtol=1e-14)
        except TypeError:
            print(coords_jack_reuse)
            print(shift)
            print(sh1, sh2)
            raise
    else:
        for i, _ in enumerate(coords_jack_reuse):
            time = int(coords_jack_reuse[i][0])
            shiftc = shift if not shift else shift[time]
            coords_jack_reuse[i][1] += shiftc
            assert np.allclose(
                check[i][1] + shiftc,
                coords_jack_reuse[i][1], rtol=1e-14)
    return coords_jack_reuse

if not GEVP:
    def collapse_shift(shift):
        """Collapse the shift structure for non GEVP fits"""
        if hasattr(shift, '__iter__'):
            ret = []
            for i in shift:
                ret.append(i[0])
            ret = np.array(ret)
        else:
            ret = shift
        return ret
else:
    def collapse_shift(shift):
        """Collapse the shift structure for non GEVP fits"""
        return shift


if JACKKNIFE_FIT == 'FROZEN':
    pass

elif JACKKNIFE_FIT in ('DOUBLE', 'SINGLE'):
    @PROFILE
    def jackknife_fit(params, reuse, reuse_blocked, coords, _=None):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        (N.B. substitute t^2 for chi^2 if doing a correlated fit)
        """
        # storage for results
        result_min = ResultMin(params, coords)
        result_min.pvalue.zero(params.num_configs)
        result_min.phase_shift.arr = alloc_phase_shift(params)
        result_min.alloc_sys_arr(params)
        result_min.min_params.arr = np.zeros((params.num_configs,
                                              len(START_PARAMS)))
        result_min.energy.arr = np.zeros((params.num_configs,
                                          len(START_PARAMS)
                                          if not GEVP else params.dimops))
        coords_jack = np.copy(copy.deepcopy(coords))
        result_min.misc.error_bars = alloc_errbar_arr(params, len(coords))

        # p-value fiducial cut:  cut below this pvalue
        # as it is 5 \sigma away from an acceptable chi^2/dof (t^2/dof)
        chisq_fiduc_cut = chisqfiduc(params.num_configs,
                                     result_min.misc.dof)
        # similarly, cut below 5 sigma below chisq/dof = 1
        chisq_fiduc_overfit_cut = overfit_chisq_fiduc(
            params.num_configs, result_min.misc.dof)

        skip_votes = []
        # loop over configs, doing a fit for each one
        # rearrange loop order so we can check goodness of fit
        # on more stable sloppy samples

        config_range = (np.array(range(
            params.num_configs)) + SUPERJACK_CUTOFF) % params.num_configs
        config_range = range(
            params.num_configs) if latfit.config.BOOTSTRAP else config_range

        start_loop = True

        for config_num in config_range:

            # copy the jackknife block into coords_jack
            if config_num < len(reuse) and len(reuse) == len(reuse_blocked):
                assert np.all(reuse[config_num] == reuse_blocked[
                    config_num]),\
                    str(reuse[config_num].shape)+" "+str(reuse_blocked[
                        config_num].shape)
            if not latfit.config.BOOTSTRAP:
                coords_jack = copy_block(params, reuse_blocked[config_num],
                                         coords_jack)
            else:
                # we still need the time data in coords (xmin, xmax, xstep)
                coords_jack = copy_block(params, reuse[0],
                                         coords_jack)

            # get the data for the minimizer, and the error bars
            coords_jack, covinv_jack, result_min.misc.error_bars[
                config_num] = get_doublejk_data(params, coords_jack,
                                                reuse, reuse_blocked,
                                                config_num)
            if start_loop:
                mkmin.SPARAMS = list(np.copy(START_PARAMS))
                mkmin.PARAMS = params
                mkmin.prealloc_chi(covinv_jack, coords_jack)

            # minimize chi^2 (t^2) given the inv. covariance matrix and data
            result_min_jack = mkmin.mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                assert not np.isnan(result_min_jack.status),\
                    str(result_min_jack.status)
                result_min.misc.status = result_min_jack.status
                raise NoConvergence

            if start_loop:
                mkmin.SPARAMS = result_min_jack.x
                start_loop = False

            result_min.min_params.arr[config_num] = result_min_jack.x

            # store results for this fit
            result_min.chisq.arr[config_num] = result_min_jack.fun
            toomanybadfitsp(result_min)

            # store the result
            result_min.systematics.arr[config_num], _ = \
                getsystematic(params, result_min.min_params.arr[config_num])
            result_min.systematics.arr[config_num],\
                params.energyind = getsystematic(
                    params, result_min.min_params.arr[config_num])
            result_min.energy.arr[config_num] = getenergies(
                params, result_min.min_params.arr[config_num])

            if result_min_jack.fun/result_min.misc.dof < 10 and\
               list(result_min.systematics.arr[config_num][:-1]):
                print('systematics:',
                      result_min.systematics.arr[config_num][:-1])

            # we shifted the GEVP energy spectrum down
            # to fix the leading order around the world term
            # so shift it back
            if not latfit.config.BOOTSTRAP:

                result_min.energy.arr[config_num] += correction_en(
                    result_min, config_num)

            # compute phase shift, if necessary
            if CALC_PHASE_SHIFT and not latfit.config.BOOTSTRAP:
                result_min.phase_shift.arr[config_num] = phase_shift_jk(
                    params, result_min.energy.arr[config_num])

            # compute p value for this fit
            result_min.pvalue.arr[config_num] = result_min.funpvalue(
                result_min_jack.fun)

            sys_str = str(result_min.systematics.arr[config_num][-1])\
                if not np.isnan(result_min.systematics.arr[config_num][-1])\
                   else ''

            # print results for this config
            if VERBOSE:
                print("config", config_num, ":",
                      result_min.energy.arr[config_num],
                      sys_str,
                      torchi(), result_min_jack.fun/result_min.misc.dof,
                      "p-value=", result_min.pvalue.arr[config_num],
                      'dof=', result_min.misc.dof, "rank=", MPIRANK)

            assert not np.isnan(result_min.pvalue.arr[
                config_num]), "pvalue is nan"
            # use sloppy configs to check if fit will work
            if config_num in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF] and\
               not latfit.config.BOOTSTRAP:

                # check if chi^2 (t^2) too big, too small
                if result_min_jack.fun > chisq_fiduc_cut or\
                   (SKIP_OVERFIT and result_min_jack.fun < \
                    chisq_fiduc_overfit_cut):
                    skip_votes.append(config_num)

                if config_num == 1+SUPERJACK_CUTOFF:
                    skip_range(params, result_min, skip_votes,
                               result_min_jack, chisq_fiduc_cut)

        # reset the precomputed quantities
        mkmin.dealloc_chi()
        # average results, compute jackknife uncertainties

        # pickle/unpickle the jackknifed arrays
        result_min = pickl(result_min)

        # compute p-value jackknife uncertainty
        result_min.pvalue.val, result_min.pvalue.err = jack_mean_err(
            result_min.pvalue.arr)

        # get the optimal params
        result_min.min_params.val, result_min.min_params.err = jack_mean_err(
            result_min.min_params.arr)

        # compute the mean, error on the params
        result_min.energy.val, result_min.energy.err = jack_mean_err(
            result_min.energy.arr)
        print('param err:', result_min.energy.err,
              'np.std:', np.std(result_min.energy.arr, axis=0))

        # compute the systematics and errors
        if SYS_ENERGY_GUESS is not None:
            result_min.systematics.val, result_min.systematics.err =\
                jack_mean_err(result_min.systematics.arr)

        # average the point by point error bars
        result_min.misc.error_bars = em.acmean(result_min.misc.error_bars,
                                               axis=0)

        # compute phase shift and error in phase shift
        if CALC_PHASE_SHIFT:
            phase_shift_scatter = phase_shift_scatter_len_avg(result_min)

            result_min = unpack_min_data(result_min, *phase_shift_scatter)

        # compute mean, jackknife uncertainty of chi^2 (t^2)
        result_min.chisq.val, result_min.chisq.err = jack_mean_err(
            result_min.chisq.arr)

        print(torchi(), result_min.chisq.val/result_min.misc.dof,
              "std dev:", np.std(result_min.chisq.arr, ddof=1))

        return result_min, result_min.energy.err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

@PROFILE
def toomanybadfitsp(result_min):
    """If there have already been too many fits with large chi^2 (t^2),
    the average chi^2 (t^2) is not going to be good so abort the fit
    (test)
    """
    avg = em.acmean(result_min.chisq.arr)
    pvalue = result_min.funpvalue(avg)
    if pvalue < PVALUE_MIN and not latfit.config.BOOTSTRAP and not NOLOOP:
        raise TooManyBadFitsError(chisq=avg, pvalue=pvalue, uncorr=UNCORR)


@PROFILE
def overfit_chisq_fiduc(num_configs, dof, guess=None):
    """Find the overfit 5 sigma cut
    (see chisqfiduc for the lower cut on the upper bound)
    """
    key = (num_configs, dof)
    t2correction = (num_configs-dof)/(num_configs-1)/dof
    cor = t2correction
    if key in overfit_chisq_fiduc.cache:
        ret = overfit_chisq_fiduc.cache[key]
    else:
        cut = stats.f.cdf(dof*cor, dof, num_configs-dof)
        lbound = 3e-7
        func = lambda tau: ((1-cut*lbound)-(
            stats.f.sf(abs(tau)*cor, dof, num_configs-dof)))**2
        sol = abs(float(fsolve(func, 1e-5 if guess is None else guess)))
        sol2 = dof
        assert abs(func(sol)) < 1e-12, "fsolve failed:"+str(num_configs)+\
            " "+str(dof)
        diff = (sol2-sol)/(num_configs-SUPERJACK_CUTOFF-1)
        assert diff > 0,\
            "bad solution to p-value solve, chi^2(/t^2)/dof solution > 1"
        ret = sol2-diff
        overfit_chisq_fiduc.cache[key] = ret
    return ret
overfit_chisq_fiduc.cache = {}

def skip_range(params, result_min, skip_votes,
               result_min_jack, chisq_fiduc_cut):
    """Raise an error if we should skip this fit range"""
    skiprange = False
    # don't skip the fit range until we confirm
    # on 2nd config
    if len(skip_votes) == 2:
        skiprange = True
    elif len(skip_votes) == 1:
        # approximate the difference as the stddev:
        # sqrt(\sum(x-<x>)**2/(N));
        # mult by sqrt(N-1) to get the variance in chi^2
        # (t^2)
        # if we have one bad fit and another which is within
        # 5 sigma of the bad chi^2 (t^2),
        # skip, else throw an error
        skiprange = abs(result_min_jack.fun-result_min.chisq.arr[
            0+SUPERJACK_CUTOFF]) < 5*np.sqrt(
                2*result_min.misc.dof/(
                    params.num_configs-1))
    if skiprange and not latfit.config.BOOTSTRAP and not NOLOOP:
        raise BadChisq(
            chisq=result_min_jack.fun/result_min.misc.dof,
            dof=result_min.misc.dof, uncorr=UNCORR)
    if skip_votes:
        # the second sample should never have a good fit
        # if the first one has that bad a fit
        print("fiducial cut =", chisq_fiduc_cut)
        print("dof=", result_min.misc.dof)
        print(abs(result_min_jack.fun-result_min.chisq.arr[0]),
              2*5*result_min.misc.dof/np.sqrt(
                  params.num_configs))
        print(result_min_jack.fun, result_min.chisq.arr[0])
        print("Bad jackknife distribution:"+\
                str(result_min.chisq.arr[0]/result_min.misc.dof)+" "+\
                str(result_min.chisq.arr[1]/result_min.misc.dof)+" "+\
                str(result_min.pvalue.arr[0])+" "+\
                str(result_min.pvalue.arr[1])+" ")
        #sys.exit(1)
        if not latfit.config.BOOTSTRAP and not NOLOOP:
            raise BadJackknifeDist(uncorr=UNCORR)

@PROFILE
def chisqfiduc(num_configs, dof):
    """Find the chi^2/dof (t^2/dof) cutoff (acceptance upper bound)
    defined as > 5 sigma away from an acceptable pvalue
    2*dof is the variance in chi^2 (t^2)
    """
    key = (num_configs, dof)
    t2correction = (num_configs-dof)/(num_configs-1)/dof
    cor = t2correction
    if key in chisqfiduc.mem:
        ret = chisqfiduc.mem[key]
    else:
        func = lambda tau: PVALUE_MIN*3e-7-(stats.f.sf(tau*cor, dof,
                                                       num_configs-dof))
        func2 = lambda tau: PVALUE_MIN-(stats.f.sf(tau*cor, dof,
                                                   num_configs-dof))
        # guess about 2 for the max chi^2/dof
        sol = float(fsolve(func, dof))
        sol2 = float(fsolve(func2, dof))
        assert abs(func(sol)) < 1e-8, "fsolve failed."
        assert abs(func2(sol2)) < 1e-8, "fsolve2 failed."
        # known variance of chi^2 is 2*dof,
        # but skewed at low dof (here chosen to be < 10)
        # thus, the notion of a "5 sigma fluctuation" is only defined
        # as dof->inf
        # so we have a factor of 2 to make low dof p-value cut less aggressive
        #ret = sol+5*(2 if dof < 10 else\
        # 1)*np.sqrt(2*dof)/(num_configs-SUPERJACK_CUTOFF)
        diff = (sol-sol2)/(num_configs-SUPERJACK_CUTOFF-1)
        ret = sol2+diff
        #print(ret/dof, sol/dof, num_configs, dof, PVALUE_MIN,
        #      1-stats.chi2.cdf(ret, dof), 1-stats.chi2.cdf(sol, dof))
        chisqfiduc.mem[key] = ret
    return ret
chisqfiduc.mem = {}

@PROFILE
def correction_en(result_min, config_num):
    """Correct the jackknifed E_pipi"""
    delta_e_around_the_world = DELTA_E_AROUND_THE_WORLD
    delta_e2_around_the_world = DELTA_E2_AROUND_THE_WORLD
    if hasattr(delta_e_around_the_world, '__iter__') and\
       np.asarray(delta_e_around_the_world).shape:
        latw = len(delta_e_around_the_world)

        # block the ensemble if needed
        if latw != len(result_min.energy.arr):
            delta_e_around_the_world = block_ensemble(
                len(result_min.energy.arr), delta_e_around_the_world)

            latw = len(delta_e_around_the_world)
            assert latw == 1 or latw == len(result_min.energy.arr),\
                "bug:  array mismatch"
        if delta_e2_around_the_world is not None:
            assert len(delta_e2_around_the_world) == latw

            if latw != len(result_min.energy.arr):
                delta_e2_around_the_world = block_ensemble(
                    len(result_min.energy.arr), delta_e2_around_the_world)

        corre1 = delta_e_around_the_world[config_num] if latw > 1 else\
            delta_e_around_the_world[0]
    else:
        corre1 = delta_e_around_the_world
    if hasattr(delta_e2_around_the_world, '__iter__') and\
       np.asarray(delta_e2_around_the_world).shape:
        corre2 = delta_e2_around_the_world[config_num]
    else:
        corre2 = delta_e2_around_the_world if\
            delta_e2_around_the_world is not None else 0
    if FIT_SPACING_CORRECTION and not PIONRATIO and GEVP:
        corre3 = misc.correct_epipi(result_min.energy.arr[config_num],
                                    config_num=config_num)
    else:
        corre3 = 0
    ret = 0
    if GEVP:
        ret = add_corrections(corre1, corre2, corre3)
    return ret

def add_corrections(corre1, corre2, corre3):
    """Add corrections, zeroing the None's"""
    ret = corre1 if corre1 is not None else 0
    ret = ret+corre2 if corre2 is not None else ret+0
    ret = ret+corre3 if corre3 is not None else ret+0
    return ret


@PROFILE
def unnan_coords(coords):
    """replace nan's with 0 in coords"""
    for i, _ in enumerate(coords):
        coords[i][1] = np.nan_to_num(coords[i][1])
    return coords


@PROFILE
def unpack_min_data(result_min, phase_shift_data, scattering_length_data):
    """Unpack the returned results of phase_shift_scatter_len_avg"""
    result_min.phase_shift.val,\
        result_min.phase_shift.err,\
        result_min.phase_shift.arr = phase_shift_data
    result_min.scattering_length.val,\
        result_min.scattering_length.err,\
        result_min.scattering_length.arr = scattering_length_data
    return result_min

@PROFILE
def getsystematic(params, arr):
    """Get the fit parameters which are not the energies"""
    arr = np.asarray(arr)
    params.energyind = None
    if len(arr) != params.dimops and arr.shape and EFF_MASS:
        temp = list(arr)
        if not (len(START_PARAMS)-1) % 2 and (
                MATRIX_SUBTRACTION or not NOATWSUB or ISOSPIN == 1):
            params.energyind = 2
        elif not (len(START_PARAMS)-1) % 3:
            assert None, "no longer supported"
            assert not MATRIX_SUBTRACTION and NOATWSUB and ISOSPIN != 1
            params.energyind = 3
        elif not (len(START_PARAMS)-1) % 4:
            assert None, "no longer supported"
            assert not MATRIX_SUBTRACTION and NOATWSUB and ISOSPIN != 1
            params.energyind = 4
        del temp[params.energyind-1::params.energyind]
        ret = [item for item in arr if item not in temp]
        ret.append(arr[-1])
        ret = np.array(ret)
    else:
        ret = None
    return ret, params.energyind

@PROFILE
def getenergies(params, arr):
    """Get the energies from an array
    (array may contain other parameters)
    """
    params.energyind = 2 if params.energyind is None else params.energyind
    arr = np.asarray(arr)
    if len(arr) != params.dimops and EFF_MASS:
        ret = arr[0::params.energyind][:-1]
    else:
        ret = arr
    for i, j in zip(sorted(list(ret)), ret):
        if i != j:
            print("mis-sorted energies:", ret)
            if not latfit.config.BOOTSTRAP:
                raise EnergySortError
    return ret

@PROFILE
def phase_shift_scatter_len_avg(result_min):
    """Average the phase shift results, calc scattering length"""
    if not GEVP:
        try:
            result_min.energy.arr = result_min.energy.arr[:, 1]
        except IndexError:
            try:
                result_min.energy.arr = result_min.energy.arr[:, 0]
            except IndexError:
                sys.exit(1)

    # get rid of configs were phase shift calculation failed
    # (good for debug only)
    phase_shift_arr = np.delete(result_min.phase_shift.arr,
                                prune_phase_shift_arr(
                                    result_min.phase_shift.arr), axis=0)

    if np.asarray(phase_shift_arr).shape:

        # calculate scattering length via energy, phase shift
        scattering_length = -1.0*np.tan(
            phase_shift_arr)/np.sqrt(
                (result_min.energy.arr**2/4-PION_MASS**2).astype(complex))

        scattering_length_arr = np.array(scattering_length)

        # calc mean, err on phase shift and scattering length
        phase_shift, phase_shift_err = \
            jack_mean_err(phase_shift_arr)
        scattering_length, scattering_length_err = \
            jack_mean_err(scattering_length)

    else:
        phase_shift = None
        phase_shift_err = None
        scattering_length = None
        scattering_length_err = None
    phase_shift_data = (phase_shift, phase_shift_err, phase_shift_arr)
    assert np.asarray(scattering_length_arr).shape,\
        "scattering length array: "+str(np.asarray(scattering_length_arr))
    scattering_length_data = (scattering_length, scattering_length_err,
                              scattering_length_arr)
    return phase_shift_data, scattering_length_data


@PROFILE
def alloc_phase_shift(params):
    """Get an empty array for Nconfig phase shifts"""
    nphase = 1 if not GEVP else params.dimops
    if GEVP:
        ret = np.zeros((params.num_configs, nphase), dtype=np.complex)
    else:
        ret = np.zeros((params.num_configs), dtype=np.complex)
    return ret

@PROFILE
def pickl(result_min):
    """Pickle or unpickle the results from the jackknife fit loop
    to do: make more general use **kwargs
    """
    if PICKLE == 'pickle':
        pickle.dump(result_min.energy.arr, open(unique_pickle_file("result_min.energy.arr"), "wb"))
        pickle.dump(result_min.phase_shift.arr, open(
            unique_pickle_file("phase_shift.arr"), "wb"))
        pickle.dump(result_min.chisq.arr,
                    open(unique_pickle_file("chisq_arr"), "wb"))
    elif PICKLE == 'unpickle':
        _, rangei = unique_pickle_file("result_min.energy.arr", True)
        _, rangej = unique_pickle_file("phase_shift", True)
        _, rangek = unique_pickle_file("chisq_arr", True)
        for i in range(rangei):
            result_min.energy.arr /= (rangei+1)
            result_min.energy.arr += 1.0/(rangei+1)*pickle.load(open(
                "result_min.energy.arr"+str(i)+".p", "rb"))
        for j in range(rangej):
            result_min.phase_shift.arr /= (rangej+1)
            result_min.phase_shift.arr += 1.0/(
                rangej+1)*pickle.load(open(
                    "phase_shift.arr"+str(j)+".p", "rb"))
        for k in range(rangek):
            result_min.chisq.arr /= (rangek+1)
            result_min.chisq.arr += 1.0/(rangek+1)*pickle.load(open(
                "chisq_arr"+str(k)+".p", "rb"))
    elif PICKLE is None:
        pass
    return result_min

@PROFILE
def unique_pickle_file(filestr, reti=False):
    """Get a unique file string so we don't overwrite when pickling"""
    i = 0
    while os.path.exists(filestr+"%s.p" % i):
        if PICKLE == 'clean':
            os.remove(filestr+"%s.p")
        i += 1
    unique_filestr = filestr+str(i)+".p"
    if reti:
        retval = (unique_filestr, i)
    else:
        retval = unique_filestr
    return retval


@PROFILE
def prune_fit_range(covinv_jack, coords_jack, debug=False):
    """Zero out parts of the inverse covariance matrix to exclude items
    from fit range.
    Thus, the contribution to chi^2 (or, usually, t^2) will be 0.
    """
    excl = latfit.config.FIT_EXCL
    dimops1 = len(excl) == 1
    for i, xcoord in enumerate(coords_jack[:, 0]):
        if not debug:
            break
        for opa, _ in enumerate(excl):
            for j, _ in enumerate(excl[opa]):
                if xcoord == excl[opa][j]:
                    if not dimops1:
                        assert covinv_jack[
                            i, :, opa, :].all() == 0, "Prune failed."
                        assert covinv_jack[
                            :, i, opa, :].all() == 0, "Prune failed."
                        assert covinv_jack[
                            :, i, :, opa].all() == 0, "Prune failed."
                        assert covinv_jack[
                            i, :, :, opa].all() == 0, "Prune failed."
                    else:
                        assert covinv_jack[i, :].all() == 0, "Prune failed."
                        assert covinv_jack[:, i].all() == 0, "Prune failed."
                        assert covinv_jack[:, i].all() == 0, "Prune failed."
                        assert covinv_jack[i, :].all() == 0, "Prune failed."
    return covinv_jack

@PROFILE
def prune_phase_shift_arr(arr):
    """Get rid of jackknife samples for which the phase shift calc failed.
    (useful for testing, not useful for final output graphs)
    """
    dellist = []
    for i, phi in enumerate(arr):
        if np.isnan(em.acsum(phi)):  # delete the config
            print("Bad phase shift in jackknife block # "+
                  str(i)+", omitting.")
            dellist.append(i)
            raise ZetaError(
                "bad phase shift (nan)") # remove this if debugging
    return dellist

@PROFILE
def phase_shift_jk(params, epipi_arr):
    """Compute the nth jackknifed phase shift"""
    try:
        if params.dimops > 1 or GEVP:
            retlist = [zeta(epipi) for epipi in epipi_arr]
        else:
            retlist = zeta(epipi_arr)
    except ZetaError:
        retlist = None
        raise
    return retlist


@PROFILE
def copy_block(params, blk, out):
    """Copy a jackknife block (for a particular config)
    for later possible modification"""
    if params.dimops > 1 or GEVP:
        for time in range(len(params.time_range)):
            out[time, 1] = copy.deepcopy(np.nan_to_num(blk[time]))
    else:
        out[:, 1] = copy.deepcopy(np.nan_to_num(blk))
    return out


@PROFILE
def alloc_errbar_arr(params, time_length):
    """Allocate an array. Each config gives us a jackknife fit,
    and a set of error bars.
    We store the error bars in this array, indexed
    by config.
    """
    if params.dimops > 1 or GEVP:
        errbar_arr = np.zeros((params.num_configs, time_length,
                               params.dimops),
                              dtype=np.float)
    else:
        errbar_arr = np.zeros((params.num_configs, time_length),
                              dtype=np.float)
    return errbar_arr


@PROFILE
def get_covjack(cov_factor, params):
    """Get jackknife estimate of covariance matrix for this fit,
    note: no normalization is done here (it will need to be normalized
    depending on the jackknifing scheme)
    Also, the GEVP covjack has indexing that goes
    time, gevp index, time, gevp index
    """
    if params.dimops == 1 and not GEVP:  # i.e. if not using the GEVP
        covjack = np.einsum('ai, aj->ij', cov_factor, cov_factor)
    else:
        covjack = np.einsum('aim, ajn->imjn', cov_factor, cov_factor)
    return covjack

def dropimag(arr):
    """Drop the imaginary part if it's all 0"""
    if np.all(np.imag(arr) == 0.0):
        ret = np.real(arr)
    else:
        ret = arr
    return ret

def svd_check(corrjack):
    """Check the correlation matrix eigenvalues
    cut on condition numbers > 10^10"""
    evals = np.linalg.eigvals(corrjack)
    evals = sorted(list(np.abs(evals)))
    assert len(evals) > 1, str(evals)
    cond = evals[-1]/evals[0]
    assert cond >= 1, str(evals)+" "+str(cond)
    if cond > 1e10:
        print("condition number of correlation matrix is > 1e10:", cond)
        raise PrecisionLossError

if CORRMATRIX:
    @PROFILE
    def invert_cov(covjack, params2):
        """Invert the covariance matrix via correlation matrix
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params2.dimops == 1:  # i.e. if not using the GEVP
            if UNCORR:
                covjack = np.diagflat(np.diag(covjack))
            covjack = dropimag(covjack)
            corrjack = np.zeros(covjack.shape)
            weightings = dropimag(np.sqrt(np.diag(covjack)))
            reweight = np.diagflat(1./weightings)
            corrjack = kdot(reweight, kdot(covjack, reweight))
            svd_check(corrjack)
            covinv_jack = kdot(kdot(reweight, inv(corrjack)), reweight)
        else:
            lent = len(covjack)  # time extent
            reweight = np.zeros((lent, params2.dimops,
                                 lent, params2.dimops))
            for i in range(lent):
                for j in range(params2.dimops):
                    reweight[i][j][i][j] = 1.0/np.sqrt(covjack[i][j][i][j])
            corrjack = actensordot(
                actensordot(reweight, covjack), reweight)
            if UNCORR:
                diagcorr = np.zeros(corrjack.shape)
                for i in range(lent):
                    for j in range(params2.dimops):
                        diagcorr[i][j][i][j] = corrjack[i][j][i][j]
                corrjack = diagcorr
            covinv_jack = swap(actensordot(reweight, actensordot(
                tensorinv(corrjack), reweight)), 1, 2)
        return covinv_jack

else:
    @PROFILE
    def invert_cov(covjack, params2):
        """Invert the covariance matrix,
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params2.dimops == 1:  # i.e. if not using the GEVP
            covinv_jack = inv(covjack)
        else:
            covinv_jack = swap(tensorinv(covjack), 1, 2)
        return covinv_jack

if JACKKNIFE_FIT == 'SINGLE':
    def normalize_covinv(covinv_jack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the covinv (single jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = ((nsize)*(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covinv_jack * norm

    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (single jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = 1/ ((nsize)*(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covjack * norm

elif JACKKNIFE_FIT == 'DOUBLE':
    @PROFILE
    def normalize_covinv(covinv_jack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the covinv (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = ((nsize)/(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covinv_jack * norm

    @PROFILE
    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = 1/((nsize)/(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covjack * norm


def nconfigs_minus_block(total_configs, bsize):
    """Number of configs less block size
    used for normalizing cov
    block reduction is eliminated if bootstrapping
    """
    if latfit.config.BOOTSTRAP:
        ret = total_configs
    else:
        ret = total_configs-bsize
    return ret

def update_coords(coords, reuse):
    """Update coords (for debugging only)"""
    mean = em.acmean(reuse)
    assert len(coords) == len(mean)
    for i, _ in enumerate(coords):
        coords[i][1] = mean[i]
    return coords

@PROFILE
def get_doublejk_data(params, coords_jack, reuse,
                      reuse_blocked, config_num):
    """Primarily, get the inverse covariance matrix for the particular
    double jackknife fit we are on (=config_num)
    reuse_inv is the original unjackknifed data
    coords_jack are the coordinates, we also return truncated coordinates
    if the fit fails.
    """

    reuse_new = np.array(copy.deepcopy(np.array(reuse)))
    reuse_blocked_new = np.array(copy.deepcopy(np.array(reuse_blocked)))
    reuse_new = apply_shift(reuse_new, coords_jack)
    reuse_blocked_new = apply_shift(reuse_blocked_new, coords_jack)

    # coords_jack_new = update_coords(coords_jack, reuse_new)

    # original data, obtained by reversing single jackknife procedure
    # we set the noise level to mimic the jackknifed data
    # so the random data is in some sense already "jackknifed"
    if RANDOMIZE_ENERGIES:
        reuse_inv = np.array(copy.deepcopy(np.array(reuse_new)))
    else:
        reuse_inv = inverse_jk(reuse_new, params.num_configs)

    # bootstrap ensemble
    reuse_blocked_new, coords_jack_new = bootstrap_ensemble(
        reuse_inv, coords_jack, reuse_blocked_new)

    # delete a block of configs
    reuse_inv_red = delblock(config_num, reuse_inv)

    flag = 2
    while flag > 0:
        cov_factor = getcovfactor(params, reuse_blocked_new,
                                  config_num, reuse_inv_red)
        covjack = get_covjack(cov_factor, params)
        covinv_jack_pruned, flag = prune_covjack(params, covjack,
                                                 coords_jack_new, flag)
    covinv_jack = prune_fit_range(covinv_jack_pruned, coords_jack_new)
    return coords_jack_new, covinv_jack, jack_errorbars(covjack, params)

def compare_correlations(coords_jack, coords_jack_new):
    """Pearson r examination"""
    cj1 = []
    cjnew = []
    for i, _ in enumerate(coords_jack):
        cj1.append(coords_jack[i][1])
        cjnew.append(coords_jack_new[i][1])
    cj1 = np.asarray(cj1).T
    cjnew = np.asarray(cjnew).T
    for i, (ena, enb) in enumerate(zip(cj1, cjnew)):
        print("pearson r of en:", i, ":", stats.pearsonr(ena, enb))
        print("std of en:", i, ":", np.std(ena), np.std(enb))



@PROFILE
def prune_covjack(params, covjack, coords_jack, flag):
    """Prune the covariance matrix based on aonfig excluded time slices"""
    excl = []
    len_time = len(params.time_range)
    # convert x-coordinates to index basis
    for i, xcoord in enumerate(coords_jack[:, 0]):
        for opa, _ in enumerate(latfit.config.FIT_EXCL):
            for j in range(len(latfit.config.FIT_EXCL[opa])):
                if xcoord == latfit.config.FIT_EXCL[opa][j]:
                    excl.append(opa*len_time+i)
    # rotate tensor basis to dimops, dimops, time, time
    # (or time, time if not GEVP)
    covjack = swap(covjack, len(covjack.shape)-1, 0)
    # invert the pruned covariance matrix
    marray, flag = invertmasked(params, len_time, excl, covjack)
    symp(marray)
    covinv_jack = np.zeros(covjack.shape, dtype=float)
    if params.dimops == 1 and not GEVP:
        covinv_jack = np.copy(marray.data)
    else:
        # fill in the pruned dimensions with 0's
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                data = marray.data[
                    opa*len_time:(opa+1)*len_time,
                    opb*len_time:(opb+1)*len_time]
                assert np.asarray(data).shape == np.asarray(
                    covinv_jack[opa, opb]).shape
                covinv_jack[opa, opb, :, :] = data
        # to put things back in time, time, dimops, dimops basis
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-1, 0)
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-2, 1)
    return covinv_jack, flag

def symp(matrix):
    """Assert matrix is symmetric"""
    for i, _ in enumerate(matrix):
        for j, _ in enumerate(matrix):
            eva = matrix[i][j]
            evb = matrix[j][i]
            err = str(eva)+" "+str(evb)
            try:
                assert eva == evb, err
            except (AssertionError, ValueError):
                try:
                    assert np.allclose(eva, evb, rtol=1e-8)
                except AssertionError:
                    print(err)
                    raise PrecisionLossError


def invp(matrix, invf):
    """Check if matrix is inverted"""
    assert np.allclose(np.dot(matrix, invf), np.eye(len(invf)), rtol=1e-14)

@PROFILE
def invertmasked(params, len_time, excl, covjack):
    """invert the covariance matrix with pruned operator basis
    and fit range"""
    dim = int(np.sqrt(np.prod(list(covjack.shape))))
    matrix = np.zeros((dim, dim))
    if params.dimops == 1 and not GEVP:
        matrix = np.copy(covjack)
    else:
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                if opa != opb and UNCORR_OP:
                    matrix[opa*len_time:(opa+1)*len_time,
                           opb*len_time:(opb+1)*len_time] = 0
                else:
                    matrix[opa*len_time:(opa+1)*len_time,
                           opb*len_time:(opb+1)*len_time] = covjack[
                               opa, opb, :, :]
    mask = np.zeros(matrix.shape)
    mask[excl, :] = 1
    mask[:, excl] = 1
    marray = ma.masked_array(matrix, dtype=float,
                             fill_value=0, copy=True, mask=mask)
    matrix = np.delete(matrix, excl, axis=0)
    matrix = np.delete(matrix, excl, axis=1)
    if invertmasked.params2 is None:
        params2 = namedtuple('temp', ['dimops', 'num_configs'])
        params2.dimops = 1
        params2.num_configs = params.num_configs
    try:
        matrix_inv = invert_cov(matrix, params2)
        invp(matrix_inv, matrix)
        matrix = matrix_inv
        symp(matrix)
        marray[np.logical_not(marray.mask)] = normalize_covinv(
            matrix, params2).reshape(-1)
        flag = 0
    except np.linalg.linalg.LinAlgError as err:
        if str(err) == 'Singular matrix':
            print("Covariance matrix is singular",
                  "in jackknife fit.")
            print("Attempting to continue",
                  "fit with every other time slice",
                  "eliminated.")
            print("Plotted error bars should be " +
                  "considered suspect.")
            raise np.linalg.linalg.LinAlgError
            # flag = 1 # old way of handling error
        raise
    marray[marray.mask] = marray.fill_value
    return marray, flag
invertmasked.params2 = None

@PROFILE
def jack_errorbars(covjack, params):
    """Get error bars for this fit,
    given the covariance matrix
    """
    covjack = normalize_cov(covjack, params)
    lent = len(covjack)
    if len(covjack.shape) == 4:
        error_bars = np.zeros((covjack[0][0].shape), dtype=np.float)
        for i in range(lent):
            error_bars[i] = np.sqrt(np.diag(covjack[i, :, i, :]))
    elif len(covjack.shape) == 2:
        error_bars = np.sqrt(np.diag(covjack))
    else:
        print("***ERROR***")
        print("badly formed covariance matrix " +
              "provided to jackknife errorbar finder")
        print("shape =", covjack.shape)
        sys.exit(1)
    return error_bars

if JACKKNIFE_FIT == 'SINGLE':
    def getcovfactor(_, reuse_blocked, config_num, reuse_inv_red):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        block == reuse
        SINGLE elimination jackknife
        SINGLE case is just the usual way to calculate the covariance matrix
(no jackknifing after the first jackknife)
        """
        return reuse_inv_red-reuse_blocked[config_num]

elif JACKKNIFE_FIT == 'DOUBLE':
    @PROFILE
    def getcovfactor(params, reuse_blocked, config_num,
                     reuse_inv_red, bsize=JACKKNIFE_BLOCK_SIZE):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        inverse block with the ith config
        or block of configs deleted == reuse_inv_red.
        block == reuse
        DOUBLE elimination jackknife
        """
        lcon = len(reuse_blocked) if RANDOMIZE_ENERGIES else 1
        if latfit.config.BOOTSTRAP or RANDOMIZE_ENERGIES:
            ret = (reuse_blocked-em.acmean(
                reuse_blocked, axis=0))/np.sqrt(lcon)
        else:
            num_configs_reduced = (params.num_configs-1)*bsize
            assert len(reuse_inv_red) == num_configs_reduced
            try:
                assert np.allclose(em.acmean(reuse_inv_red, axis=0),
                                   reuse_blocked[config_num], rtol=1e-14)
            except AssertionError:
                print(em.acmean(reuse_inv_red, axis=0))
                print(reuse_blocked[config_num])
                raise
            ret = np.array([
                np.mean(
                    np.delete(
                        reuse_inv_red, i, axis=0), axis=0)
                for i in range(num_configs_reduced)]) -\
                    reuse_blocked[config_num]
            assert len(ret) == num_configs_reduced
        return ret
