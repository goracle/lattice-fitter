"""Fit under a jackknife"""
import sys
import copy
import os
import pickle
import mpi4py
from mpi4py import MPI
from scipy import stats
import numpy as np

from latfit.makemin.mkmin import mkmin
from latfit.analysis.superjack import jack_mean_err
from latfit.mathfun.block_ensemble import block_ensemble

# error
from latfit.analysis.errorcodes import NoConvergence, TooManyBadFitsError
from latfit.analysis.errorcodes import BadChisq, BadJackknifeDist
from latfit.analysis.errorcodes import EnergySortError, ZetaError

from latfit.analysis.result_min import ResultMin
from latfit.analysis.covops import get_doublejk_data

# util
from latfit.utilities.postfit.compare_print import trunc
from latfit.utilities import exactmean as em
from latfit.utilities.zeta.zeta import zeta

# config
from latfit.config import START_PARAMS
from latfit.config import JACKKNIFE_FIT, UNCORR
from latfit.config import EFF_MASS
from latfit.config import GEVP, FIT_SPACING_CORRECTION
from latfit.config import NOLOOP, ALTERNATIVE_PARALLELIZATION
from latfit.config import SYS_ENERGY_GUESS
from latfit.config import PVALUE_MIN, NOATWSUB, PIONRATIO
from latfit.config import PICKLE, MATRIX_SUBTRACTION
from latfit.config import CALC_PHASE_SHIFT, PION_MASS
from latfit.config import SUPERJACK_CUTOFF, SLOPPYONLY
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.config import DELTA_E2_AROUND_THE_WORLD
from latfit.config import ISOSPIN, VERBOSE
from latfit.config import SKIP_OVERFIT

# dynamic
import latfit.analysis.hotelling as hotelling
import latfit.finalout.mkplot
import latfit.config
import latfit.analysis.misc as misc
import latfit.makemin.mkmin as mkmin


MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False
DOWRITE = ALTERNATIVE_PARALLELIZATION and not MPIRANK\
    or not ALTERNATIVE_PARALLELIZATION


SUPERJACK_CUTOFF = 0 if SLOPPYONLY else SUPERJACK_CUTOFF

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


if JACKKNIFE_FIT == 'FROZEN':
    pass

elif JACKKNIFE_FIT in ('DOUBLE', 'SINGLE'):
    @PROFILE
    def jackknife_fit(meta, params, reuse, reuse_blocked, coords):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        (N.B. substitute t^2 for chi^2 if doing a correlated fit)
        """
        # storage for results
        result_min = ResultMin(meta, params, coords)
        coords_jack = np.copy(copy.deepcopy(coords))

        # p-value fiducial cut:  cut below this pvalue
        # as it is 5 \sigma away from an acceptable chi^2/dof (t^2/dof)
        chisq_fiduc_cut = hotelling.chisqfiduc(params.num_configs,
                                               result_min.misc.dof)
        # similarly, cut below 5 sigma below chisq/dof = 1
        chisq_fiduc_overfit_cut = hotelling.overfit_chisq_fiduc(
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

            halfway = int(np.floor(len(config_range)/2)) == list(
                config_range).index(config_num)

            if ALTERNATIVE_PARALLELIZATION:
                assert not latfit.config.BOOTSTRAP, "not supported"
                if config_num not in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF]:
                    if config_num % MPISIZE != MPIRANK and MPISIZE > 1:
                        if not (ISOSPIN == 0 and halfway):
                            continue

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

            if ISOSPIN == 0 and halfway:
                start_loop = True

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
               list(result_min.systematics.arr[config_num][:-1]) and VERBOSE:
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
                      sys_str, hotelling.torchi(),
                      trunc(result_min_jack.fun/result_min.misc.dof),
                      "p-value=", trunc(result_min.pvalue.arr[config_num]),
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

        if ALTERNATIVE_PARALLELIZATION:
            result_min.gather()

        # pickle/unpickle the jackknifed arrays
        result_min = pickl(result_min)

        # compute p-value jackknife uncertainty
        result_min.pvalue.val, result_min.pvalue.err =\
            jack_mean_err(result_min.pvalue.arr)

        # print out the jackknife blocks for manual management
        if NOLOOP:
            result_min.printjack(meta)

        # get the optimal params
        result_min.min_params.val, result_min.min_params.err = jack_mean_err(
            result_min.min_params.arr)

        # compute the mean, error on the params
        result_min.energy.val, result_min.energy.err = jack_mean_err(
            result_min.energy.arr)
        if VERBOSE and DOWRITE:
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

        if VERBOSE and DOWRITE:
            print(hotelling.torchi(), result_min.chisq.val/result_min.misc.dof,
                  "std dev:", np.std(result_min.chisq.arr, ddof=1))

        return result_min, result_min.energy.err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

@PROFILE
def toomanybadfitsp(result_min):
    """If there have already been too many fits with large chi^2 (t^2),
    the average chi^2 (t^2) is probably not going to be good
    so abort the fit.
    """
    avg = em.acmean(result_min.chisq.arr)
    pvalue = result_min.funpvalue(avg)
    cond = pvalue < PVALUE_MIN and not latfit.config.BOOTSTRAP and not NOLOOP
    if cond:
        raise TooManyBadFitsError(chisq=avg/result_min.misc.dof,
                                  pvalue=pvalue, uncorr=UNCORR)


def skip_range(params, result_min, skip_votes,
               result_min_jack, chisq_fiduc_cut):
    """Raise an error if we should skip this fit range"""
    skiprange = False
    zero = 0+SUPERJACK_CUTOFF
    one = 1+SUPERJACK_CUTOFF
    nconf = params.num_configs-SUPERJACK_CUTOFF
    dof = result_min.misc.dof
    # don't skip the fit range until we confirm
    # on 2nd config
    var = np.sqrt(hotelling.var(result_min.misc.dof, nconf))
    var_approx = np.sqrt(2*dof)
    div = 1/np.sqrt(nconf-1)
    diff = abs(result_min_jack.fun-result_min.chisq.arr[zero])
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
        skiprange = diff < 5*var*div
    if skiprange and not latfit.config.BOOTSTRAP and not NOLOOP:
        raise BadChisq(
            chisq=result_min_jack.fun/result_min.misc.dof,
            dof=result_min.misc.dof, uncorr=UNCORR)
    if skip_votes:
        # the second sample should never have a good fit
        # if the first one has that bad a fit
        if DOWRITE:
            print("fiducial cut =", chisq_fiduc_cut)
            print("dof=", result_min.misc.dof)
            print("first two chi^2's:",
                result_min_jack.fun, result_min.chisq.arr[zero])
            print("var, var_approx, div, diff", var, var_approx, div, diff)
            print("Bad jackknife distribution:"+\
                    str(result_min.chisq.arr[zero]/result_min.misc.dof)+" "+\
                    str(result_min.chisq.arr[one]/result_min.misc.dof)+" "+\
                    str(result_min.pvalue.arr[zero])+" "+\
                    str(result_min.pvalue.arr[one])+" ")
        #sys.exit(1)
        if not latfit.config.BOOTSTRAP and not NOLOOP:
            raise BadJackknifeDist(uncorr=UNCORR)

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
            if VERBOSE:
                print("miss-sorted energies:", ret)
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
def pickl(result_min):
    """Pickle or unpickle the results from the jackknife fit loop
    to do: make more general use **kwargs
    """
    if PICKLE == 'pickle':
        pickle.dump(result_min.energy.arr,
                    open(unique_pickle_file("result_min.energy.arr"), "wb"))
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
