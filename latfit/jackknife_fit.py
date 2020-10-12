"""Fit under a jackknife"""
import sys
import copy
import os
import pickle
from multiprocessing import Pool
from multiprocessing import current_process
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
from latfit.analysis.errorcodes import FitSuccess

from latfit.analysis.result_min import ResultMin, funpvalue
from latfit.analysis.covops import get_doublejk_data

# multiproc, id of worker
from latfit.analysis.uniqueid import unique_id

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

def cuts_chisq(num_configs, dof):
    """Get cutoff values for bad/good chisq"""
    # p-value fiducial cut:  cut below this pvalue
    # as it is 5 \sigma away from an acceptable chi^2/dof (t^2/dof)
    chisq_fiduc_cut = hotelling.chisqfiduc(num_configs, dof)

    # similarly, cut below 5 sigma below chisq/dof = 1
    chisq_fiduc_overfit_cut = hotelling.overfit_chisq_fiduc(
        num_configs, dof)

    return chisq_fiduc_cut, chisq_fiduc_overfit_cut


if JACKKNIFE_FIT == 'FROZEN':
    pass

elif JACKKNIFE_FIT in ('DOUBLE', 'SINGLE'):
    @PROFILE
    def jackknife_fit(meta, params, reuse_coords, fullfit=True):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        (N.B. substitute t^2 for chi^2 if doing a correlated fit)
        """
        # storage for results
        result_min = ResultMin(meta, params, reuse_coords[2])

        # loop over configs, doing a fit for each one
        # rearrange loop order so we can check goodness of fit
        # on more stable sloppy samples
        skip_votes = []
        config_range = (np.array(range(
            params.num_configs)) + SUPERJACK_CUTOFF) % params.num_configs
        config_range = range(
            params.num_configs) if latfit.config.BOOTSTRAP else config_range

        # arguments which don't change with respect to jackknife sample
        const_args = (config_range, params, reuse_coords,
                      result_min.misc.dof, fullfit)

        if not fullfit or ALTERNATIVE_PARALLELIZATION or\
           current_process().name != 'MainProcess':

            # make sure we are detecting a running pool instead of some
            # other set of processes running outside of what we expect
            if current_process().name != 'MainProcess':
                assert 'ForkPoolWorker' in current_process().name,\
                    current_process().name

            for config_num in config_range:

                res_dict = jackknife_iter(config_num, const_args)

                if not res_dict:
                    if ALTERNATIVE_PARALLELIZATION:
                        assert meta.options.procs == 1, meta.options.procs
                        continue
                    assert None, "we should not be here; bug"

                # store result
                result_min.store_dict(res_dict, config_num)

                # post min check(s)
                # check if chi^2 average
                # (even with padded zeros for configs we haven't fitted yet)
                # is already too bad
                toomanybadfitsp(result_min)

                # check first two configs for an early abort
                # use sloppy configs to check if fit will work
                skip_votes = first_two_configs_abort(
                    (params.num_configs, config_num), result_min, skip_votes)

        else:

            # we are doing the full fit, so parallelize
            # over jackknife samples using multiproc
            result_min = multiprocess_jackknife(
                result_min, meta, config_range, const_args)

        # post fit processing
        result_min = post_fit(meta, result_min)
        # perform a check to be sure all energy values found are non-zero
        assert np.all(result_min.energy.arr), result_min.energy.arr

        return result_min, result_min.energy.err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

def multiprocess_jackknife(result_min, meta, config_range, const_args):
    """Do jackknife fits in parallel using multiprocess"""

    _, _, _, _, fullfit = const_args
    assert fullfit, "if we aren't doing a full fit,"+\
        " we don't need to parallelize"

    # do first config in common to set the guess
    res_dict = jackknife_iter(config_range[0], const_args)
    result_min = post_single_iter(result_min, res_dict, config_range[0])

    # reset the guess in I=0,
    # since fits are more unstable and we don't want to underestimate the noise.
    if not ISOSPIN:  
        # setup
        hidx = list(config_range).index(half_range(config_range))
        range1 = config_range[1:hidx]
        range2 = config_range[hidx+1:]

        # everything after the first config up to,
        # but not including, the halfway point
        result_min = call_multiproc(result_min, range1,
                                    meta.options.procs, const_args)

        # halfway point, to reset the guess
        res_dict = jackknife_iter(config_range[hidx], const_args)
        result_min = post_single_iter(result_min, res_dict, config_range[hidx])

    else:
        range2 = config_range[1:]

    # everything else
    result_min = call_multiproc(result_min, range2,
                                meta.options.procs, const_args)

    return result_min

def call_multiproc(result_min, config_range, procs, const_args):
    """Do the jackknife multiprocessing"""
    argtup = [(config_num, const_args) for config_num in config_range]
    poolsize = min(procs, len(argtup))
    with Pool(poolsize) as pool:
        results = pool.starmap(jackknife_iter, argtup)
    for config_num, res_dict in zip(config_range, results):
        result_min = post_single_iter(result_min, res_dict, config_num)
    return result_min


def post_single_iter(result_min, res_dict, config_num):
    """Post-process a single jackknife iteration"""
    result_min.store_dict(res_dict, config_num)
    toomanybadfitsp(result_min)
    return result_min

def jackknife_iter(config_num, const_args):
    """Fit to one jackknife sample
    (function to be parallelized)
    """
    # unpack
    config_range, params, reuse_coords, dof, fullfit = const_args
    reuse, reuse_blocked, _ = reuse_coords

    # check if we should skip this iteration
    loop_location = update_location(config_num, config_range)
    loop_location['skip'] = False
    if location_check(loop_location, fullfit):
        loop_location['skip'] = True
        #continue

    res_dict = {}
    if not loop_location['skip']:
        # get the data for the minimizer
        # (and the error bars)
        coords_jack = get_jack_coords(params, config_num, reuse_coords)
        coords_jack, covinv_jack, res_dict['misc.error_bars'] = \
            get_doublejk_data(
                params, coords_jack, reuse, reuse_blocked, config_num)

        # minimize chi^2
        result_min_jack = find_min(
            params, coords_jack, covinv_jack, loop_location)

        res_dict['min_params'] = result_min_jack.x
        res_dict['chisq'] = result_min_jack.fun
        res_dict = store_result(params, res_dict)
        res_dict = compute_min_derived_quantities(
            params, res_dict, config_num, dof)

        print_single_config_info(res_dict, dof, config_num)

    return res_dict


def location_check(loop_location, fullfit):
    """Check if we should skip this iteration"""
    config_num = loop_location['num']
    ret = False
    if not fullfit:
        if config_num not in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF]:
            raise FitSuccess

    # this skip is only for MPI parallelization of loop
    if ALTERNATIVE_PARALLELIZATION:
        assert not latfit.config.BOOTSTRAP, "not supported"

        # all MPI ranks should calculate the first two (check, sloppy) configs
        if config_num not in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF]:

            # check rank
            if config_num % MPISIZE != MPIRANK and MPISIZE > 1:

                # we need to not skip the midpoint either
                if not (not ISOSPIN and loop_location['halfway']):
                    ret = True
    return ret

def update_location(config_num, config_range):
    """Update where we are in the loop"""
    loop_location = {'start_loop': False}
    loop_location['num'] = config_num
    loop_location['halfway'] = ishalfway(config_num, config_range)
    if loop_location['halfway']:
        assert half_range(config_range) == config_num, (
            config_range, config_num)
    if config_num == config_range[0] or (
            loop_location['halfway'] and not ISOSPIN):
        loop_location['start_loop'] = True
    return loop_location

def ishalfway(config_num, config_range):
    """Find out if config_num is halfway through config_range"""
    ret = int(np.floor(len(config_range)/2)) == list(
        config_range).index(config_num)
    return ret

def half_range(config_range):
    """Find halfway point of range config_range"""
    retset = False
    ret = None
    for i in config_range:
        if ishalfway(i, config_range):
            ret = i
            retset = True
            break
    assert retset, ("bug:", config_range)
    return ret


def get_jack_coords(params, config_num, reuse_coords):
    """Get (jackknife) sample coordinates to fit to
    copy the jackknife block into coords_jack
    """
    reuse, reuse_blocked, coords = reuse_coords
    coords_jack = np.copy(copy.deepcopy(coords))
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
        coords_jack = copy_block(params, reuse[0], coords_jack)

    return coords_jack



def compute_min_derived_quantities(params, res_dict, config_num, dof):
    """Compute quantities derived from optimal function params"""

    # we shifted the GEVP energy spectrum down
    # to fix the leading order around the world term
    # so shift it back
    if not latfit.config.BOOTSTRAP:
        res_dict['energy'] += correction_en(
            res_dict['energy'], config_num, params.num_configs)

    # compute phase shift, if necessary
    if CALC_PHASE_SHIFT and not latfit.config.BOOTSTRAP:
        res_dict['phase_shift'] = phase_shift_jk(params, res_dict['energy'])

    # compute p value for this fit
    res_dict['pvalue'] = funpvalue(res_dict['chisq'], dof, params.num_configs)
    assert not np.isnan(res_dict['pvalue']), "pvalue is nan"
    return res_dict


def find_min(params, coords_jack, covinv_jack, loop_location):
    """Find chi^2 min"""

    # where are we in the fit loop?
    # we reuse min guesses, so our min finder depends on the
    # loop location
    # halfway = loop_location['halfway']
    # config_num = loop_location['num']

    start_loop = loop_location['start_loop']

    # I = 0 has been observed to get caught in a local min
    # if we reuse the result for a guess.
    # Thus, resetting the guess halfway ensures we don't
    # underestimate the error
    #if ISOSPIN == 0 and halfway:
    #    start_loop = True

    if start_loop or not mkmin.prealloc_chi.allocd:
        mkmin.SPARAMS = list(np.copy(START_PARAMS))
        mkmin.PARAMS = params
        mkmin.prealloc_chi(covinv_jack, coords_jack)
        start_loop = True

    print("rank:", MPIRANK, "worker id:", unique_id(),
          "config:", loop_location['num'], "params:", mkmin.SPARAMS)

    # minimize chi^2 (t^2) given the inv. covariance matrix and data
    result_min_jack = mkmin.mkmin(covinv_jack, coords_jack)
    if result_min_jack.status != 0:
        assert not np.isnan(result_min_jack.status),\
            str(result_min_jack.status)
        #result_min.misc.status = result_min_jack.status
        raise NoConvergence

    if start_loop:
        mkmin.SPARAMS = result_min_jack.x
        #start_loop = False

    return result_min_jack




if VERBOSE or ALTERNATIVE_PARALLELIZATION:
    def print_single_config_info(res_dict, dof, config_num):
        """Prints for a fit to a single jackknife sample"""

        sys_str = str(res_dict['systematics'][-1])\
            if not np.isnan(res_dict['systematics'][-1])\
                else ''

        if res_dict['chisq']/dof < 10 and\
           list(res_dict['systematics'][:-1]):
            print('systematics:', res_dict['systematics'][:-1],
                  "config:", config_num)

        # print results for this config
        print("config", config_num, ":", res_dict['energy'],
              sys_str, hotelling.torchi(),
              trunc(res_dict['chisq']/dof),
              "p-value=", trunc(res_dict['pvalue']),
              'dof=', dof, "rank=", MPIRANK)

else:
    def print_single_config_info(*_):
        """verbose mode turned off; print nothing"""


def first_two_configs_abort(config_info, result_min, skip_votes):
    """Check first two configs;
    throw an error to abort fit loop if they are bad"""

    # setup/unpack
    num_configs, config_num = config_info
    chisq_cuts = cuts_chisq(num_configs, result_min.misc.dof)
    chisq_fiduc_cut, chisq_fiduc_overfit_cut = chisq_cuts

    if config_num in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF] and\
        not latfit.config.BOOTSTRAP:

        # check if chi^2 (t^2) too big, too small
        chisq = result_min.chisq.arr[config_num]
        if chisq > chisq_fiduc_cut or\
            (SKIP_OVERFIT and chisq < \
            chisq_fiduc_overfit_cut):
            skip_votes.append(config_num)

        if config_num == 1+SUPERJACK_CUTOFF:
            skip_range(num_configs, result_min,
                       skip_votes, chisq_fiduc_cut, chisq)

    return skip_votes


def post_fit(meta, result_min):
    """Post jackknife fit loop tasks"""

    # reset the precomputed quantities
    mkmin.dealloc_chi()
    # average results, compute jackknife uncertainties

    if ALTERNATIVE_PARALLELIZATION:
        result_min.gather()

    # retrieve/save result
    result_min = post_fit_io(meta, result_min)

    # compute std error and average of fit results
    result_min = post_fit_avg_err(result_min)

    # print any summary statements
    post_fit_statements(result_min)

    return result_min

def post_fit_avg_err(result_min):
    """Compute averages and error of loop result"""
    # compute p-value jackknife uncertainty
    result_min.pvalue.val, result_min.pvalue.err =\
        jack_mean_err(result_min.pvalue.arr)

    # get the optimal params
    result_min.min_params.val, result_min.min_params.err = jack_mean_err(
        result_min.min_params.arr)

    # compute the mean, error on the params
    result_min.energy.val, result_min.energy.err = jack_mean_err(
        result_min.energy.arr)

    # compute the systematics and errors
    if SYS_ENERGY_GUESS is not None:
        result_min.systematics.val, result_min.systematics.err =\
            jack_mean_err(result_min.systematics.arr)

    # average the point by point error bars
    result_min.misc.error_bars = em.acmean(result_min.misc.error_bars, axis=0)

    # compute phase shift and error in phase shift
    if CALC_PHASE_SHIFT:
        phase_shift_scatter = phase_shift_scatter_len_avg(result_min)

        result_min = unpack_min_data(result_min, *phase_shift_scatter)

    # compute mean, jackknife uncertainty of chi^2 (t^2)
    result_min.chisq.val, result_min.chisq.err = jack_mean_err(
        result_min.chisq.arr)

    return result_min


def post_fit_io(meta, result_min):
    """Do i/o with result of fit loop"""
    # pickle/unpickle the jackknifed arrays
    # print out the jackknife blocks for manual management
    result_min = pickl(result_min)
    if NOLOOP:
        result_min.printjack(meta)
    return result_min


if VERBOSE:
    def post_fit_statements(result_min):
        """Prints after the fit loop"""
        print('param err:', result_min.energy.err,
              'np.std:', np.std(result_min.energy.arr, axis=0))

        print(hotelling.torchi(), result_min.chisq.val/result_min.misc.dof,
              "std dev:", np.std(result_min.chisq.arr, ddof=1))
else:
    def post_fit_statements(_):
        """print nothing; verbose mode is off"""


def store_result(params, res_dict):
    """store the result for this config_num"""
    res_dict['systematics'], _ = \
        getsystematic(params, res_dict['min_params'])
    res_dict['systematics'],\
        params.energyind = getsystematic(
            params, res_dict['min_params'])
    res_dict['energy'] = getenergies(
        params, res_dict['min_params'])
    return res_dict


@PROFILE
def toomanybadfitsp(result_min):
    """If there have already been too many fits with large chi^2 (t^2),
    the average chi^2 (t^2) is probably not going to be good
    so abort the fit.
    """
    avg = em.acmean(result_min.chisq.arr)
    pvalue = result_min.calc_pvalue(avg)
    cond = pvalue < PVALUE_MIN and not latfit.config.BOOTSTRAP and not NOLOOP
    if cond:
        raise TooManyBadFitsError(chisq=avg/result_min.misc.dof,
                                  pvalue=pvalue, uncorr=UNCORR)


def skip_range(num_configs, result_min, skip_votes, chisq_fiduc_cut, chisq):
    """Raise an error if we should skip this fit range"""
    skiprange = False
    zero = 0+SUPERJACK_CUTOFF
    one = 1+SUPERJACK_CUTOFF
    nconf = num_configs-SUPERJACK_CUTOFF
    dof = result_min.misc.dof
    # don't skip the fit range until we confirm
    # on 2nd config
    var = np.sqrt(hotelling.var(result_min.misc.dof, nconf))
    var_approx = np.sqrt(2*dof)
    div = 1/np.sqrt(nconf-1)
    diff = abs(chisq-result_min.chisq.arr[zero])
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
            chisq=chisq/result_min.misc.dof,
            dof=result_min.misc.dof, uncorr=UNCORR)
    if skip_votes:
        # the second sample should never have a good fit
        # if the first one has that bad a fit
        if DOWRITE:
            print("fiducial cut =", chisq_fiduc_cut)
            print("dof=", result_min.misc.dof)
            print("first two chi^2's:", chisq, result_min.chisq.arr[zero])
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
def correction_en(energy, config_num, num_configs):
    """Correct the jackknifed E_pipi"""
    delta_e_around_the_world = DELTA_E_AROUND_THE_WORLD
    delta_e2_around_the_world = DELTA_E2_AROUND_THE_WORLD
    if hasattr(delta_e_around_the_world, '__iter__') and\
       np.asarray(delta_e_around_the_world).shape:
        latw = len(delta_e_around_the_world)

        # block the ensemble if needed
        if latw != num_configs:
            delta_e_around_the_world = block_ensemble(
                num_configs, delta_e_around_the_world)

            latw = len(delta_e_around_the_world)
            assert latw in (1, num_configs),\
                "bug:  array mismatch"
        if delta_e2_around_the_world is not None:
            assert len(delta_e2_around_the_world) == latw

            if latw != num_configs:
                delta_e2_around_the_world = block_ensemble(
                    num_configs, delta_e2_around_the_world)

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
        corre3 = misc.correct_epipi(energy, config_num=config_num)
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
