"""Fit under a jackknife"""
import sys
import os
import numbers
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
from latfit.mathfun.block_ensemble import delblock
from latfit.mathfun.block_ensemble import bootstrap_ensemble

from latfit.config import START_PARAMS
from latfit.config import JACKKNIFE_FIT
from latfit.config import CORRMATRIX, EFF_MASS
from latfit.config import GEVP, FIT_SPACING_CORRECTION
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.config import UNCORR, SYS_ENERGY_GUESS
from latfit.config import PVALUE_MIN, NOATWSUB, PIONRATIO
from latfit.config import PICKLE, MATRIX_SUBTRACTION
from latfit.config import CALC_PHASE_SHIFT, PION_MASS
from latfit.config import SUPERJACK_CUTOFF, SLOPPYONLY
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.config import DELTA_E2_AROUND_THE_WORLD
from latfit.config import ISOSPIN
from latfit.config import SKIP_OVERFIT
from latfit.utilities.zeta.zeta import zeta, ZetaError
import latfit.finalout.mkplot
import latfit.config
import latfit.analysis.misc as misc
from latfit.utilities import exactmean as em
from latfit.utilities.actensordot import actensordot

SUPERJACK_CUTOFF = 0 if SLOPPYONLY else SUPERJACK_CUTOFF

MIN = namedtuple('min',
                 ['x', 'fun', 'status',
                  'pvalue', 'pvalue_err' 'chisq_err',
                  'error_bars', 'dof', 'phase_shift',
                  'phase_shift_err', 'scattering_length',
                  'scattering_length_err'])

WINDOW = []

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


class ResultMin:
    def __init__(self):
        self.x = None
        self.x_arr = None
        self.x_err = None
        self.fun = None
        self.status = None
        self.systematics = None
        self.systematics_arr = None
        self.systematics_err = None
        self.pvalue = None
        self.pvalue_err = None
        self.pvalue_arr = None
        self.chisq_err = None
        self.chisq_arr = None
        self.error_bars = None
        self.dof = None
        self.phase_shift = None
        self.phase_shift_err = None
        self.scattering_length = None
        self.scattering_length_arr = None
        self.scattering_length_err = None
        self.phase_shift_arr = None
        self.num_configs = None

    @PROFILE
    def alloc_sys_arr(self, params):
        """alloc array for systematics"""
        syslength = len(START_PARAMS)-params.dimops*(1 if GEVP else 0)
        syslength = max(1, syslength)
        self.systematics_arr = np.zeros((params.num_configs, syslength))
        self.num_configs = params.num_configs

    @PROFILE
    def funpvalue(self, chisq):
        """Give pvalue from Hotelling t^2 stastistic
        (often referred to incorrectly as chi^2;
        is actually a sort of correlated chi^2)
        """
        ret = None
        correction = (self.num_configs-self.dof)/(self.num_configs-1)
        correction /= self.dof
        correction = 1 if UNCORR else correction
        cor = correction
        if self.dof is not None:
            ret = stats.f.sf(chisq*cor, self.dof, self.num_configs-self.dof)
        return ret

def torchi():
    """Are we calculating Hotelling's t^2 statistic or a true chi^2?
    return the corresponding string.
    """
    if UNCORR:
        ret = 'chisq/dof='
    else:
        ret = 't^2/dof='
    return ret

if JACKKNIFE_FIT == 'FROZEN':
    def jackknife_fit(params, reuse, reuse_blocked, coords, covinv):
        """Fit under a frozen (single) jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        (N.B. substitute t^2 for chi^2 if doing a correlated fit)
        """
        assert 0, "not currently supported"
        result_min = namedtuple(
            'min', ['x', 'fun', 'status', 'chisq_err', 'dof'])
        result_min.status = 0
        result_min.dof = len(coords)*params.dimops-len(START_PARAMS)
        # one fit for every jackknife block (N fits for N configs)
        coords_jack = np.copy(coords)
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)))
        chisq_min_arr = np.zeros(params.num_configs)
        covinv_jack = covinv
        for config_num in range(params.num_configs):
            if params.dimops > 1:
                for time in range(len(params.time_range)):
                    coords_jack[time, 1] = reuse[config_num][time]
            else:
                coords_jack[:, 1] = reuse[config_num]
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status
                result_min_jack = mkmin(covinv_jack, coords_jack, 'minuit')
                result_min.status = result_min_jack.status
                if result_min_jack.status != 0:
                    result_min_jack = mkmin(
                        covinv_jack, coords_jack, 'L-BFGS-B')
                    result_min.status = result_min_jack.status
            print("config", config_num, ":",
                  result_min_jack.x, torchi(),
                  result_min_jack.fun/result_min.dof)
            chisq_min_arr[config_num] = result_min_jack.fun
            min_arr[config_num] = result_min_jack.x
        result_min.x = em.acmean(min_arr, axis=0)
        param_err = np.sqrt(params.prefactor*em.acsum(
            (min_arr-result_min.x)**2, 0))
        result_min.fun = em.acmean(chisq_min_arr)
        result_min.chisq_err = np.sqrt(params.prefactor*em.acsum(
            (chisq_min_arr-result_min.fun)**2))
        return result_min, param_err



elif JACKKNIFE_FIT == 'DOUBLE' or JACKKNIFE_FIT == 'SINGLE':
    @PROFILE
    def jackknife_fit(params, reuse, reuse_blocked, coords, covinv=None):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        (N.B. substitute t^2 for chi^2 if doing a correlated fit)
        """
        if covinv is None:
            pass

        # storage for results
        result_min = ResultMin()

        # no errors gives us 0 status
        result_min.status = 0

        # compute degrees of freedom
        result_min.dof = len(coords)*params.dimops-len(START_PARAMS)
        for i in coords[:, 0]:
            for j in latfit.config.FIT_EXCL:
                if i in j and i in WINDOW:
                    result_min.dof -= 1
        if result_min.dof < 1:
            print("dof < 1. dof =", result_min.dof)
            print("fit window:", WINDOW)
            print("excl:", latfit.config.FIT_EXCL)
            raise DOFNonPos(dof=result_min.dof)

        # alloc storage
        # one fit for every jackknife block (N fits for N configs)

        # fit by fit p-values
        result_min.pvalue = np.zeros(params.num_configs)

        #phase shift
        result_min.phase_shift = alloc_phase_shift(params)

        # allocate systematics array
        result_min.alloc_sys_arr(params)

        # allocate storage for jackknifed x,y coordinates
        coords_jack = np.copy(coords)

        # storage for fit by fit optimized fit params
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)
                            if not GEVP else params.dimops))

        # storage for fit by fit chi^2 (t^2)
        chisq_min_arr = np.zeros(params.num_configs)

        # fit by fit error bars (we eventually plot the averaged set)
        result_min.error_bars = alloc_errbar_arr(params, len(coords))

        # p-value fiducial cut:  cut below this pvalue
        # as it is 5 \sigma away from an acceptable chi^2/dof (t^2/dof)
        chisq_fiduc_cut = chisqfiduc(params.num_configs,
                                     result_min.dof)
        # similarly, cut below 5 sigma below chisq/dof = 1
        chisq_fiduc_overfit_cut = overfit_chisq_fiduc(
            params.num_configs, result_min.dof)

        skip_votes = []
        # loop over configs, doing a fit for each one
        # rearrange loop order so we can check goodness of fit
        # on more stable sloppy samples

        # blocked ensemble
        # reuse_blocked = block_ensemble(params, reuse)

        for config_num in (np.array(range(params.num_configs))+
                           SUPERJACK_CUTOFF)%params.num_configs:

            assert isinstance(config_num, np.int64), str(config_num)
            # if config_num>160: break # for debugging only

            # copy the jackknife block into coords_jack
            assert np.all(reuse[config_num] == reuse_blocked[config_num])
            copy_block(params, reuse[config_num], coords_jack)

            # get the data for the minimizer, and the error bars
            coords_jack, covinv_jack, result_min.error_bars[
                config_num] = get_doublejk_data(params, coords_jack,
                                                reuse, reuse_blocked,
                                                config_num)

            # minimize chi^2 (t^2) given the inv. covariance matrix and data
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status
                raise NoConvergence

            # store results for this fit
            chisq_min_arr[config_num] = result_min_jack.fun
            toomanybadfitsp(result_min, chisq_min_arr)

            # store the result
            temp, ind = getsystematic(params, result_min_jack.x)
            result_min.systematics_arr[config_num] = temp
            result_min.systematics_arr[config_num], params.energyind = getsystematic(
                params, result_min_jack.x)
            min_arr[config_num] = getenergies(params, result_min_jack.x)

            if result_min_jack.fun/result_min.dof < 10:
                print('systematics:', result_min.systematics_arr[config_num][:-1])

            # we shifted the GEVP energy spectrum down
            # to fix the leading order around the world term so shift it back
            min_arr[config_num] = np.asarray(min_arr[config_num])+\
                correction(min_arr, config_num)

            # compute phase shift, if necessary
            if CALC_PHASE_SHIFT:
                result_min.phase_shift[config_num] = phase_shift_jk(
                    params, min_arr[config_num])

            # compute p value for this fit
            result_min.pvalue[config_num] = result_min.funpvalue(
                result_min_jack.fun)

            # print results for this config
            print("config", config_num, ":", min_arr[config_num],
                  result_min.systematics_arr[config_num][-1],
                  torchi(), result_min_jack.fun/result_min.dof,
                  "p-value=", result_min.pvalue[config_num],
                  'dof=', result_min.dof)

            assert not np.isnan(result_min.pvalue[
                config_num]), "pvalue is nan"
            # use sloppy configs to check if fit will work
            if config_num in [0+SUPERJACK_CUTOFF, 1+SUPERJACK_CUTOFF]:

                # check if chi^2 (t^2) too big, too small
                if result_min_jack.fun > chisq_fiduc_cut or\
                   (SKIP_OVERFIT and result_min_jack.fun < \
                    chisq_fiduc_overfit_cut):
                    skip_votes.append(config_num)

                if config_num == 1+SUPERJACK_CUTOFF:
                    skip_range = False
                    # don't skip the fit range until we confirm
                    # on 2nd config
                    if len(skip_votes) == 2:
                        skip_range = True
                    elif len(skip_votes) == 1:
                        # approximate the difference as the stddev:
                        # sqrt(\sum(x-<x>)**2/(N));
                        # mult by sqrt(N-1) to get the variance in chi^2
                        # (t^2)
                        # if we have one bad fit and another which is within
                        # 5 sigma of the bad chi^2 (t^2),
                        # skip, else throw an error
                        skip_range = abs(result_min_jack.fun-chisq_min_arr[
                            0+SUPERJACK_CUTOFF])<5*np.sqrt(2*result_min.dof/(
                                params.num_configs-1))
                    if skip_range:
                        raise BadChisq(
                            chisq=result_min_jack.fun/result_min.dof,
                            dof=result_min.dof)
                    elif len(skip_votes) > 0:
                        # the second sample should never have a good fit
                        # if the first one has that bad a fit
                        print("fiducial cut =", chisq_fiduc_cut)
                        print("dof=", result_min.dof)
                        print(abs(result_min_jack.fun-chisq_min_arr[0]),
                              2*5*result_min.dof/np.sqrt(params.num_configs))
                        print(result_min_jack.fun, chisq_min_arr[0])
                        print("Bad jackknife distribution:"+\
                              str(chisq_min_arr[0]/result_min.dof)+" "+\
                              str(chisq_min_arr[1]/result_min.dof)+" "+\
                              str(result_min.pvalue[0])+" "+\
                              str(result_min.pvalue[1])+" ")
                        #sys.exit(1)
                        raise BadJackknifeDist

        # average results, compute jackknife uncertainties

        # pickle/unpickle the jackknifed arrays
        min_arr, result_min, chisq_min_arr = pickl(min_arr, result_min,
                                                   chisq_min_arr)

        # for title printing
        latfit.finalout.mkplot.NUM_CONFIGS = len(min_arr)

        # compute p-value jackknife uncertainty
        result_min.pvalue_arr = np.array(result_min.pvalue)
        result_min.pvalue, result_min.pvalue_err = jack_mean_err(
            result_min.pvalue)

        # store arrays for fit range averaging
        result_min.x_arr = np.array(min_arr)

        # compute the mean, error on the params
        result_min.x, param_err = jack_mean_err(min_arr)
        result_min.x_err = np.array(param_err)

        # compute the systematics and errors
        if SYS_ENERGY_GUESS is not None:
            result_min.systematics, result_min.systematics_err =\
                jack_mean_err(result_min.systematics_arr)

        # average the point by point error bars
        result_min.error_bars = em.acmean(result_min.error_bars, axis=0)


        # compute phase shift and error in phase shift
        if CALC_PHASE_SHIFT:
            phase_shift_data, scattering_length_data = phase_shift_scatter_len_avg(
                min_arr, result_min.phase_shift, params)

            result_min =  unpack_min_data(
                result_min, phase_shift_data, scattering_length_data)

        # compute mean, jackknife uncertainty of chi^2 (t^2)
        result_min.chisq_arr = np.array(chisq_min_arr) # remove this redundancy
        result_min.fun, result_min.chisq_err = jack_mean_err(chisq_min_arr)
        assert em.acmean(chisq_min_arr) == result_min.fun
        print(torchi(), result_min.fun/result_min.dof)

        return result_min, param_err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

@PROFILE
def toomanybadfitsp(result_min, chisq_min_arr):
    """If there have already been too many fits with large chi^2 (t^2),
    the average chi^2 (t^2) is not going to be good so abort the fit
    (test)
    """
    avg = em.acmean(chisq_min_arr)
    pvalue = result_min.funpvalue(avg)
    if pvalue < PVALUE_MIN:
        raise TooManyBadFitsError(chisq=avg, pvalue=pvalue)


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

@PROFILE
def chisqfiduc(num_configs, dof):
    """Find the chi^2/dof (t^2/dof) cutoff (acceptance upper bound)
    defined as > 5 \sigma away from an acceptable pvalue
    2*dof is the variance in chi^2 (t^2)
    """
    key = (num_configs, dof)
    t2correction = (num_configs-dof)/(num_configs-1)/dof
    cor = t2correction
    if key in chisqfiduc.mem:
        ret = chisqfiduc.mem[key]
    else:
        func = lambda tau: PVALUE_MIN*3e-7-(stats.f.sf(tau*cor, dof, num_configs-dof))
        func2 = lambda tau: PVALUE_MIN-(stats.f.sf(tau*cor, dof, num_configs-dof))
        # guess about 2 for the max chi^2/dof
        sol = float(fsolve(func, dof))
        sol2 = float(fsolve(func2, dof))
        assert abs(func(sol)) < 1e-8, "fsolve failed."
        assert abs(func2(sol2)) < 1e-8, "fsolve2 failed."
        # known variance of chi^2 is 2*dof, but skewed at low dof (here chosen to be < 10)
        # thus, the notion of a "5 sigma fluctuation" is only defined as dof->inf
        # so we have a factor of 2 to make low dof p-value cut less aggressive
        #ret = sol+5*(2 if dof < 10 else 1)*np.sqrt(2*dof)/(num_configs-SUPERJACK_CUTOFF)
        diff = (sol-sol2)/(num_configs-SUPERJACK_CUTOFF-1)
        ret = sol2+diff
        #print(ret/dof, sol/dof, num_configs, dof, PVALUE_MIN,
        #      1-stats.chi2.cdf(ret, dof), 1-stats.chi2.cdf(sol, dof))
        chisqfiduc.mem[key] = ret
    return ret
chisqfiduc.mem = {}

@PROFILE
def correction(min_arr, config_num):
    """Correct the jackknifed E_pipi"""
    if hasattr(DELTA_E_AROUND_THE_WORLD, '__iter__') and\
       np.asarray(DELTA_E_AROUND_THE_WORLD).shape:
        latw = len(DELTA_E_AROUND_THE_WORLD)
        assert latw == 1 or latw == len(min_arr), "bug:  array mismatch"
        corre1 = DELTA_E_AROUND_THE_WORLD[config_num] if latw > 1 else DELTA_E_AROUND_THE_WORLD[0]
    else:
        corre1 = DELTA_E_AROUND_THE_WORLD
    if hasattr(DELTA_E2_AROUND_THE_WORLD, '__iter__') and\
       np.asarray(DELTA_E2_AROUND_THE_WORLD).shape:
        corre2 = DELTA_E2_AROUND_THE_WORLD[config_num]
    else:
        corre2 = DELTA_E2_AROUND_THE_WORLD if\
            DELTA_E2_AROUND_THE_WORLD is not None else 0
    if FIT_SPACING_CORRECTION and not PIONRATIO:
        corre3 = misc.correct_epipi(min_arr[config_num],
                                    config_num=config_num)
    else:
        corre3 = 0
    ret = 0
    if GEVP:
        ret = corre1+corre2+corre3
    return ret


@PROFILE
def unnan_coords(coords):
    """replace nan's with 0 in coords"""
    for i in range(len(coords)):
        coords[i][1] = np.nan_to_num(coords[i][1]) 
    return coords


@PROFILE
def unpack_min_data(result_min, phase_shift_data, scattering_length_data):
    """Unpack the returned results of phase_shift_scatter_len_avg"""
    result_min.phase_shift,\
        result_min.phase_shift_err,\
        result_min.phase_shift_arr = phase_shift_data
    result_min.scattering_length,\
        result_min.scattering_length_err,\
        result_min.scattering_length_arr = scattering_length_data
    return result_min

@PROFILE
def getsystematic(params, arr):
    """Get the fit parameters which are not the energies"""
    arr = np.asarray(arr)
    params.energyind = None
    if len(arr) != params.dimops and len(arr) and EFF_MASS:
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
    for i,j in zip(sorted(list(ret)), ret):
        if i != j:
            print("mis-sorted energies:", ret)
            raise EnergySortError
    return ret

@PROFILE
def phase_shift_scatter_len_avg(min_arr, phase_shift, params):
    """Average the phase shift results, calc scattering length"""
    if not GEVP:
        try:
            min_arr = min_arr[:, 1]
        except IndexError:
            try:
                min_arr = min_arr[:, 0]
            except IndexError:
                sys.exit(1)

    # get rid of configs were phase shift calculation failed
    # (good for debug only)
    phase_shift = np.delete(phase_shift,
                            prune_phase_shift_arr(
                                phase_shift),
                            axis=0)

    if len(phase_shift) > 0:

        # calculate scattering length via energy, phase shift
        scattering_length = -1.0*np.tan(
            phase_shift)/np.sqrt(
                (min_arr**2/4-PION_MASS**2).astype(complex))

        scattering_length_arr = np.array(scattering_length)
        phase_shift_arr = np.array(phase_shift)

        # calc mean, err on phase shift and scattering length
        phase_shift, phase_shift_err = \
            jack_mean_err(phase_shift)
        scattering_length, scattering_length_err = \
            jack_mean_err(scattering_length)

    else:
        phase_shift = None
        phase_shift_err = None
        scattering_length = None
        scattering_length_err = None
    phase_shift_data = (phase_shift, phase_shift_err, phase_shift_arr)
    scattering_length_data = (scattering_length, scattering_length_err, scattering_length_arr)
    return phase_shift_data, scattering_length_data


@PROFILE
def alloc_phase_shift(params):
    """Get an empty array for Nconfig phase shifts"""
    nphase = 1 if not GEVP else params.dimops
    ret = np.zeros((
        params.num_configs, nphase), dtype=np.complex) if \
        params.dimops > 1 else np.zeros((
            params.num_configs), dtype=np.complex)
    return ret


@PROFILE
def jack_mean_err(arr, arr2=None, sjcut=SUPERJACK_CUTOFF, nosqrt=False):
    """Calculate error in arr over axis=0 via jackknife factor
    first n configs up to and including sjcut are exact
    the rest are sloppy.
    """
    len_total = len(arr)
    len_sloppy = len_total-sjcut
    arr2 = arr if arr2 is None else arr2

    if sjcut == 0:
        assert not sjcut, "sjcut bug"
    if not sjcut:
        assert sjcut == 0, "sjcut bug"

    # get jackknife correction prefactors
    exact_prefactor = (sjcut-1)/sjcut if sjcut else 0
    exact_prefactor_inv = sjcut/(sjcut-1) if sjcut else 0
    assert not np.isnan(exact_prefactor), "exact prefactor is nan"
    sloppy_prefactor = (len_sloppy-1)/len_sloppy
    assert not np.isnan(sloppy_prefactor), "sloppy prefactor is nan"
    overall_prefactor = (len_total-1)/len_total
    assert not np.isnan(overall_prefactor), "sloppy prefactor is nan"
    if not sjcut:
        assert overall_prefactor == sloppy_prefactor, "bug"
    assert arr.shape == arr2.shape, "Shape mismatch"

    # calculate error on exact and sloppy
    if sjcut:
        errexact = exact_prefactor*em.acsum(
            (arr[:sjcut]-em.acmean(arr[:sjcut], axis=0))*(
                arr2[:sjcut]-em.acmean(arr2[:sjcut], axis=0)),
            axis=0)
    else:
        errexact = 0
    if isinstance(errexact, numbers.Number):
        assert not np.isnan(errexact), "exact err is nan"
    else:
        assert not any(np.isnan(errexact)), "exact err is nan"

    if sjcut == 0:
        assert errexact == 0, "non-zero error in the non-existent"+\
            " exact samples"
    errsloppy = sloppy_prefactor*em.acsum(
        (arr[sjcut:]-em.acmean(arr[sjcut:], axis=0))*(
            arr2[sjcut:]-em.acmean(arr2[sjcut:], axis=0)),
        axis=0)
    if isinstance(errsloppy, numbers.Number):
        assert not np.isnan(errsloppy), "sloppy err is nan"
    else:
        assert not any(np.isnan(errsloppy)), "sloppy err is nan"

    # calculate the superjackknife errors
    # (redundant prefactor multiplies, but perhaps clearer)
    err = overall_prefactor*(errsloppy/sloppy_prefactor+\
                             errexact*exact_prefactor_inv)
    try:
        err = err if nosqrt else np.sqrt(err)
        flag = False
    except FloatingPointError:
        flag = True
    assert err.shape == np.array(arr)[0].shape, "Shape is not preserved (bug)."

    # calculate the mean
    mean = em.acmean(arr, axis=0)
    if isinstance(mean, numbers.Number):
        mean = float(mean)
        assert not np.isnan(mean), "mean is nan"
    else:
        assert not any(np.isnan(mean)), "mean is nan"
    if not flag:
        if isinstance(err, numbers.Number):
            err = float(err)
            assert not np.isnan(err), "err is nan"
        else:
            assert not any(np.isnan(err)), "err is nan"
    else:
        mean = np.nan
        err = np.nan

    return mean, err

@PROFILE
def pickl(min_arr, result_min, chisq_min_arr):
    """Pickle or unpickle the results from the jackknife fit loop
    to do: make more general use **kwargs
    """
    if PICKLE == 'pickle':
        pickle.dump(min_arr, open(unique_pickle_file("min_arr"), "wb"))
        pickle.dump(result_min.phase_shift, open(
            unique_pickle_file("phase_shift"), "wb"))
        pickle.dump(chisq_min_arr,
                    open(unique_pickle_file("chisq_min_arr"), "wb"))
    elif PICKLE == 'unpickle':
        _, rangei = unique_pickle_file("min_arr", True)
        _, rangej = unique_pickle_file("phase_shift", True)
        _, rangek = unique_pickle_file("chisq_min_arr", True)
        for i in range(rangei):
            min_arr /= (rangei+1)
            min_arr += 1.0/(rangei+1)*pickle.load(open(
                "min_arr"+str(i)+".p", "rb"))
        for j in range(rangej):
            result_min.phase_shift /= (rangej+1)
            result_min.phase_shift += 1.0/(
                rangej+1)*pickle.load(open(
                    "phase_shift"+str(j)+".p", "rb"))
        for k in range(rangek):
            chisq_min_arr /= (rangek+1)
            chisq_min_arr += 1.0/(rangek+1)*pickle.load(open(
                "chisq_min_arr"+str(k)+".p", "rb"))
    elif PICKLE is None:
        pass
    return min_arr, result_min, chisq_min_arr

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
        if params.dimops > 1:
            retlist = [zeta(epipi) for epipi in epipi_arr]
        else:
            retlist = zeta(epipi_arr)
    except ZetaError:
        retlist = None
        raise
    return retlist


@PROFILE
def copy_block_no_sidefx(params, blk):
    """Copy a jackknife block (for a particular config)
    for later possible modification"""
    if params.dimops > 1:
        print("copy_block not supported in this form")
        sys.exit(1)
        retblk = np.array([
            blk[time] for time in range(len(params.time_range))
        ])
    else:
        retblk = blk
    return retblk


@PROFILE
def copy_block(params, blk, out):
    """Copy a jackknife block (for a particular config)
    for later possible modification"""
    if params.dimops > 1:
        for time in range(len(params.time_range)):
            out[time, 1] = np.nan_to_num(blk[time])
    else:
        out[:, 1] = np.nan_to_num(blk)


@PROFILE
def alloc_errbar_arr(params, time_length):
    """Allocate an array. Each config gives us a jackknife fit,
    and a set of error bars.
    We store the error bars in this array, indexed
    by config.
    """
    if params.dimops > 1:
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
    if params.dimops == 1:  # i.e. if not using the GEVP
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


if CORRMATRIX:
    @PROFILE
    def invert_cov(covjack, params):
        """Invert the covariance matrix via correlation matrix
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params.dimops == 1:  # i.e. if not using the GEVP
            if UNCORR:
                covjack = np.diagflat(np.diag(covjack))
            covjack = dropimag(covjack)
            corrjack = np.zeros(covjack.shape)
            weightings = dropimag(np.sqrt(np.diag(covjack)))
            reweight = np.diagflat(1./weightings)
            corrjack = kdot(reweight, kdot(covjack, reweight))
            covinv_jack = kdot(kdot(reweight, inv(corrjack)), reweight)
        else:
            lent = len(covjack)  # time extent
            reweight = np.zeros((lent, params.dimops, lent, params.dimops))
            for i in range(lent):
                for j in range(params.dimops):
                    reweight[i][j][i][j] = 1.0/np.sqrt(covjack[i][j][i][j])
            corrjack = actensordot(
                actensordot(reweight, covjack), reweight)
            if UNCORR:
                diagcorr = np.zeros(corrjack.shape)
                for i in range(lent):
                    for j in range(params.dimops):
                        diagcorr[i][j][i][j] = corrjack[i][j][i][j]
                corrjack = diagcorr
            covinv_jack = swap(actensordot(reweight, actensordot(
                tensorinv(corrjack), reweight)), 1, 2)
        return covinv_jack

else:
    @PROFILE
    def invert_cov(covjack, params):
        """Invert the covariance matrix,
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params.dimops == 1:  # i.e. if not using the GEVP
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
        return covinv_jack * ((nsize)*(nsize-1))

    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (single jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        return covjack / ((nsize)*(nsize-1))

elif JACKKNIFE_FIT == 'DOUBLE':
    @PROFILE
    def normalize_covinv(covinv_jack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the covinv (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        return covinv_jack * ((nsize)/(nsize-1))

    @PROFILE
    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        return covjack / ((nsize)/(nsize-1))


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



@PROFILE
def get_doublejk_data(params, coords_jack, reuse, reuse_blocked, config_num):
    """Primarily, get the inverse covariance matrix for the particular
    double jackknife fit we are on (=config_num)
    reuse_inv is the original unjackknifed data
    coords_jack are the coordinates, we also return truncated coordinates
    if the fit fails.
    """

    # original data, obtained by reversing single jackknife procedure
    reuse_inv = inverse_jk(reuse, params.num_configs)

    # bootstrap ensemble
    reuse_blocked = bootstrap_ensemble(reuse_inv, reuse_blocked)

    # delete a block of configs
    reuse_inv_red = delblock(config_num, reuse_inv)

    flag = 2
    while flag > 0:
        if flag == 1:
            if len(coords_jack) == 1:
                print("Continuation failed.")
                sys.exit(1)
            assert None, "Not supported."
            coords_jack = coords_jack[1::2]
            cov_factor = np.delete(
                getcovfactor(params, reuse_blocked, config_num, reuse_inv_red),
                np.s_[::2], axis=1)
        else:
            cov_factor = getcovfactor(params, reuse_blocked, config_num, reuse_inv_red)
        covjack = get_covjack(cov_factor, params)
        covinv_jack_pruned, flag = prune_covjack(params, covjack,
                                                 coords_jack, flag)
    covinv_jack = prune_fit_range(covinv_jack_pruned, coords_jack)
    return coords_jack, covinv_jack, jack_errorbars(covjack, params)

@PROFILE
def prune_covjack(params, covjack, coords_jack, flag):
    """Prune the covariance matrix based on config excluded time slices"""
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
    covinv_jack = np.zeros(covjack.shape, dtype=float)
    if params.dimops == 1:
        covinv_jack = np.copy(marray.data)
    else:
        # fill in the pruned dimensions with 0's
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                covinv_jack[opa, opb, :, :] = marray.data[
                    opa*len_time:(opa+1)*len_time, opb*len_time:(
                        opb+1)*len_time]
        # to put things back in time, time, dimops, dimops basis
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-1, 0)
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-2, 1)
    return covinv_jack, flag

@PROFILE
def invertmasked(params, len_time, excl, covjack):
    """invert the covariance matrix with pruned operator basis and fit range"""
    dim = int(np.sqrt(np.prod(list(covjack.shape))))
    matrix = np.zeros((dim, dim))
    if params.dimops == 1:
        matrix = np.copy(covjack)
    else:
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                matrix[opa*len_time:(opa+1)*len_time,
                       opb*len_time:(opb+1)*len_time] = covjack[opa, opb, :, :]
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
        matrix = invert_cov(matrix, params2)
        marray[~marray.mask] = normalize_covinv(matrix, params2).reshape(-1)
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
        else:
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

class TooManyBadFitsError(Exception):
    """Error if too many jackknifed fits have a large chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, pvalue=None, message=''):
        print("***ERROR***")
        if UNCORR:
            print("Too many fits have bad chi^2")
            print("chi^2 average up to this point:", chisq)
        else:
            print("Too many fits have bad t^2")
            print("t^2 average up to this point:", chisq)
        print("pvalue up to this point:", pvalue)
        super(TooManyBadFitsError, self).__init__(message)
        self.message = message

class EnergySortError(Exception):
    """If the energies are not sorted in ascending order
    (if the systematic errors are large)
    """
    @PROFILE
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        print("Energies are not sorted in ascending order")
        super(EnergySortError, self).__init__(message)
        self.message = message

class BadJackknifeDist(Exception):
    """Exception for bad jackknife distribution"""
    @PROFILE
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        if UNCORR:
            print("Bad jackknife distribution, variance in chi^2 too large")
        else:
            print("Bad jackknife distribution, variance in t^2 too large")
        super(BadJackknifeDist, self).__init__(message)
        self.message = message

class NoConvergence(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        print("Minimizer failed to converge")
        super(NoConvergence, self).__init__(message)
        self.message = message

class DOFNonPos(Exception):
    """Exception for dof < 0"""
    @PROFILE
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        print("dof < 1: dof=", dof)
        print("FIT_EXCL=", latfit.config.FIT_EXCL)
        super(DOFNonPos, self).__init__(message)
        self.dof = dof
        self.message = message

class BadChisq(Exception):
    """Exception for bad chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, message='', dof=None):
        print("***ERROR***")
        if UNCORR:
            print("chisq/dof >> 1 or p-value >> 0.5 chi^2/dof =",
                  chisq, "dof =", dof)
        else:
            print("t^2/dof >> 1 or p-value >> 0.5 t^2/dof =",
                  chisq, "dof =", dof)
        super(BadChisq, self).__init__(message)
        self.chisq = chisq
        self.dof = dof
        self.message = message


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
        if latfit.config.BOOTSTRAP:
            ret = reuse_blocked-em.acmean(reuse_blocked, axis=0)
        else:
            num_configs_reduced = (params.num_configs-1)*bsize
            ret = np.array([
                em.acmean(np.delete(reuse_inv_red, i, axis=0), axis=0)
                for i in range(num_configs_reduced)]) - reuse_blocked[config_num]
        return ret
