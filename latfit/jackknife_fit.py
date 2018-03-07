"""Fit under a jackknife"""
import sys
from collections import namedtuple
import numpy as np
from numpy import swapaxes as swap
from numpy.linalg import inv, tensorinv
from scipy import stats

from latfit.extract.inverse_jk import inverse_jk
from latfit.makemin.mkmin import mkmin

from latfit.config import START_PARAMS
from latfit.config import JACKKNIFE_FIT
from latfit.config import CORRMATRIX
from latfit.config import GEVP
from latfit.config import CALC_PHASE_SHIFT, PION_MASS
from latfit.utilities.zeta.zeta import zeta, ZetaError

if JACKKNIFE_FIT == 'FROZEN':
    def jackknife_fit(params, reuse, coords, covinv):
        """Fit under a frozen (single) jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        """
        result_min = namedtuple(
            'min', ['x', 'fun', 'status', 'err_in_chisq', 'dof'])
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
            print("config", config_num, ":",
                  result_min_jack.x, "chisq/dof=",
                  result_min_jack.fun/result_min.dof)
            chisq_min_arr[config_num] = result_min_jack.fun
            min_arr[config_num] = result_min_jack.x
        result_min.x = np.mean(min_arr, axis=0)
        param_err = np.sqrt(params.prefactor*np.sum(
            (min_arr-result_min.x)**2, 0))
        result_min.fun = np.mean(chisq_min_arr)
        result_min.err_in_chisq = np.sqrt(params.prefactor*np.sum(
            (chisq_min_arr-result_min.fun)**2))
        return result_min, param_err


elif JACKKNIFE_FIT == 'DOUBLE' or JACKKNIFE_FIT == 'SINGLE':
    def jackknife_fit(params, reuse, coords, covinv=None):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        """
        if covinv is None:
            pass

        # storage for results
        result_min = namedtuple('min',
                                ['x', 'fun', 'status',
                                 'pvalue', 'pvalue_err' 'err_in_chisq',
                                 'error_bars', 'dof', 'phase_shift',
                                 'phase_shift_err', 'scattering_length',
                                 'scattering_length_err'])

        # no errors gives us 0 status
        result_min.status = 0

        # compute degrees of freedom
        result_min.dof = len(coords)*params.dimops-len(START_PARAMS)

        # alloc storage
        # one fit for every jackknife block (N fits for N configs)

        # fit by fit p-values
        result_min.pvalue = np.zeros(params.num_configs)

        #phase shift
        #nphase = 1 if not GEVP else params.dimops-1
        nphase = 1 if not GEVP else params.dimops
        result_min.phase_shift = np.zeros((
            params.num_configs, nphase), dtype=np.complex) if \
            params.dimops > 1 else np.zeros((
                params.num_configs), dtype=np.complex)

        # allocate storage for jackknifed x,y coordinates
        coords_jack = np.copy(coords)

        # storage for fit by fit optimized fit params
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)))

        # storage for fit by fit chi^2
        chisq_min_arr = np.zeros(params.num_configs)

        # original data, obtained by reversing single jackknife procedure
        reuse_inv = inverse_jk(reuse, params.time_range, params.num_configs)

        # fit by fit error bars (we eventually plot the averaged set)
        result_min.error_bars = alloc_errbar_arr(params, len(coords))

        # loop over configs, doing a fit for each one
        for config_num in range(params.num_configs):

            # if config_num>160: break # for debugging only

            # copy the jackknife block into coords_jack
            copy_block(params, reuse[config_num], coords_jack)

            # get the data for the minimizer, and the error bars
            coords_jack, covinv_jack, result_min.error_bars[
                config_num] = get_doublejk_data(params, coords_jack,
                                                reuse, config_num,
                                                reuse_inv)

            # minimize chi^2 given the inverse covariance matrix and data
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status

            # store results for this fit
            chisq_min_arr[config_num] = result_min_jack.fun
            min_arr[config_num] = result_min_jack.x

            # compute phase shift, if necessary
            result_min.phase_shift[config_num] = phase_shift_jk(
                params, result_min_jack.x)

            # compute p value for this fit
            result_min.pvalue[config_num] = 1 - stats.chi2.cdf(
                result_min_jack.fun, result_min.dof)

            # print results for this config
            print("config", config_num, ":", result_min_jack.x,
                  "chisq/dof=", result_min_jack.fun/result_min.dof,
                  "p-value=", result_min.pvalue[config_num])

        # average results, compute jackknife uncertainties

        # compute p-value jackknife uncertainty
        result_min.pvalue_err = np.sqrt(params.prefactor*np.sum((
            result_min.pvalue-np.mean(result_min.pvalue))**2))
        # average the p-value
        result_min.pvalue = np.mean(result_min.pvalue)

        # average the point by point error bars
        result_min.error_bars = np.mean(result_min.error_bars, axis=0)

        # compute the average fit params
        result_min.x = np.mean(min_arr, axis=0)

        # compute phase shift and error in phase shift
        if CALC_PHASE_SHIFT:
            result_min.phase_shift = np.delete(result_min.phase_shift,
                                               prune_phase_shift_arr(
                                                   result_min.phase_shift),
                                               axis=0)
            if len(result_min.phase_shift) > 0:
                result_min.scattering_length = -1.0*np.tan(
                    result_min.phase_shift)/np.sqrt(
                        (min_arr**2/4-PION_MASS**2).astype(complex))
                result_min.phase_shift_err = np.sqrt(
                    params.prefactor*np.sum((
                        result_min.phase_shift-np.mean(
                        result_min.phase_shift, axis=0))**2, axis=0))
                result_min.scattering_length_err = np.sqrt(
                    params.prefactor*np.sum((
                        result_min.scattering_length-np.mean(
                        result_min.scattering_length, axis=0))**2, axis=0))
                result_min.phase_shift = np.mean(
                    result_min.phase_shift, axis=0)
                result_min.scattering_length = np.mean(
                    result_min.scattering_length, axis=0)
            else:
                result_min.phase_shift = None

        # compute the error on the params
        param_err = np.sqrt(params.prefactor*np.sum(
            (min_arr-result_min.x)**2, 0))

        # compute average chi^2
        result_min.fun = np.mean(chisq_min_arr)

        # compute jackknife uncertainty on chi^2
        result_min.err_in_chisq = np.sqrt(params.prefactor*np.sum(
            (chisq_min_arr-result_min.fun)**2))

        return result_min, param_err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

def prune_phase_shift_arr(arr):
    """Get rid of jackknife samples for which the phase shift calc failed.
    (useful for testing, not useful for final output graphs)
    """
    dellist = []
    for i, phi in enumerate(arr):
        if np.isnan(np.sum(phi)):  # delete the config
            print("Bad phase shift in jackknife block # "+
                    str(i)+", omitting.")
            dellist.append(i)
    return dellist

def phase_shift_jk(params, epipi_arr):
    """Compute the nth jackknifed phase shift"""
    try:
        if params.dimops > 1:
            retlist = [zeta(epipi) for epipi in epipi_arr]
        else:
            retlist = zeta(epipi_arr)
    except ZetaError:
        retlist = None
    return retlist

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


def copy_block(params, blk, out):
    """Copy a jackknife block (for a particular config)
    for later possible modification"""
    if params.dimops > 1:
        for time in range(len(params.time_range)):
            out[time, 1] = blk[time]
    else:
        out[:, 1] = blk


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


if CORRMATRIX:
    def invert_cov(covjack, params):
        """Invert the covariance matrix via correlation matrix"""
        if params.dimops == 1:  # i.e. if not using the GEVP
            corrjack = np.zeros(covjack.shape)
            weightings = np.sqrt(np.diag(covjack))
            reweight = np.diagflat(1./weightings)
            np.dot(reweight, np.dot(covjack, reweight), out=corrjack)
            covinv_jack = np.dot(np.dot(reweight, inv(corrjack)), reweight)
        else:
            lent = len(covjack)  # time extent
            reweight = np.zeros((lent, params.dimops, lent, params.dimops))
            for i in range(lent):
                for j in range(params.dimops):
                    reweight[i][j][i][j] = 1.0/np.sqrt(covjack[i][j][i][j])
            corrjack = np.tensordot(np.tensordot(reweight, covjack), reweight)
            covinv_jack = swap(np.tensordot(reweight, np.tensordot(
                tensorinv(corrjack), reweight)), 1, 2)
        return covinv_jack

else:
    def invert_cov(covjack, params):
        """Invert the covariance matrix"""
        if params.dimops == 1:  # i.e. if not using the GEVP
            covinv_jack = inv(covjack)
        else:
            covinv_jack = swap(tensorinv(covjack), 1, 2)
        return covinv_jack

if JACKKNIFE_FIT == 'SINGLE':
    def normalize_covinv(covinv_jack, params):
        """do the proper normalization of the covinv (single jk)"""
        return covinv_jack * ((params.num_configs-1)*(params.num_configs-2))

    def normalize_cov(covjack, params):
        """do the proper normalization of the
        covariance matrix (single jk)"""
        return covjack / ((params.num_configs-1)*(params.num_configs-2))

elif JACKKNIFE_FIT == 'DOUBLE':
    def normalize_covinv(covinv_jack, params):
        """do the proper normalization of the covinv (double jk)"""
        return covinv_jack * ((params.num_configs-1)/(params.num_configs-2))

    def normalize_cov(covjack, params):
        """do the proper normalization of the
        covariance matrix (double jk)"""
        return covjack / ((params.num_configs-1)/(params.num_configs-2))


def get_doublejk_data(params, coords_jack, reuse, config_num, reuse_inv):
    """Primarily, get the inverse covariance matrix for the particular
    double jackknife fit we are on (=config_num)
    reuse_inv is the original unjackknifed data
    coords_jack are the coordinates, we also return truncated coordinates
    if the fit fails.
    """
    flag = 2
    while flag > 0:
        if flag == 1:
            if len(coords_jack) == 1:
                print("Continuation failed.")
                sys.exit(1)
            coords_jack = coords_jack[1::2]
            cov_factor = np.delete(
                getcovfactor(params, reuse, config_num, reuse_inv),
                np.s_[::2], axis=1)
        else:
            cov_factor = getcovfactor(params, reuse, config_num, reuse_inv)
        covjack = get_covjack(cov_factor, params)
        try:
            covinv_jack = invert_cov(covjack, params)
            covinv_jack = normalize_covinv(covinv_jack, params)
            flag = 0
        except np.linalg.linalg.LinAlgError as err:
            if str(err) == 'Singular matrix':
                print("Covariance matrix is singular",
                      "in jackknife fit.")
                print("Failing config_num=", config_num)
                print("Attempting to continue",
                      "fit with every other time slice",
                      "eliminated.")
                print("Plotted error bars should be " +
                      "considered suspect.")
                flag = 1
            else:
                raise
    return coords_jack, covinv_jack, jack_errorbars(covjack, params)


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
    def getcovfactor(params, reuse, config_num, reuse_inv):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        block == reuse
        SINGLE elimination jackknife
        """
        if params:
            pass
        return np.delete(reuse_inv, config_num, axis=0)-reuse[config_num]

elif JACKKNIFE_FIT == 'DOUBLE':
    def getcovfactor(params, reuse, config_num, reuse_inv):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        block == reuse
        DOUBLE elimination jackknife
        """
        return np.array([
            np.mean(np.delete(np.delete(
                reuse_inv, config_num,
                axis=0), i, axis=0), axis=0)
            for i in range(params.num_configs-1)]) - reuse[config_num]
