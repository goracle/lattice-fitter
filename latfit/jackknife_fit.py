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

from latfit.extract.inverse_jk import inverse_jk
from latfit.makemin.mkmin import mkmin

from latfit.config import START_PARAMS
from latfit.config import JACKKNIFE_FIT
from latfit.config import CORRMATRIX
from latfit.config import GEVP
from latfit.config import UNCORR
from latfit.config import PICKLE
from latfit.config import CALC_PHASE_SHIFT, PION_MASS
from latfit.config import SUPERJACK_CUTOFF
from latfit.config import DELTA_E_AROUND_THE_WORLD
from latfit.utilities.zeta.zeta import zeta, ZetaError
import latfit.finalout.mkplot
import latfit.config

if JACKKNIFE_FIT == 'FROZEN':
    def jackknife_fit(params, reuse, coords, covinv):
        """Fit under a frozen (single) jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        """
        assert 0, "not currently supported"
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
        for i in coords[:, 0]:
            for j in latfit.config.FIT_EXCL:
                if i in j:
                    result_min.dof -= 1
        if result_min.dof < 1:
            raise DOFNonPos(dof=result_min.dof)

        # alloc storage
        # one fit for every jackknife block (N fits for N configs)

        # fit by fit p-values
        result_min.pvalue = np.zeros(params.num_configs)

        #phase shift
        result_min.phase_shift = alloc_phase_shift(params)

        # allocate storage for jackknifed x,y coordinates
        coords_jack = np.copy(coords)

        # storage for fit by fit optimized fit params
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)))

        # storage for fit by fit chi^2
        chisq_min_arr = np.zeros(params.num_configs)


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
                                                reuse, config_num)

            # minimize chi^2 given the inverse covariance matrix and data
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status

            # store results for this fit
            chisq_min_arr[config_num] = result_min_jack.fun

            # we shifted the GEVP energy spectrum down
            # to fix the leading order around the world term so shift it back

            result_min_jack.x = np.asarray(result_min_jack.x)+\
                DELTA_E_AROUND_THE_WORLD

            # store the result
            min_arr[config_num] = result_min_jack.x

            # compute phase shift, if necessary
            if CALC_PHASE_SHIFT:
                result_min.phase_shift[config_num] = phase_shift_jk(
                    params, result_min_jack.x)

            # compute p value for this fit
            result_min.pvalue[config_num] = 1 - stats.chi2.cdf(
                result_min_jack.fun, result_min.dof)

            # print results for this config
            print("config", config_num, ":", result_min_jack.x,
                  "chisq/dof=", result_min_jack.fun/result_min.dof,
                  "p-value=", result_min.pvalue[config_num])

            if result_min_jack.fun/result_min.dof > 5:
                raise BadChisqJackknife(result_min_jack.fun/result_min.dof)

        # average results, compute jackknife uncertainties

        # pickle/unpickle the jackknifed arrays
        min_arr, result_min, chisq_min_arr = pickl(min_arr, result_min, chisq_min_arr)

        # for title printing
        latfit.finalout.mkplot.NUM_CONFIGS = len(min_arr)

        # compute p-value jackknife uncertainty
        result_min.pvalue, result_min.pvalue_err = jack_mean_err(result_min.pvalue)

        # compute the mean, error on the params
        result_min.x, param_err = jack_mean_err(min_arr)

        # average the point by point error bars
        result_min.error_bars = np.mean(result_min.error_bars, axis=0)


        # compute phase shift and error in phase shift
        if CALC_PHASE_SHIFT:

            min_arr, result_min.phase_shift, \
                result_min.phase_shift_err, \
                result_min.scattering_length, \
                result_min.scattering_length_err = phase_shift_scatter_len_avg(min_arr,
                                                                               result_min.phase_shift)

        # compute mean, jackknife uncertainty of chi^2
        result_min.fun, result_min.err_in_chisq = jack_mean_err(chisq_min_arr)

        return result_min, param_err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)


def phase_shift_scatter_len_avg(min_arr, phase_shift):
    """Average the phase shift results, calc scattering length"""
    if not GEVP:
        try:
            min_arr = min_arr[:, 1]
        except IndexError:
            try:
                min_arr = min_arr[:, 0]
            except IndexError:
                raise

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
    return min_arr, phase_shift, phase_shift_err, scattering_length, scattering_length_err


def alloc_phase_shift(params):
    """Get an empty array for Nconfig phase shifts"""
    nphase = 1 if not GEVP else params.dimops
    ret = np.zeros((
        params.num_configs, nphase), dtype=np.complex) if \
        params.dimops > 1 else np.zeros((
            params.num_configs), dtype=np.complex)
    return ret


def jack_mean_err(arr, sjcut=SUPERJACK_CUTOFF):
    """Calculate error in arr over axis=0 via jackknife factor
    first n configs up to and including sjcut are exact
    the rest are sloppy.
    """
    len_total = len(arr)
    len_sloppy = len_total-sjcut

    # get jackknife correction prefactors
    exact_prefactor = (sjcut-1)/sjcut if sjcut else 0
    sloppy_prefactor = (len_sloppy-1)/(len_sloppy)

    # calculate error on exact and sloppy
    if sjcut:
        errexact = np.sqrt(exact_prefactor*np.sum(
            (arr[:sjcut]-np.mean(arr[:sjcut], axis=0))**2, axis=0))
    else:
        errexact = 0
    errsloppy = np.sqrt(sloppy_prefactor*np.sum(
        (arr[sjcut:]-np.mean(arr[sjcut:], axis=0))**2, axis=0))

    # add errors in quadrature (assumes errors are small,
    # decorrelated, linear approx)
    err = np.sqrt(errsloppy**2+errexact**2)/2

    # calculate the mean
    mean = np.mean(arr, axis=0)
    if isinstance(mean, numbers.Number):
        mean = float(mean)
    if isinstance(err, numbers.Number):
        err = float(err)

    return mean, err

def pickl(min_arr, result_min, chisq_min_arr):
    """Pickle or unpickle the results from the jackknife fit loop
    to do: make more general use **kwargs
    """
    if PICKLE == 'pickle':
        pickle.dump(min_arr, open(unique_pickle_file("min_arr"), "wb"))
        pickle.dump(result_min.phase_shift, open(unique_pickle_file("phase_shift"), "wb"))
        pickle.dump(chisq_min_arr, open(unique_pickle_file("chisq_min_arr"), "wb"))
    elif PICKLE == 'unpickle':
        _, rangei = unique_pickle_file("min_arr", True)
        _, rangej = unique_pickle_file("phase_shift", True)
        _, rangek = unique_pickle_file("chisq_min_arr", True)
        for i in range(rangei):
            min_arr /= (rangei+1)
            min_arr += 1.0/(rangei+1)*pickle.load(open("min_arr"+str(i)+".p", "rb"))
        for j in range(rangej):
            result_min.phase_shift /= (rangej+1)
            result_min.phase_shift += 1.0/(
                rangej+1)*pickle.load(open("phase_shift"+str(j)+".p", "rb"))
        for k in range(rangek):
            chisq_min_arr /= (rangek+1)
            chisq_min_arr += 1.0/(rangek+1)*pickle.load(open("chisq_min_arr"+str(k)+".p", "rb"))
    elif PICKLE is None:
        pass
    return min_arr, result_min, chisq_min_arr

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


def prune_fit_range(covinv_jack, coords_jack):
    """Zero out parts of the inverse covariance matrix to exclude items
    from fit range.  Thus, the contribution to chi^2 will be 0.
    """
    excl = latfit.config.FIT_EXCL
    for i, xcoord in enumerate(coords_jack[:, 0]):
        for opa, _ in enumerate(excl):
            for j, _ in enumerate(excl[opa]):
                if xcoord == excl[opa][j]:
                    assert covinv_jack[i, :, opa, :].all() == 0, "Prune failed."
                    assert covinv_jack[:, i, opa, :].all() == 0, "Prune failed."
                    assert covinv_jack[:, i, :, opa].all() == 0, "Prune failed."
                    assert covinv_jack[i, :, :, opa].all() == 0, "Prune failed."
    return covinv_jack

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
            raise ZetaError("bad phase shift (nan)") # remove this if debugging
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
        """Invert the covariance matrix via correlation matrix
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params.dimops == 1:  # i.e. if not using the GEVP
            if UNCORR:
                covjack = np.diagflat(np.diag(covjack))
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
            if UNCORR:
                diagcorr = np.zeros(corrjack.shape)
                for i in range(lent):
                    for j in range(params.dimops):
                        diagcorr[i][j][i][j] = corrjack[i][j][i][j]
                corrjack = diagcorr
            covinv_jack = swap(np.tensordot(reweight, np.tensordot(
                tensorinv(corrjack), reweight)), 1, 2)
        return covinv_jack

else:
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


def get_doublejk_data(params, coords_jack, reuse, config_num):
    """Primarily, get the inverse covariance matrix for the particular
    double jackknife fit we are on (=config_num)
    reuse_inv is the original unjackknifed data
    coords_jack are the coordinates, we also return truncated coordinates
    if the fit fails.
    """

    # original data, obtained by reversing single jackknife procedure
    reuse_inv = inverse_jk(reuse, params.time_range, params.num_configs)

    flag = 2
    while flag > 0:
        if flag == 1:
            if len(coords_jack) == 1:
                print("Continuation failed.")
                sys.exit(1)
            assert None, "Not supported."
            coords_jack = coords_jack[1::2]
            cov_factor = np.delete(
                getcovfactor(params, reuse, config_num, reuse_inv),
                np.s_[::2], axis=1)
        else:
            cov_factor = getcovfactor(params, reuse, config_num, reuse_inv)
        covjack = get_covjack(cov_factor, params)
        covinv_jack_pruned, flag = prune_covjack(params, covjack,
                                                 coords_jack, flag)
    covinv_jack = prune_fit_range(covinv_jack_pruned, coords_jack)
    return coords_jack, covinv_jack, jack_errorbars(covjack, params)

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
                covinv_jack[opa, opb, :, :] = marray.data[opa*len_time:(opa+1)*len_time,
                                                          opb*len_time:(opb+1)*len_time]
        # to put things back in time, time, dimops, dimops basis
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-1, 0)
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-2, 1)
    return covinv_jack, flag

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

class DOFNonPos(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        print("dof < 1: dof=", dof)
        super(DOFNonPos, self).__init__(message)
        self.dof = dof
        self.message = message

class BadChisqJackknife(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, chisq=None, message=''):
        print("***ERROR***")
        print("chisq/dof > 5 chisq=", chisq)
        super(BadChisqJackknife, self).__init__(message)
        self.dof = chisq
        self.message = message


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
