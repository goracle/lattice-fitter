"""Fit under a jackknife"""
import sys
from collections import namedtuple
import numpy as np
from numpy import swapaxes as swap
from numpy.linalg import inv, tensorinv

from latfit.extract.inverse_jk import inverse_jk
from latfit.makemin.mkmin import mkmin

from latfit.config import START_PARAMS
from latfit.config import JACKKNIFE_FIT

if JACKKNIFE_FIT == 'FROZEN':
    def jackknife_fit(params, reuse, coords, time_range, covinv):
        """Fit under a frozen (single) jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        """
        params = namedtuple('fit_params', [
            'dimops', 'num_configs', 'prefactor'])
        result_min = namedtuple(
            'min', ['x', 'fun', 'status', 'err_in_chisq'])
        result_min.status = 0
        #one fit for every jackknife block (N fits for N configs)
        coords_jack = np.copy(coords)
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)))
        chisq_min_arr = np.zeros(params.num_configs)
        covinv_jack = covinv
        for config_num in range(params.num_configs):
            if params.dimops > 1:
                for time in range(len(time_range)):
                    coords_jack[time, 1] = reuse[config_num][time]
            else:
                coords_jack[:, 1] = reuse[config_num]
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status
            print("config", config_num, ":",
                  result_min_jack.x, "chisq=", result_min_jack.fun)
            chisq_min_arr[config_num] = result_min_jack.fun
            min_arr[config_num] = result_min_jack.x
        result_min.x = np.mean(min_arr, axis=0)
        param_err = np.sqrt(params.prefactor*np.sum((min_arr-result_min.x)**2, 0))
        result_min.fun = np.mean(chisq_min_arr)
        result_min.err_in_chisq = np.sqrt(params.prefactor*np.sum(
            (chisq_min_arr-result_min.fun)**2))
        return result_min, param_err

elif JACKKNIFE_FIT == 'DOUBLE' or JACKKNIFE_FIT == 'SINGLE':
    def jackknife_fit(params, reuse, coords, time_range, covinv=None):
        """Fit under a double jackknife.
        returns the result_min which has the minimized params ('x'),
        jackknife avg value of chi^2 ('fun') and error in chi^2
        and whether the minimizer succeeded on all fits
        ('status' == 0 if all were successful)
        """
        if covinv is None:
            pass
        result_min = namedtuple('min',
                                ['x', 'fun', 'status', 'err_in_chisq'])
        result_min.status = 0
        #one fit for every jackknife block (N fits for N configs)
        coords_jack = np.copy(coords)
        min_arr = np.zeros((params.num_configs, len(START_PARAMS)))
        chisq_min_arr = np.zeros(params.num_configs)
        reuse_inv = inverse_jk(reuse, time_range, params.num_configs)
        for config_num in range(params.num_configs):
            #if config_num>160: break #for debugging only
            if params.dimops > 1:
                for time in range(len(time_range)):
                    coords_jack[time, 1] = reuse[config_num][time]
            else:
                coords_jack[:, 1] = reuse[config_num]
            coords_jack, covinv_jack = get_doublejk_data(
                params, coords_jack, reuse, config_num, reuse_inv)
            result_min_jack = mkmin(covinv_jack, coords_jack)
            if result_min_jack.status != 0:
                result_min.status = result_min_jack.status
            print("config", config_num, ":", result_min_jack.x,
                  "chisq=", result_min_jack.fun)
            chisq_min_arr[config_num] = result_min_jack.fun
            min_arr[config_num] = result_min_jack.x
        result_min.x = np.mean(min_arr, axis=0)
        param_err = np.sqrt(params.prefactor*np.sum((min_arr-result_min.x)**2, 0))
        result_min.fun = np.mean(chisq_min_arr)
        result_min.err_in_chisq = np.sqrt(params.prefactor*np.sum(
            (chisq_min_arr-result_min.fun)**2))
        return result_min, param_err
else:
    print("***ERROR***")
    print("Bad jackknife_fit value specified.")
    sys.exit(1)

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
            cov_factor = np.delete(getcovfactor(params, reuse, config_num, reuse_inv), np.s_[::2], axis=1)
        else:
            cov_factor = getcovfactor(params, reuse, config_num, reuse_inv)
        try:
            if params.dimops == 1:
                covinv_jack = inv(np.einsum(
                    'ai,aj->ij', cov_factor, cov_factor))
            else:
                covinv_jack = swap(
                    tensorinv(np.einsum('aim,ajn->imjn', cov_factor,
                                        cov_factor)), 1, 2)
            #do the proper normalization of the covinv
            if JACKKNIFE_FIT == 'SINGLE':
                covinv_jack *= ((params.num_configs-1)*(params.num_configs-2))
            elif JACKKNIFE_FIT == 'DOUBLE':
                covinv_jack *= ((params.num_configs-1)/(params.num_configs-2))
            flag = 0
        except np.linalg.linalg.LinAlgError as err:
            if str(err) == 'Singular matrix':
                print("Covariance matrix is singular",
                      "in jackknife fit.")
                print("Failing config_num=", config_num)
                print("Attempting to continue",
                      "fit with every other time slice",
                      "eliminated.")
                flag = 1
            else:
                raise
    return coords_jack, covinv_jack

if JACKKNIFE_FIT == 'SINGLE':
    def getcovfactor(params, reuse, config_num, reuse_inv):
        """Get the factor which will be squared 
        when computing jackknife covariance matrix 
        (for this config number == config)
        inverse block == reuse_inv
        block == reuse
        SINGLE elimination jackknife
        """
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
        return np.mean([np.delete(np.delete(reuse_inv, config_num, axis=0), i, axis=0)
                    for i in range(params.num_configs-1)], axis=0)- reuse[config_num]
