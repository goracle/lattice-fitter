"""Calculates chi^2 (t^2)"""
import sys
from math import exp
from numpy import dot
import numpy as np

from latfit.config import fit_func
from latfit.config import GEVP, START_PARAMS, SYSTEMATIC_EST
from latfit.config import SYS_ENERGY_GUESS

if SYSTEMATIC_EST and False:

    LSTART = len(START_PARAMS)
    RANGE_START = range(LSTART)
    def fit_func_systematic(ctime, trial_params):
        """ansatz to estimate systematic errors"""
        return [fit_func(ctime, trial_params[:LSTART])[i]+
                trial_params[LSTART+2*i]*exp(-(trial_params[
                        LSTART+(2*i+1)]-trial_params[i]*0)*ctime)
                for i in RANGE_START]

else:
    fit_func_systematic = fit_func

RCORD = None
SYMRANGE = None
COUNT = None
PRODRANGE = None

if GEVP:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 (or, more likely, Hotelling's t^2)
        given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        # testing
        # print("break")
        # print(covinv)
        # print("break 2")
        # print(coords[0][1]-fit_func(coords[0][0], trial_params))
        retval = np.sum(
            np.fromiter((np.linalg.multi_dot([
            (coords[outer][1] - fit_func_systematic(
                coords[outer][0], trial_params)),
            covinv[outer][inner], (
                coords[inner][1]-fit_func_systematic(
                    coords[inner][0], trial_params))])
                         for outer, inner in SYMRANGE),
                        dtype=np.float, count=COUNT))
        retval *= 2
        if retval.imag != 0 and not np.isnan(retval.imag):
            llll = [dot(dot((
                coords[outer][1] - fit_func_systematic(
                    coords[outer][0], trial_params)),
                            covinv[outer][inner]),
                        (coords[inner][1]-fit_func_systematic(
                            coords[inner][0], trial_params)))
                    for outer in range(len(coords))
                    for inner in range(len(coords))]
            print("***ERROR***")
            print("imaginary part of chi^2 (t^2) is non-zero")
            print(coords[0][0])
            print("sep")
            print(coords[0][1])
            print("sep1")
            print("trial_params:", trial_params)
            print((coords[0][1]-fit_func_systematic(
                coords[0][0][0], trial_params)))
            print("sep2")
            print(covinv[0][0])
            print("sep3")
            print(dot((coords[0][1] - fit_func_systematic(
                coords[0][0], trial_params)), covinv[0][0]))
            print("sep4")
            print(llll)
            print("sep5")
            print(np.sum(llll))
            sys.exit(1)
        return retval.real
else:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 (t^2) given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        retval = np.sum(np.fromiter(
            (np.dot(np.dot((coords[outer][1] - fit_func_systematic(
                coords[outer][0], trial_params)[0]), covinv[outer][inner]),
                 (coords[inner][1]-fit_func_systematic(
                     coords[inner][0], trial_params)[0]))
             for outer, inner in SYMRANGE),
            dtype=np.float, count=COUNT))
        #retval2 = np.sum(np.fromiter(
        #    (np.dot(np.dot((coords[outer][1] - fit_func_systematic(
        #        coords[outer][0], trial_params)), covinv[outer][inner]),
        #         (coords[inner][1]-fit_func_systematic(
        #             coords[inner][0], trial_params)))
        #     for outer, inner in SYMRANGE),
        #    dtype=np.float, count=COUNT))
        #assert retval == retval2, str(retval)+" "+str(retval2)
        return retval*2
