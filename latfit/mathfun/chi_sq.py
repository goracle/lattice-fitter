"""Calculates chi^2"""
import sys
from numpy import dot
from math import exp
import numpy as np

from latfit.config import fit_func
from latfit.config import GEVP, START_PARAMS, SYSTEMATIC_EST

if SYSTEMATIC_EST:

    def fit_func_systematic(ctime, trial_params):
        """ansatz to estimate systematic errors"""
        return [fit_func(ctime, trial_params[:len(START_PARAMS)])[i]+
                trial_params[len(START_PARAMS)+2*i]*exp(-(
                    trial_params[len(START_PARAMS)+(2*i+1)]-trial_params[i]*0)*ctime)
                for i in range(len(START_PARAMS))
        ]

else:
    def fit_func_systematic(ctime, trial_params):
        """blank copy of fit func"""
        return fit_func(ctime, trial_params) 

if GEVP:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        # print("break")
        # print(covinv)
        # print("break 2")
        # print(coords[0][1]-fit_func(coords[0][0], trial_params))
        retval = np.sum([dot(dot(
            (coords[outer][1] - fit_func_systematic(coords[outer][0], trial_params)),
            covinv[outer][inner]), (
                coords[inner][1]-fit_func_systematic(coords[inner][0], trial_params)))
                         for outer in range(len(coords))
                         for inner in range(len(coords))])
        if retval.imag != 0:
            llll = [dot(dot((
                coords[outer][1] - fit_func_systematic(coords[outer][0], trial_params)),
                            covinv[outer][inner]),
                        (coords[inner][1]-fit_func_systematic(coords[inner][0],
                                                   trial_params)))
                    for outer in range(len(coords))
                    for inner in range(len(coords))]
            print("***ERROR***")
            print("imaginary part of chi^2 is non-zero")
            print(coords[0][0])
            print("sep")
            print(coords[0][1])
            print("sep1")
            print((coords[0][1]-fit_func_systematic(coords[0][0][0], trial_params)))
            print("sep2")
            print(covinv[0][0])
            print("sep3")
            print(dot((coords[0][1] - fit_func_systematic(coords[0][0],
                                               trial_params)), covinv[0][0]))
            print("sep4")
            print(llll)
            print("sep5")
            print(np.sum(llll))
            sys.exit(1)
        return retval.real
else:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        return np.sum([dot(dot((
            coords[outer][1] - fit_func(coords[outer][0], trial_params)),
                               covinv[outer][inner]),
                           (coords[inner][1]-fit_func(coords[inner][0],
                                                      trial_params)))
                       for outer in range(len(coords))
                       for inner in range(len(coords))])
