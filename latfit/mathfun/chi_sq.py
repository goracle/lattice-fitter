from numpy import dot
import numpy as np
import sys

from latfit.config import fit_func
from latfit.config import GEVP

if GEVP:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        #print("break")
        #print(covinv)
        #print("break 2")
        #print(coords[0][1]-fit_func(coords[0][0],trial_params))
        retval=np.sum([dot(dot((coords[outer][1]- fit_func(coords[outer][0], trial_params)),covinv[outer][inner]),(coords[inner][1]-fit_func(coords[inner][0],trial_params))) for outer in range(len(coords)) for inner in range(len(coords))])
        if retval.imag!=0:
            sys.exit(1)
        return retval.real
else:
    def chi_sq(trial_params, covinv, coords):
        """Compute chi^2 given a set of trial parameters,
        the inverse covariance matrix, and the x-y coordinates to fit.
        """
        return np.sum([dot(dot((coords[outer][1]- fit_func(coords[outer][0], trial_params)),covinv[outer][inner]),(coords[inner][1]-fit_func(coords[inner][0],trial_params))) for outer in range(len(coords)) for inner in range(len(coords))])
