from math import fsum
from numpy import dot
import numpy as np
from latfit.config import fit_func

def chi_sq(trial_params, covinv, coords):
    """Compute chi^2 given a set of trial parameters,
    the inverse covariance matrix, and the x-y coordinates to fit.
    """
    return fsum([dot((np.array(coords[outer][1])- fit_func(coords[outer][0], trial_params))*covinv[outer][inner],(np.array(coords[inner][1])-fit_func(coords[inner][0],trial_params))) for outer in range(len(coords)) for inner in range(len(coords))])
