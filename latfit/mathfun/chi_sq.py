from math import fsum
from latfit.mathfun.fit_func import fit_func

def chi_sq(trial_params, covinv, coords, switch):
    """Compute chi^2 given a set of trial parameters,
    the inverse covariance matrix, and the x-y coordinates to fit.
    """
    return fsum([(coords[outer][1]-
                  fit_func(coords[outer][0], trial_params, switch))*
                 covinv[outer][inner]*(coords[inner][1]-
                                       fit_func(coords[inner][0],
                                                trial_params, switch))
                 for outer in range(len(coords))
                 for inner in range(len(coords))])

#delete me
#CHI_SQ = fsum([(COORDS[i][1]-fit_func(COORDS[i][0], trial_params))*
#                   COVINV[i][j]*(COORDS[j][1]-fit_func(COORDS[j][0],
#                                                       a_0, energy))
#                  for i in range(len(COORDS))
#                  for j in range(len(COORDS))])
