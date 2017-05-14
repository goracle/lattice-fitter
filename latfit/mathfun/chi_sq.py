from math import fsum
from latfit.config import fit_func

def chi_sq(trial_params, covinv, coords):
    """Compute chi^2 given a set of trial parameters,
    the inverse covariance matrix, and the x-y coordinates to fit.
    """
    return fsum([(coords[outer][1]- fit_func(coords[outer][0], trial_params))* covinv[outer][inner]*(coords[inner][1]-fit_func(coords[inner][0],trial_params)) for outer in range(len(coords)) for inner in range(len(coords))])

##Superfluous test code
    #l=len(coords)
    #b=0
    #for i in range(l):
    #    for j in range(l):
    #        b+=a[i+j*l]
            #print i,j,a[i+j*l],b
    #return fsum(a)
