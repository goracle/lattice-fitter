"""Does a matrix or tensor inverse, checks for errors."""
import sys
from numpy.linalg import inv, det, tensorinv
from numpy import swapaxes as swap
import numpy as np

from latfit.mathfun.maptomat import maptomat

class AvgCovSingular(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, message=''):
        print("***ERROR***")
        print("Average covariance matrix is singular")
        super(AvgCovSingular, self).__init__(message)
        self.message = message

def covinv_avg(cov, dimops=1):
    """Return the inverse of the average covariance matrix.
    In the case of the GEVP, return the tensor inverse of the
    covariance tensor.
    """
    try:
        if dimops == 1:
            covinv = inv(cov)
        else:
            # swap axes, take inverse, swap back
            covinv = swap(tensorinv(swap(cov, 1, 2)), 1, 2)
    except np.linalg.linalg.LinAlgError as err:
        if str(err) == 'Singular matrix':
            print("Average covariance matrix is singular.")
            print("Check to make sure plot range does",
                  "not contain a mirror image.")
        raise
    return covinv

"""
            retcov = maptomat(cov, dimops)
            if dimops > 1:
                try:
                    covinv = inv(retcov)
                    print("Regular matrix inversion succeeded.  Weird.")
                    print("check map tensor to matrix", "function for errors")
                except np.linalg.linalg.LinAlgError as err:
                    if err == 'Singular matrix':
                        print("Regular matrix inversion also failed")
                        print("rows:")
                        for i, _ in enumerate(retcov):
                            print(np.array2string(retcov[i], separator=', '))
                        print("columns:")
                        for i in range(len(retcov)):
                            print(np.array2string(
                                np.transpose(retcov)[i], separator=', '))
                        print("det=", det(retcov))
            sys.exit(1)

"""
