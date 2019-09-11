"""Does a matrix or tensor inverse, checks for errors."""
from numpy.linalg import inv, tensorinv
from numpy import swapaxes as swap
import numpy as np

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
