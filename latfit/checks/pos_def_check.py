"""Check to see if covariance matrix is positive definite."""
import sys
import numpy as np
from numpy.linalg import cholesky as posdefexcept
from numpy.linalg import eigvals


def pos_def_check(ccov):
    """Check to see if (cov) matrix is positive definite.  If it is, then
    it must have a Cholesky decomposition.
    The posdefexcept finds this decomposition, and raises a LinAlgError
    if the matrix is not positive definite.
    The program then tells the user to select a different domain.
    The data may still be useable.
    Some people on the internet suggest this is faster, and I was going
    to use a canned routine anyway, so this one won.
    """
    try:
        posdefexcept(ccov)
    except np.linalg.linalg.LinAlgError:
        print("***ERROR***")
        print("Covariance matrix is not positive definite.")
        print("Choose a different domain to fit.")
        print("The data may still be useable.")
        print("List of eigenvalues:")
        testeig = eigvals(ccov)
        for entry in testeig:
            print(entry)
        sys.exit(1)
    return 0
