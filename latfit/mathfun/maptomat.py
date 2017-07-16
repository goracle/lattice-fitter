"""Map a covariance tensor (in use during GEVP analysis) to a matrix"""
import sys
import numpy as np
from numpy import swapaxes

def maptomat(cov, dimops=1):
    """Map tensor to matrix, using dimops to get dimensions of
    covariance tensor's third and fourth rank.
    New matrix is blocks of dimension dimops x dimops, indexed by first two
    ranks.
    """
    if dimops == 1:
        retcov = cov
    else:
        dimcov = len(cov)
        retcov = np.zeros((dimops*dimcov, dimops*dimcov))

    try:
        retcov = np.array(
            [[swapaxes(cov, 1, 2)[i][opa][j][opb]
              for j in range(dimcov) for opb in range(dimops)]
             for i in range(dimcov) for opa in range(dimops)])
    except IndexError:
        print("***ERROR***")
        print("Dimension mismatch in mapping covariance",
              "tensor to matrix.")
        print("Make sure time indices (i,j) and",
              "operator indices (a,b) are like",
              "cov[i][a][j][b].")
        sys.exit(1)
    return retcov
