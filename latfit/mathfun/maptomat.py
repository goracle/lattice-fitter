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
        len_time = len(cov)
        retcov = np.zeros((dimops*len_time, dimops*len_time))
        for i in range(len_time):
            for j in range(len_time):
                for a in range(dimops):
                    for b in range(dimops):
                        try:
                            retcov[i*dimops+a][j*dimops+b] = swapaxes(
                                cov, 1, 2)[i][a][j][b]
                        except IndexError:
                            print("***ERROR***")
                            print("Dimension mismatch in mapping covariance",
                                  "tensor to matrix.")
                            print("Make sure time indices (i,j) and",
                                  "operator indices (a,b) are like",
                                  "cov[i][a][j][b].")
                            sys.exit(1)
    return retcov
