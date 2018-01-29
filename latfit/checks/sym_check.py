"""Check to see if matrix is symmetric"""
import sys


def sym_check(ccov):
    """Perform a symmetry check on the covariance matrix, just in case
    Note, pos def => symmetric.
    I don't know why the covariance matrix would ever be non-symmetric
    unless the data were mangled.
    """
    for ciii, _ in enumerate(ccov):
        for cjjj in range(ciii+1, len(ccov)):
            if ccov[ciii][cjjj] == ccov[cjjj][ciii]:
                pass
            else:
                print("***ERROR***")
                print("The provided covariance matrix is not symmetric.")
                print("Good fits need a symmetric covariance matrix.")
                print("Please provide different data.")
                print("Exiting.")
                print(sys.exit(1))
    return 0
