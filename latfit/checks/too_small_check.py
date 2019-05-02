"""Check to see if eigenvals are too small"""
import sys
from numpy.linalg import eigvals


def too_small_check(ccov, eigcut=10**(-10)):
    """Check to see if the matrix eigenvalues are too small.
    This can cause problems when computing chi^2 (t^2) due to precision loss
    """
    testeig = eigvals(ccov)
    flag = 0
    for entry in testeig:
        if entry < eigcut:
            flag = 1
            print("***Warning***")
            print("Range selected has a covariance matrix with")
            print("very small eigenvalues.  This can cause problems")
            print("in computing chi^2 (t^2), as well as quantities derived")
            print("from chi^2 (t^2). The cuttoff is set at:", eigcut)
            print("Problematic eigenvalue = ", entry)
            break
    if flag == 1:
        print("List of eigenvalues of covariance matrix:")
        for entry in testeig:
            print(entry)
        while True:
            print("Continue? (y/n)")
            cresp = str(input())
            if (cresp == "n" or cresp == "no"
                    or cresp == "No" or cresp == "N"):
                sys.exit(0)
            if (cresp == "y" or cresp == "yes"
                    or cresp == "Yes" or cresp == "Y"):
                break
            else:
                print("Sorry, I didn't understand that.")
                continue
    return 0
