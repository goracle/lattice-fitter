"""Test argument to log"""
import sys
from warnings import warn
import numpy as np

SENT = object()


def test_arg(arg, sent=None):
    """Test if arg to log is less than zero (imaginary mass)
    """
    if arg <= 0 and sent != 0:
        # print("***ERROR***")
        warn("argument to log in eff. mass"+" calc is than 0: "+str(
            arg))
        print("argument to log in effective mass",
              "calc is less than 0:", arg)
        return False
    return True


def zero_p(corr1, corr2=None, times=None):
    """Check to see if denominator of effective mass equation is 0."""
    errlevel = 0
    if corr1 is None and np.array_equal(corr2, np.zeros(corr2.shape)):
        errlevel = 1
    elif np.array_equal(corr1, corr2):
        errlevel = 2
    if errlevel:
        corrs = (corr1, corr2)
        print("Error in zero_p.")
        for i in range(errlevel):
            print("corrs["+str(i)+"] = ", corrs[i])
        if times is not None:
            print("problematic time slices:", times)
        sys.exit(1)


def testsol(sol, corrs, times=None):
    """Test ratio in effective mass equation to see if it's less < 0."""
    if not test_arg(sol, SENT):
        print("Error in testsol.")
        for i, corr in enumerate(corrs):
            print("corrs["+str(i)+"] = ", corr)
        if times is not None:
            print("problematic time slices:", times)
        sys.exit(1)
