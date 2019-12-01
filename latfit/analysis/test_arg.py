"""Test argument to log"""
import sys
from warnings import warn
import numpy as np
#import mpi4py
#mpi4py.rc.recv_mprobe = False
#from mpi4py import MPI

SENT = object()

#MPIRANK = MPI.COMM_WORLD.rank

class NegLogArgument(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, arg=None, message='', corrs=None, no_print=False):
        if not no_print:
            warn("argument to log in eff. mass"+" calc is than 0: "+str(
                arg))
            print("argument to log in effective mass",
                  "calc is less than 0:", arg)
            super(NegLogArgument, self).__init__(message)
            if corrs is not None:
                for i, corr in enumerate(corrs):
                    print("corrs["+str(i)+"] = ", corr)
        self.message = message


def test_arg(arg, sent=None):
    """Test if arg to log is less than zero (imaginary mass)
    """
    if arg <= 0 and sent != 0:
        # print("***ERROR***")
        raise NegLogArgument(arg=arg)


def zero_p(corr1, corr2=None, times=None):
    """Check to see if denominator of effective mass equation is 0."""
    errlevel = 0
    if corr1 is None and np.array_equal(corr2, np.zeros(corr2.shape)):
        errlevel = 1
    elif np.array_equal(corr1, corr2):
        errlevel = 2
    try:
        np.testing.assert_allclose(corr1, corr2)
        errlevel = 2
    except AssertionError:
        pass
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
    if sol <= 0 and SENT != 0:
        no_print = True
        if times is not None and isinstance(times, list):
            if times[0] not in testsol.problemtimes:
                no_print = False
                testsol.problemtimes.append(times[0])
                print("problematic time slices:", times)
        raise NegLogArgument(arg=sol, corrs=corrs, no_print=no_print)
testsol.problemtimes = []
