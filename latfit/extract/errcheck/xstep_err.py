"""Check stepsize for errors."""
import os

from latfit.procargs import procargs
from latfit.config import STYPE


def xstep_err(xstep, input_f):
    """Check for error in the domain step size.
    Return the step size.
    """
    xstep = -1
    if isinstance(xstep, str):
        try:
            opstemp = xstep
            opstemp = float(opstemp)
        except ValueError:
            print("***ERROR***")
            print("Invalid step size.")
            print("Expecting an float >= 0.")
            procargs(["h"])
        if opstemp >= 0:
            xstep = opstemp
        else:
            xstep = -1
    # We only care about step size for multi file setup
    if xstep == -1 and (os.path.isdir(input_f) or STYPE == 'hdf5'):
        print("Assuming domain step size is 1 (int).")
        xstep = 1
    return xstep
