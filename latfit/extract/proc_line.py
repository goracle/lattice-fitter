"""Process the line."""
import sys
from warnings import warn
import numpy as np

def proc_line(line, pifile="BLANK"):
    """take the real and test for error"""
    try:
        linesp = line.split()
    except AttributeError:
        linesp = [line.real, line.imag]
    if len(linesp) == 2:
        warn("Taking the real (first column).")
        retval = np.float(linesp[0])
    elif len(linesp) == 1:
        retval = np.float(line)
    else:
        print("***ERROR***")
        print("Unknown block format.")
        print("File=", pifile)
        sys.exit(1)
    return retval
