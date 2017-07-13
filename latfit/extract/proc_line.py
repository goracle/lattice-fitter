"""Process the line."""
import sys
from warnings import warn
import numpy as np

def proc_line(line, pifile="BLANK"):
    """take the real and test for error"""
    linesp = line.split()
    if len(linesp) == 2:
        warn("Taking the real (first column).")
        return float(linesp[0])
    elif len(linesp) == 1:
        return np.float(line)
    else:
        print("***ERROR***")
        print("Unknown block format.")
        print("File=", pifile)
        sys.exit(1)
