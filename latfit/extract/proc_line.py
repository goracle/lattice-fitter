from warnings import warn
import numpy as np
import sys

#take the real and test for error
def proc_line(line,pifile="BLANK"):
    l = line.split()
    if len(l) == 2:
        warn("Taking the real (first column).")
        return float(l[0])
    elif len(l) == 1:
        return np.float128(line)
    else:
        print("***ERROR***")
        print("Unknown block format.")
        print("File=", pifile)
        sys.exit(1)
