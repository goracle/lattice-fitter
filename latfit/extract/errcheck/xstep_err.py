from latfit.procargs import procargs
import os

def xstep_err(xstep, INPUT):
    """Check for error in the domain step size.
    Return the step size.
    """
    XSTEP = -1
    if isinstance(xstep, str):
        try:
            OPSTEMP = xstep
            OPSTEMP = float(OPSTEMP)
        except ValueError:
            print "***ERROR***"
            print "Invalid step size."
            print "Expecting an float >= 0."
            procargs(["h"])
        if OPSTEMP >= 0:
            XSTEP = OPSTEMP
        else:
            XSTEP = -1
    #We only care about step size for multi file setup
    if XSTEP == -1 and os.path.isdir(INPUT):
        print "Assuming domain step size is 1 (int)."
        XSTEP = 1
    return XSTEP
