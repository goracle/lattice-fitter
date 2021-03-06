"""Check for input errors in xmin, xmax"""
import sys

from latfit.procargs import procargs
from latfit.config import FIT, TLOOP, VERBOSE
from latfit.analysis.errorcodes import DOFNonPos

if not FIT:

    def fitrange_err(options, *xargs):
        """Set fit range to be maximum."""
        if xargs:
            pass
        return options.xmin, options.xmax

else:

    def fitrange_err(options, xmin, xmax):
        """Return fit range after checking for errors."""
        xmin = float(xmin)
        xmax = float(xmax)
        assert int(xmin) == float(xmin), xmin
        assert int(xmax) == float(xmax), xmax
        xmin = int(xmin)
        xmax = int(xmax)
        sent1 = object()
        sent2 = object()
        fitmin1 = sent1
        fitmax1 = sent2
        if isinstance(options.fitmax, str) and float(options.fitmax) <= xmax:
            try:
                fitmax1 = float(options.fitmax)
            except ValueError:
                print("***ERROR***")
                print("Invalid max for fit range.")
                procargs(["h"])
        if isinstance(options.fitmin, str) and float(options.fitmin) >= xmin:
            try:
                fitmin1 = float(options.fitmin)
            except ValueError:
                print("***ERROR***")
                print("Invalid min for fit range.")
                procargs(["h"])
        if fitmin1 == sent1 and fitmax1 == sent2:
            if VERBOSE:
                print("Assuming full fit range: ("+str(xmin)+', '+str(xmax)+')')
            fitmin1 = xmin
            fitmax1 = xmax
        elif fitmin1 == sent1:
            fitmin1 = xmin
        elif fitmax1 == sent2:
            fitmax1 = xmax
        if fitmax1 < fitmin1:
            fitmin1, fitmax1 = swap_minmax(fitmin1, fitmax1)
        return fitmin1, fitmax1


def xlim_err(xmin, xmax):
    """Check for errors in the input of xmin and xmax.
    Return xmin and xmax.
    """
    sent1 = object()
    sent2 = object()
    xmin1 = sent1
    xmax1 = sent2
    if isinstance(xmax, str):
        try:
            xmax1 = float(xmax)
        except ValueError:
            print("***ERROR***")
            print("Invalid xmax.")
            procargs(["h"])
    if isinstance(xmin, str):
        try:
            xmin1 = float(xmin)
        except ValueError:
            print("***ERROR***")
            print("Invalid xmin.")
            procargs(["h"])
    if xmin1 == sent1 or xmax1 == sent2:
        print("Now, input valid domain (abscissa).")
        print("xmin<=x<=xmax")
        if xmin1 == sent1:
            print("x min=")
            xmin1 = float(input())
        if xmax1 == sent2:
            print("x max=")
            xmax1 = float(input())
    if xmax1 < xmin1:
        xmin1, xmax1 = swap_minmax(xmin1, xmax1)
    return xmin1, xmax1


def swap_minmax(xmin, xmax):
    """Try to swap xmin with xmax if xmax < xmin"""
    while not TLOOP:
        print("xmax < xmin.  Contradiction.", "Swap xmax for xmin? (y/n)")
        response = str(input())
        if response in ("n", "no", "No", "N"):
            while True:
                print("Abort? (y/n)")
                response = str(input())
                if response in ("n", "no", "No", "N"):
                    break
                if response in ("y", "yes", "Yes", "Y"):
                    sys.exit(0)
                else:
                    print("Sorry, I didn't understand that.")
                    continue
            print("Input new values.")
            print("x min=")
            xmin = float(input())
            print("x max=")
            xmax = float(input())
            if xmax < xmin:
                continue
            break
        if response in ("y", "yes", "Yes", "Y"):
            xmin, xmax = xmax, xmin
            break
        print("Sorry, I didn't understand that.")
    else:
        raise DOFNonPos
    return xmin, xmax
