"""Check for input errors in xmin, xmax"""
import sys

from latfit.procargs import procargs

def xlim_err(xmin, xmax):
    """Check for errors in the input of xmin and xmax.
    Return xmin and xmax.
    """
    sent1 = object()
    sent2 = object()
    xmin = sent1
    xmax = sent2
    if isinstance(xmax, str):
        try:
            xmax = float(xmax)
        except ValueError:
            print("***ERROR***")
            print("Invalid xmax.")
            procargs(["h"])
    if isinstance(xmin, str):
        try:
            xmin = float(xmin)
        except ValueError:
            print("***ERROR***")
            print("Invalid xmin.")
            procargs(["h"])
    if xmin == sent1 or xmax == sent2:
        print("Now, input valid domain (abscissa).")
        print("xmin<=x<=xmax")
        if xmin == sent1:
            print("x min=")
            xmin = float(input())
        if xmax == sent2:
            print("x max=")
            xmax = float(input())
    if xmax < xmin:
        swap_minmax(xmin, xmax)
    return xmin, xmax

def swap_minmax(xmin, xmax):
    """Try to swap xmin with xmax if xmax < xmin"""
    while True:
        print("xmax < xmin.  Contradiction.", "Swap xmax for xmin? (y/n)")
        response = str(input())
        if (response == "n" or response == "no" or
                response == "No" or response == "N"):
            while True:
                print("Abort? (y/n)")
                response = str(input())
                if (response == "n" or response == "no" or
                        response == "No" or response == "N"):
                    break
                if (response == "y" or response == "yes"
                        or response == "Yes" or response == "Y"):
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
            else:
                break
        if (response == "y" or response == "yes"
                or response == "Yes" or response == "Y"):
            xmin, xmax = xmax, xmin
            break
        else:
            print("Sorry, I didn't understand that.")
            continue
    return xmin, xmax
