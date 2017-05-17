from latfit.procargs import procargs
import sys

def xlim_err(xmin, xmax):
    """Check for errors in the input of xmin and xmax.
    Return xmin and xmax.
    """
    SENT1 = object()
    SENT2 = object()
    XMIN = SENT1
    XMAX = SENT2
    if isinstance(xmax, str):
        try:
            XMAX = float(xmax)
        except ValueError:
            print "***ERROR***"
            print "Invalid xmax."
            procargs(["h"])
    if isinstance(xmin, str):
        try:
            XMIN = float(xmin)
        except ValueError:
            print "***ERROR***"
            print "Invalid xmin."
            procargs(["h"])
    if XMIN == SENT1 or XMAX == SENT2:
        print "Now, input valid domain (abscissa)."
        print "xmin<=x<=xmax"
        if XMIN == SENT1:
            print "x min="
            XMIN = float(raw_input())
        if XMAX == SENT2:
            print "x max="
            XMAX = float(raw_input())
    if XMAX < XMIN:
        while True:
            print "xmax < xmin.  Contradiction."
            print "Swap xmax for xmin? (y/n)"
            RESP = str(raw_input())
            if (RESP == "n" or RESP == "no" or
                    RESP == "No" or RESP == "N"):
                while True:
                    print "Abort? (y/n)"
                    RESP = str(raw_input())
                    if (RESP == "n" or RESP == "no" or
                            RESP == "No" or RESP == "N"):
                        break
                    if (RESP == "y" or RESP == "yes"
                            or RESP == "Yes" or RESP == "Y"):
                        sys.exit(0)
                    else:
                        print "Sorry, I didn't understand that."
                        continue
                print "Input new values."
                print "x min="
                XMIN = float(raw_input())
                print "x max="
                XMAX = float(raw_input())
                if XMAX < XMIN:
                    continue
                else:
                    break
            if (RESP == "y" or RESP == "yes"
                    or RESP == "Yes" or RESP == "Y"):
                XMIN, XMAX = XMAX, XMIN
                break
            else:
                print "Sorry, I didn't understand that."
                continue
    return XMIN, XMAX
