import getopt
from collections import namedtuple
import sys

def procargs(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:",
                             ["ifolder=", "help", "ifile=",
                              "xmin=", "xmax=", 'xstep='])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print "Invalid or missing argument."
        procargs(["-h"])
    cxmin = object()
    cxmax = object()
    cxstep = object()
    cnextra = object()
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep'])
    #Get environment variables from command line.
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            print "usage(2):", sys.argv[0]
            print "-f <folder of blocks to be averaged>"
            print "Optional Arguments"
            print "--xmin=<domain lower bound>"
            print "--xmax=<domain upper bound>"
            print "--xstep=<domain step size>"
            sys.exit()
        if opt in "--xmin":
           cxmin = arg
        if opt in "--xstep":
            cxstep = arg
        if opt in "--xmax":
            cxmax = arg
        if opt in "--nextra":
            cnextra = arg
    #exiting loop
    for opt, arg in opts:
        if opt in "-i" "--ifile" "-f" "--ifolder":
            return arg, options(xmin=cxmin, xmax=cxmax,
                                        xstep=cxstep)
