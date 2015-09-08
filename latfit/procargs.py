import getopt
from collections import namedtuple

def procargs(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:s:",
                             ["ifolder=", "help", "ifile=",
                              "switch=", "xmin=", "xmax=", 'xstep=',
                              'nextra='])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print "Invalid or missing argument."
        procargs(["-h"])
    switch = -1
    cxmin = object()
    cxmax = object()
    cxstep = object()
    cnextra = object()
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep', 'nextra'])
    #Get environment variables from command line.
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            print "usage(2):", sys.argv[0]
            print "-f <folder of blocks to be averaged>"
            print "Required aruments:"
            print "-s <fit function to use>"
            print "fit function options are:"
            print "0: Pade"
            print "Optional argument for Pade:"
            print "--nextra=<number of extra arguments/2>"
            print "E.g., if you enter --nextra=3, 6 additional fit"
            print "parameters will be used."
            print "1: Exponential"
            print "Optional Arguments"
            print "--xmin=<domain lower bound>"
            print "--xmax=<domain upper bound>"
            print "--xstep=<domain step size>"
            sys.exit()
        if opt in "-s" "--switch":
            switch = arg
        if opt in "--xmin":
            cxmin = arg
        if opt in "--xstep":
            cxstep = arg
        if opt in "--xmax":
            cxmax = arg
        if opt in "--nextra":
            cnextra = arg
    if not switch in set(['0', '1']):
        print "You need to pick a fit function."
        procargs(["-h"])
    #exiting loop
    for opt, arg in opts:
        if opt in "-i" "--ifile" "-f" "--ifolder":
            return arg, switch, options(xmin=cxmin, xmax=cxmax,
                                        xstep=cxstep, nextra=cnextra)
