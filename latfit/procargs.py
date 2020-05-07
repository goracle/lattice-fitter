"""Parses the command line."""
import sys
import getopt
from recordtype import recordtype

OPTIONS = recordtype('ops', 'xmin xmax xstep trials fitmin fitmax procs')
ops = OPTIONS

def procargs(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:t:",
                             ["ifolder=", "help", "ifile=", "trials=",
                              "xmin=", "xmax=", 'xstep=', 'procs=',
                              'fitmin=', 'fitmax='])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print("Invalid or missing argument.")
        procargs(["-h"])
    cxmin = object()
    cxmax = object()
    cxstep = object()
    cfitmin = object()
    cfitmax = object()
    cprocs = object()
    # cnextra = object()
    ctrials = object()
    # Get environment variables from command line.
    for opt, arg in opts:
        if opt == '-h':
            printhelp()
            sys.exit(0)
        if opt in "--xmin":
            cxmin = arg
        elif opt in "--xstep":
            cxstep = arg
        elif opt in "--xmax":
            cxmax = arg
        elif opt in "--trials":
            ctrials = arg
        elif opt == "--fitmin":
            cfitmin = arg
        elif opt == "--fitmax":
            cfitmax = arg
        elif opt == "--procs":
            cprocs = arg
    # exiting loop
    options = OPTIONS(None, None, None, None, None, None, None)
    for opt, arg in opts:
        if opt in "-i" "--ifile" "-f" "--ifolder":
            options.xmin = cxmin
            options.xmax = cxmax
            options.xstep = cxstep
            options.trials = ctrials
            options.fitmin = cfitmin
            options.fitmax = cfitmax
            options.procs = cprocs
            retval = arg, options
            break
    return retval


def printhelp():
    """Print help."""
    print("usage:", sys.argv[0], "-i <inputfile>")
    print("usage(2):", sys.argv[0], "-f <input folder>")
    print("or, in other words,")
    print("-i <single file of averages with a precomputed",
          "covariance matrix>")
    print("-f <folder of files of blocks to be averaged (each")
    print("file in the folder is one fit point)>",
          "or, alternatively")
    print("-f <folder of files with precomputed covariance matrices")
    print("whose resulting fit parameters are averaged>\n\n" +
          "If inputing a folder of files, each with a " +
          "pre-computed covariance matrix, required argument is" +
          " --trials=<number of files to process>")
    print("i.e. you should have a block for every trial," +
          " especially if you're doing a jackknife fit.\n")
    print("Optional Arguments\n")
    print("--xmin=<domain lower bound>\n--xmax=<domain upper bound>")
    print("--xstep=<domain step size> (UNTESTED)")
    print("--fitmin=<x value to start fitting>")
    print("--fitmax=<x value to stop fitting>")
    print("These are optional in the sense that you",
          "don't have to enter them immediately")
    print("Step size through a data set is assumed to be one.")
    print("NOTE: regular expressions used to process",
          "the folders\nmay need tweaking, depending on your needs.")
    print("These values are located in proc_folder.py and in",
          "__main__.py\n")
    print("latfit by Dan Hoying,", "Copyright 2018")
    print("luscherzeta.h by Christopher Kelly,", "Copyright 2014")
    print("Both under License: Gnu Public License Version 3" +
          "\nYou should've obtained a copy of this license " +
          "in the\ndistribution of these files.")
