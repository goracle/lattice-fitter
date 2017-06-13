import getopt
from collections import namedtuple
import sys

def procargs(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:t:",
                             ["ifolder=", "help", "ifile=", "trials=",
                              "xmin=", "xmax=", 'xstep='])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print("Invalid or missing argument.")
        procargs(["-h"])
    cxmin = object()
    cxmax = object()
    cxstep = object()
    cnextra = object()
    ctrials = object()
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep', 'trials'])
    #Get environment variables from command line.
    for opt, arg in opts:
        if opt == '-h':
            print("usage:", sys.argv[0], "-i <inputfile>")
            print("usage(2):", sys.argv[0], "-f <input folder>")
            print("or, in other words,")
            print("-i <single file of averages with a precomputed covariance")
            print("matrix>")
            print("-f <folder of files of blocks to be averaged (each")
            print("file in the folder is one fit point)>")
            print("or, alternatively")
            print("-f <folder of files with precomputed covariance matrices")
            print("whose resulting fit parameters are averaged>")
            print("")
            print("If inputing a folder of files, each with a")
            print("pre-computed covariance matrix, required argument is")
            print("--trials=<number of files to process>")
            print("i.e. you should have a block for every trial, especially")
            print("if you're doing a jackknife fit.")
            print("")
            print("Optional Arguments")
            print("--xmin=<domain lower bound>")
            print("--xmax=<domain upper bound>")
            print("--xstep=<domain step size> (UNTESTED)")
            print("These are optional in the sense that you don't have")
            print("to enter them immediately, for xmin and xmax, and not")
            print("at all for xstep, whose usage is untested.")
            print("Step size through a folder is assumed to be one.")
            print("NOTE: regular expressions used to process the folders")
            print("may need tweaking, depending on your needs.")
            print("These values are located in proc_folder.py and in ")
            print("__main__.py")
            print("")
            print("latfit by Dan Hoying")
            print("Copyright 2015")
            print("License: Gnu Public License Version 3")
            print("You should've obtained a copy of this license in the")
            print("distribution of these files.")
            sys.exit()
        if opt in "--xmin":
           cxmin = arg
        if opt in "--xstep":
            cxstep = arg
        if opt in "--xmax":
            cxmax = arg
        if opt in "--trials":
            ctrials = arg
    #exiting loop
    for opt, arg in opts:
        if opt in "-i" "--ifile" "-f" "--ifolder":
            return arg, options(xmin=cxmin, xmax=cxmax,
                                        xstep=cxstep, trials=ctrials)
