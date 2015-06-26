#!/usr/bin/env python

"""Fit function to data.
Compute chi^2 and errors.
Plot fit with error bars.
"""

#plotting part

import matplotlib.pyplot as plt
#import re
import sys
import getopt
#from numbers import Number
#from decimal import Decimal
#from fractions import Fraction
#from numpy import array
#from numpy import linalg
import numpy as np
from numpy.linalg import inv
from collections import defaultdict
import os, re

#global constants

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:", ["ifolder=", "help", "ifile="])[0]
    except getopt.GetoptError:
        print "Invalid argument."
        print "usage:", sys.argv[0], "-i <inputfile>"
        print "usage(2):", sys.argv[0], "-f <folder of jackknife blocks>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            print "usage(2):", sys.argv[0], "-f <folder of jackknife blocks>"
            sys.exit()
        elif opt in "-i" "--ifile":
            return arg
        elif opt in "-f" "--ifolder":
            return arg

def tree():
    """Return a multidimensional dict"""
    return defaultdict(tree)

#type of function: pade, used for certain types of qcd theories
#be sure to put triple quotes back in afterwards for the docstring
"""def fit_func(qsq, a_0, a_1, b_1):
    Give result of function computed to fit the data given in <inputfile>
    (See main(argv))
    return qsq*(a_0+a_1/(b_1+qsq))
"""

#simple exponential
def fit_func(t, a_0, energy):
    """Give result of function computed to fit the data given"""
    return a_0*exp(-t*energy)

def proc_folder(folder, time):
    """Process folder where jackknife blocks are stored.
    Return file corresponding to current ensemble (lattice time slice).
    Assumes file is <anything>t<time><anything>
    Assumes only 1 valid file per match, e.g. ...t3... doesn't happen more 
    than once
    """
    result = []
    #build regex as a string
    my_regex = r"t" + str(time)
    for root, dirs, files in os.walk(folder):
        for name in files:
            if (re.match(my_regex, name)):
                #delete me
                print folder
                print name
                #end delete me, print test statements
                return name

def proc_file(IFILE):
    """Process the current file, determining errors"""
    INPUTFILE = open(IFILE, "r")
    pass

#main part
if __name__ == "__main__":
    #re.match(r'',input part from file)
    INPUT = main(sys.argv[1:])
    #error handling
    if not (os.path.isfile(INPUT) or os.path.isdir(INPUT)):
        print "File:", INPUT, "not found"
        print "Folder:", INPUT, "also not found."
        print "usage:", sys.argv[0], "-i <inputfile>"
        print "usage(2):", sys.argv[0], "-f <folder of jackknife blocks>"
        sys.exit(2)
    if (os.path.isfile(INPUT)):
        proc_file(INPUT)
    elif (os.path.isdir(INPUT)):
        print "Now, input valid time domain (abscissa)."
        print "time min<=t<=time max"
        print "time min="
        TMIN = int(raw_input())
        print "time max="
        TMAX = int(raw_input())
        for time in range(TMIN,TMAX+1):
            IFILE = proc_folder(INPUT, time)
            try:
                TRIAL = open(IFILE, "r")
            except TypeError:
                print "Either time range is invalid, or folder is invalid."  
                print "Double check contents of folder."
                sys.exit(1)
            proc_file(IFILE)
    """COORDS = []
    CDICT = tree()
    with open(INPUTFILE) as f:
        for line in f:
            try:
                cols = [float(x) for x in line.split()]
            except ValueError:
                print "ignored line: '", line, "'"
                continue
            #make a list of
            #[indep. variable, empirical dep variable] = COORDS
            if len(cols) == 2:
                COORDS.append([cols[0], cols[1]])
            #Find the C_ij^-1 inverse covariance matrix
            else:
                CDICT[cols[0]][cols[1]] = cols[2]
            #matchObj=re.search(r'(.*?) +(.*?)',line)
            #if matchObj:
            #    print "gr", matchObj.group(1)
            #    print "ga", matchObj.group(2)
            #print line
    #build the covariance matrix from the CDICT
    COV = [[CDICT[COORDS[i][0]][COORDS[j][0]]
            for i in range(len(COORDS))]
           for j in range(len(COORDS))]
    print COV[2][2]
    COVINV = inv(COV)
    print COVINV[2][2]
    #compute chi^2
    CHI_SQ = sum([(COORDS[i][1]-fit_func(COORDS[i][0], 1, 1, 1))*
                  COVINV[i][j]*(COORDS[j][1]-fit_func(COORDS[j][0], 1, 1, 1))
                  for i in range(len(COORDS))
                  for j in range(len(COORDS))])
    print "chi^2 = ", CHI_SQ
    #minimize chi^2
    #plotting part, example
    #needs to be eliminated
    x = np.arange(-5, 5, 0.1)
    y = np.power(x, 2)
    yerrV = 0.1 + 0.2*np.sqrt(x+5.1)
    xerrV = 0.1 + yerrV
    plt.figure()
    plt.errorbar(x, y, xerr=xerrV, yerr=yerrV)
    plt.title(r"Such example (quadratic) much LaTeX: $\sigma$")
    plt.annotate('wow', xy=(.3,.09), xytext=(-2, 15), arrowprops=dict(facecolor='black', shrink=0.05))
    plt.show()
    #end example
    """
    sys.exit()
