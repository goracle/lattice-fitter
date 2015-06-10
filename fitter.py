#!/usr/bin/env python

"""Fit function to data.
Compute chi^2 and errors.
Plot fit with error bars.
"""

#plotting part

#import matplotlib.pyplot as plt
#import re
import sys
import getopt
#from numbers import Number
#from decimal import Decimal
#from fractions import Fraction
#from numpy import array
#from numpy import linalg
from numpy.linalg import inv
from collections import defaultdict

#global constants

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "hi:", ["ifile="])[0]
    except getopt.GetoptError:
        print "File not found"
        print "usage:", sys.argv[0], "-i <inputfile>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            sys.exit()
        elif opt in "-i" "--ifile":
            return arg

def tree():
    """Return a multidimensional dict"""
    return defaultdict(tree)

def fit_func(qsq, a_0, a_1, b_1):
    """Give result of function computed to fit the data given in <inputfile>
    (See main(argv))
    """
    return qsq*(a_0+a_1/(b_1+qsq))

#main part
if __name__ == "__main__":
    #re.match(r'',input part from file)
    try:
        INPUTFILE = main(sys.argv[1:])
        IFILE = open(INPUTFILE, "r")
    except (IOError, TypeError):
        print "File:", INPUTFILE, "not found"
        print "usage:", sys.argv[0], "-i <inputfile>"
        sys.exit(2)
    COORDS = []
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
    sys.exit()
