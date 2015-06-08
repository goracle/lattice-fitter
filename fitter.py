#!/usr/bin/env python

"""Fits function to data.  Computes chi^2 and errors"""

#plotting part

import matplotlib.pyplot as plt
import re
import sys
import getopt
from numbers import Number
from decimal import Decimal
from fractions import Fraction
from numpy import array
from numpy import linalg
from numpy.linalg import inv
from collections import defaultdict

#global constants

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print "File not found"
        print "usage:", sys.argv[0], "-i <inputfile>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            sys.exit()
        elif opt in ("-i" "--ifile"):
            return arg

def tree():
    """Used to make a multidimensional dict"""
    return defaultdict(tree)

        
#main part
if __name__ == "__main__": 
    #re.match(r'',input part from file)
    try:
        inputfile = main(sys.argv[1:]) 
        input = open(inputfile,"r")
    except (IOError,TypeError):
        print "File:", inputfile, "not found"
        print "usage:", sys.argv[0], "-i <inputfile>"
        sys.exit(2)
    coords=[]
    dict=tree()
    with open(inputfile) as f:
        for line in f:
            try:
                cols = [float(x) for x in line.split()] 
            except (ValueError):
                continue
            #make a list of x y coords
            if len(cols)==2:
                coords.append([cols[0],cols[1]])
            #Find the C_ij^-1 inverse covariance matrix
            else:
                dict[cols[0]][cols[1]]=cols[2]
            #matchObj=re.search(r'(.*?) +(.*?)',line)
            #if matchObj:
            #    print "gr", matchObj.group(1)
            #    print "ga", matchObj.group(2)
            #print line
    #build the covariance matrix from the dict
    cov=array([[dict[coords[i][0]][coords[j][0]]  for i in range(len(coords))] for j in range(len(coords))])
    print cov[2,3]
    covinv=inv(cov)
    print covinv[2,2]
    print len(coords)
    sys.exit()
