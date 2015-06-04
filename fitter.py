#!/usr/bin/env python

"""Fits function to data.  Computes chi^2 and errors"""

#plotting part

import matplotlib.pyplot as plt
import re
import sys
import getopt



#global constants

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print sys.argv[0], "-i <inputfile>"
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print sys.argv[0], "-i <inputfile>"
            sys.exit()
        elif opt in ("-i" "--ifile"):
            return arg

        
#main part
if __name__ == "__main__": 
    inputfile = main(sys.argv[1:]) 
    #re.match(r'',input part from file)
    try:
        input = open(inputfile,"r")
    except IOError:
        print "File:", inputfile, "not found"
        sys.exit(2)
    sys.exit()
