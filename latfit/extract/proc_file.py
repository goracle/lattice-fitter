from __future__ import division
from collections import namedtuple
from math import fsum
from itertools import izip
from warnings import warn

from latfit.config import JACKKNIFE
from latfit.config import UNCORR

def proc_line(line,pifile="BLANK"):
    l = line.split()
    if len(l) == 2:
        warn("Taking the real (first column).")
        return float(l[0])
    elif len(l) == 1:
        return float(line)
    else:
        print "***ERROR***"
        print "Unknown block format."
        print "File=", pifile
        sys.exit(1)

CSENT = object()
def proc_file(pifile, pjfile=CSENT):
    """Process the current file.
    Return covariance matrix entry I,indexj in the case of multi-file
    structure.
    Return the covariance matrix for single file.
    """
    #initialize return value named tuple. in other words:
    #create a type of object, rets, to hold return values
    #instantiate it with return values, then return that instantiation
    rets = namedtuple('rets', ['coord', 'covar'])
    if pjfile == CSENT:
        print "***ERROR***"
        print "Missing secondary file."
        sys.exit(1)
    #within true cond. of test, we assume number of columns is one
    with open(pifile) as ithfile:
        avgone = 0
        avgtwo = 0
        count = 0
        for line in ithfile:
            avgone += proc_line(line,pifile)
            count += 1
        avgone /= count
        with open(pjfile) as jthfile:
            #do uncorrelated fit
            if UNCORR == True and pjfile != pifile:
                return rets(coord=avgone,
                            covar=0)
            counttest = 0
            for line in jthfile:
                avgtwo += proc_line(line,pjfile)
                counttest += 1
            if not counttest == count:
                print "***ERROR***"
                print "Number of rows in paired files doesn't match"
                print count, counttest
                print "Offending files:", pifile, "and", pjfile
                sys.exit(1)
            else:
                avgtwo /= count
            #cov[I][indexj]=return value for folder-style
            #applying jackknife correction of (count-1)^2
            if JACKKNIFE == 'YES':
                prefactor = (count-1.0)/(1.0*count)
            elif JACKKNIFE == 'NO':
                prefactor = (1.0)/((count-1.0)*(1.0*count))
            else:
                print "Edit the config file."
                print "Invalid value of parameter JACKKNIFE"
                sys.exit(1)
            coventry = prefactor*fsum([
                (proc_line(l1,pifile)-avgone)*(proc_line(l2,pjfile)-avgtwo)
                for l1, l2 in izip(open(pifile), open(pjfile))])
            return rets(coord=avgone,
                        covar=coventry)
    print "***Unexpected Error***"
    print "If you\'re seeing this program has a bug that needs fixing"
    sys.exit(1)
    #delete me (if working)
            #append at position i,j to the covariance matrix entry
        #store precomputed covariance matrix (if it exists)
        #in convenient form
        #try:
        #COV = [[CDICT[PROCCOORDS[i][0]][PROCCOORDS[j][0]]
        #        for i in range(len(PROCCOORDS))]
        #       for j in range(len(PROCCOORDS))]
    #delete me end (if working)
