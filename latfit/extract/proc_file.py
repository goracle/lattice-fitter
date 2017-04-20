from __future__ import division
from collections import namedtuple
from math import fsum
from itertools import izip
from warnings import warn
from math import log
import sys

from latfit.config import JACKKNIFE
from latfit.config import UNCORR
from latfit.config import EFF_MASS

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

def proc_MEFF(line1,fn1,line2,fn2):
    num = proc_line(line1,fn1)
    denom = proc_line(line2,fn2)
    if num*denom < 0:
        print "***ERROR***"
        print "Negative argument to log in effective mass calc."
        print fn1
        print fn2
        sys.exit(1)
    return log(num/denom)

CSENT = object()
def proc_file(pifile, pjfile=CSENT,i2file=CSENT,j2file=CSENT):
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
    if EFF_MASS and (i2file == CSENT or j2file == CSENT):
        print "***ERROR***"
        print "Missing time adjacent file(s)."
        sys.exit(1)
    if not EFF_MASS:
        i2file = pifile
        j2file = pjfile
    #within true cond. of test, we assume number of columns is one
    with open(pifile) as ithfile:
        avgone = 0
        avgtwo = 0
        count = 0
        for line,linei in izip(ithfile,open(i2file)):
            if EFF_MASS:
                avgone += proc_MEFF(line,pifile,linei,i2file)
            else:
                avgone += proc_line(line,pifile)
            count += 1
        avgone /= count
        with open(pjfile) as jthfile:
            #do uncorrelated fit
            if UNCORR == True and pjfile != pifile:
                return rets(coord=avgone,
                            covar=0)
            counttest = 0
            for line,linej in izip(jthfile,open(j2file)):
                if EFF_MASS:
                    avgtwo += proc_MEFF(line,pjfile,linej,j2file)
                else:
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
            if EFF_MASS:
                coventry = prefactor*fsum([(proc_MEFF(l1,pifile,li1,i2file)-avgone)*(proc_MEFF(l2,pjfile,lj2,j2file)-avgtwo) for l1, li1, l2, lj2 in izip(open(pifile), open(i2file), open(pjfile), open(j2file))])
            else:
                coventry = prefactor*fsum([(proc_line(l1,pifile)-avgone)*(proc_line(l2,pjfile)-avgtwo) for l1, l2 in izip(open(pifile), open(pjfile))])
            return rets(coord=avgone, covar=coventry)
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
