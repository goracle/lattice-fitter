from __future__ import division
from collections import namedtuple
from math import fsum
from itertools import izip
from warnings import warn
#from math import log
from math import acosh
import sys
import numpy as np

from latfit.config import JACKKNIFE
from latfit.config import UNCORR
from latfit.config import EFF_MASS
from latfit.config import C

def proc_line(line,pifile="BLANK"):
    l = line.split()
    if len(l) == 2:
        warn("Taking the real (first column).")
        return float(l[0])
    elif len(l) == 1:
        return np.float128(line)
    else:
        print "***ERROR***"
        print "Unknown block format."
        print "File=", pifile
        sys.exit(1)

def proc_MEFF(line1,line2,line3,files):
    fn1=files[0]
    fn2=files[1]
    fn3=files[2]
    C1 = proc_line(line1,fn1)
    C2 = proc_line(line2,fn2)
    C3 = proc_line(line3,fn3)
    arg = (C1+C3-2*C)/2/(C2-C)
    if arg < 1:
        print "***ERROR***"
        print "argument to acosh in effective mass calc is less than 1:",arg
        print fn1
        print fn2
        print fn3
        sys.exit(1)
    return acosh(arg)

CSENT = object()
def proc_file(pifile, pjfile=CSENT,extra_pairs=[(None,None),(None,None)]):
    """Process the current file.
    Return covariance matrix entry I,indexj in the case of multi-file
    structure.
    Return the covariance matrix for single file.
    """
    #initialize return value named tuple. in other words:
    #create a type of object, rets, to hold return values
    #instantiate it with return values, then return that instantiation
    i2file=extra_pairs[0][0]
    j2file=extra_pairs[0][1]
    i3file=extra_pairs[1][0]
    j3file=extra_pairs[1][1]
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
        i3file = pifile
        j2file = pjfile
        j3file = pjfile
    else:
        ifiles=[pifile]
        ifiles.extend([extra_pairs[i][0] for i in range(2)])
        jfiles=[pjfile]
        jfiles.extend([extra_pairs[i][1] for i in range(2)])
    #within true cond. of test, we assume number of columns is one
    with open(pifile) as ithfile:
        avgone = 0
        avgtwo = 0
        count = 0
        for linei,linei2,linei3 in izip(ithfile,open(i2file),open(i3file)):
            if EFF_MASS:
                avgone += proc_MEFF(linei,linei2,linei3,ifiles)
            else:
                avgone += proc_line(linei,pifile)
            count += 1
        avgone /= count
        with open(pjfile) as jthfile:
            #do uncorrelated fit
            if UNCORR == True and pjfile != pifile:
                return rets(coord=avgone,
                            covar=0)
            counttest = 0
            for linej,linej2,linej3 in izip(jthfile,open(j2file),open(j3file)):
                if EFF_MASS:
                    avgtwo += proc_MEFF(linej,linej2,linej3,jfiles)
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
                warn("Applying jackknife correction to cov. matrix.")
                prefactor = (count-1.0)/(1.0*count)
            elif JACKKNIFE == 'NO':
                prefactor = (1.0)/((count-1.0)*(1.0*count))
            else:
                print "Edit the config file."
                print "Invalid value of parameter JACKKNIFE"
                sys.exit(1)
            if EFF_MASS:
                coventry = prefactor*fsum([(proc_MEFF(li1,li2,li3,ifiles)-avgone)*(proc_MEFF(lj1,lj2,lj3,jfiles)-avgtwo) for li1, li2, li3, lj1, lj2, lj3 in izip(open(pifile), open(i2file), open(i3file), open(pjfile), open(j2file), open(j3file))])
            else:
                coventry = prefactor*fsum([(proc_line(l1,pifile)-avgone)*(proc_line(l2,pjfile)-avgtwo) for l1, l2 in izip(open(pifile), open(pjfile))])
    return rets(coord=avgone, covar=coventry)
