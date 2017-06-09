from __future__ import division
from collections import namedtuple
from math import fsum
from itertools import izip
from warnings import warn
import re
from math import log
from math import acosh
import sys
import numpy as np
from sympy import nsolve,cosh
from sympy.abc import x,y,z

from latfit.config import JACKKNIFE
from latfit.config import UNCORR
from latfit.config import EFF_MASS
from latfit.config import C
from latfit.config import EFF_MASS_METHOD
from latfit.config import FIT
from latfit.config import START_PARAMS
from latfit.config import fit_func_3pt_sym

#take the real and test for error
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

#almost solve a cosh, analytic
if EFF_MASS_METHOD == 1:
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

#numerically solve a system of three transcendental equations
elif EFF_MASS_METHOD == 2:
    def proc_MEFF(line1,line2,line3,files):
        fn1=files[0]
        fn2=files[1]
        fn3=files[2]
        try:
            t1=float(re.search('t([0-9]+)',fn1).group(1))
            t2=float(re.search('t([0-9]+)',fn2).group(1))
            t3=float(re.search('t([0-9]+)',fn3).group(1))
        except:
            print "Bad blocks:",fn1,fn2,fn3
            print "must have t[0-9] in name, e.g. blk.t3"
            sys.exit(1)
        C1 = proc_line(line1,fn1)
        C2 = proc_line(line2,fn2)
        C3 = proc_line(line3,fn3)
        try:
            sol = nsolve((fit_func_3pt_sym(t1,[x,y,z])-C1, fit_func_3pt_sym(t2,[x,y,z])-C2,fit_func_3pt_sym(t3,[x,y,z])-C3), (x,y,z), START_PARAMS)
        except ValueError:
            print "Solution not within tolerance."
            print C1,fn1
            print C2,fn2
            print C3,fn3
            return 0
        if sol[1] < 0:
            print "***ERROR***"
            print "negative energy found:",sol[1]
            print fn1
            print fn2
            print fn3
            sys.exit(1)
        print "Found solution:",sol[1]
        return sol[1]
#fit to a function with one free parameter
#[ C(t+1)-C(t) ]/[ C(t+2)-C(t+1) ]
elif EFF_MASS_METHOD == 3 and FIT:
    def proc_MEFF(line1,line2,line3,files):
        fn1=files[0]
        fn2=files[1]
        fn3=files[2]
        C1 = proc_line(line1,fn1)
        C2 = proc_line(line2,fn2)
        C3 = proc_line(line3,fn3)
        arg = (C2-C1)/(C3-C2)
        if arg < 1:
            print "***ERROR***"
            print "argument to acosh in effective mass calc is less than 1:",arg
            print fn1
            print fn2
            print fn3
            sys.exit(1)
        #print 'solution =',sol
        return log(arg)
else:
    print "Bad method for finding the effective mass specified:", EFF_MASS_METHOD, "with fit set to", FIT
    sys.exit(1)


CSENT = object()
def proc_file(pifile, pjfile=CSENT,extra_pairs=[(None,None),(None,None)],reuse={}):
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
    avgone = 0
    count = 0
    #get the average of the lines in the ith file
    for linei,linei2,linei3 in izip(open(pifile),open(i2file),open(i3file)):
        if EFF_MASS:
            if not linei+linei2+linei3 in reuse:
                reuse[linei+linei2+linei3]=proc_MEFF(linei,linei2,linei3,ifiles)
            if reuse[linei+linei2+linei3]==0:
                reuse[linei+linei2+linei3] = START_PARAMS[1]
            avgone += reuse[linei+linei2+linei3]
        else:
            avgone += proc_line(linei,pifile)
        count += 1
    avgone /= count
    #uncorrelated fit
    if UNCORR == True and pjfile != pifile:
        return rets(coord=avgone, covar=0)
    avgtwo = 0
    counttest = 0
    #get the average of the lines in the jth file
    for linej,linej2,linej3 in izip(open(pjfile),open(j2file),open(j3file)):
        if EFF_MASS:
            if not linej+linej2+linej3 in reuse:
                reuse[linej+linej2+linej3]=proc_MEFF(linej,linej2,linej3,jfiles)
            if reuse[linej+linej2+linej3]==0:
                reuse[linej+linej2+linej3] = START_PARAMS[1]
            avgtwo += reuse[linej+linej2+linej3]
        else:
            avgtwo += proc_line(linej,pjfile)
        counttest += 1
    #check to make sure i,j have the same number of lines
    if not counttest == count:
        print "***ERROR***"
        print "Number of rows in paired files doesn't match"
        print count, counttest
        print "Offending files:", pifile, "and", pjfile
        sys.exit(1)
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
        coventry = prefactor*fsum([(reuse[li1+li2+li3]-avgone)*(reuse[lj1+lj2+lj3]-avgtwo) for li1, li2, li3, lj1, lj2, lj3 in izip(open(pifile), open(i2file), open(i3file), open(pjfile), open(j2file), open(j3file))])
    else:
        coventry = prefactor*fsum([(proc_line(l1,pifile)-avgone)*(proc_line(l2,pjfile)-avgtwo) for l1, l2 in izip(open(pifile), open(pjfile))])
    return rets(coord=avgone, covar=coventry)
