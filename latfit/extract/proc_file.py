from collections import namedtuple
from math import fsum
from warnings import warn
import sys
import numpy as np

from latfit.mathfun.proc_MEFF import proc_MEFF
from latfit.extract.proc_line import proc_line

from latfit.config import UNCORR
from latfit.config import EFF_MASS
from latfit.config import START_PARAMS

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
        print("***ERROR***")
        print("Missing secondary file.")
        sys.exit(1)
    if EFF_MASS and (i2file == CSENT or j2file == CSENT):
        print("***ERROR***")
        print("Missing time adjacent file(s).")
        sys.exit(1)
    if EFF_MASS:
        ifiles=[pifile]
        ifiles.extend([extra_pairs[i][0] for i in range(2)])
        jfiles=[pjfile]
        jfiles.extend([extra_pairs[i][1] for i in range(2)])
    #within true cond. of test, we assume number of columns is one
    #get the average of the lines in the ith file
    try:
        count=len(reuse['i'])
    except:
        reuse['i']=np.array([])
        if not EFF_MASS:
            for linei in open(pifile):
                reuse['i']=np.append(resue['i'],proc_line(linei,pifile))
        else:
            for linei,linei2,linei3 in zip(open(pifile),open(i2file),open(i3file)):
                if not linei+linei2+linei3 in reuse:
                    reuse[linei+linei2+linei3]=proc_MEFF(linei,linei2,linei3,ifiles)
                if reuse[linei+linei2+linei3]==0:
                    reuse[linei+linei2+linei3] = START_PARAMS[1]
                reuse['i']=np.append(resue['i'],reuse[linei+linei2+linei3])
        count=len(reuse['i'])
    avgone=np.sum(reuse['i'],axis=0)/count
    #uncorrelated fit
    if UNCORR and pjfile != pifile:
        return rets(coord=avgone, covar=0)
    #get the average of the lines in the jth file
    try:
        counttest=len(reuse['j'])
    except:
        reuse['j']=np.array([])
        if not EFF_MASS:
            for linej in open(pjfile):
                reuse['j'] += np.append(resue['j'],proc_line(linej,pjfile))
        else:
            for linej,linej2,linej3 in zip(open(pjfile),open(j2file),open(j3file)):
                if not linej+linej2+linej3 in reuse:
                    reuse[linej+linej2+linej3]=proc_MEFF(linej,linej2,linej3,jfiles)
                if reuse[linej+linej2+linej3]==0:
                    reuse[linej+linej2+linej3] = START_PARAMS[1]
                reuse['j']=np.append(resue['j'],reuse[linej+linej2+linej3])
        counttest=len(reuse['j'])
    avgtwo=np.sum(reuse['j'],axis=0)/counttest
    #check to make sure i,j have the same number of lines
    if not counttest == count:
        print("***ERROR***")
        print("Number of rows in paired files doesn't match")
        print(count, counttest)
        print("Offending files:", pifile, "and", pjfile)
        sys.exit(1)
    else:
        if proc_file.CONFIGSENT != 0:
            print("Number of configurations to average over:",count)
            proc_file.CONFIGSENT = 0
    coventry = fsum([(reuse['i'][l1]-avgone)*(reuse['j'][l2]-avgtwo) for l1, l2 in zip(reuse['i'],reuse['j'])])
    return rets(coord=avgone, covar=coventry)
proc_file.CONFIGSENT = object()
