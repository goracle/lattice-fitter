import os
import sys
import numpy as np
import re
from collections import namedtuple
from linecache import getline
from scipy.linalg import eig
from warnings import warn
from itertools import chain

from latfit.mathfun.proc_MEFF import proc_MEFF
from latfit.extract.proc_line import proc_line

from latfit.config import UNCORR
from latfit.config import EFF_MASS

CSENT = object()
def gevp_proc(IFILES,IFILES2,JFILES,JFILES2,TIME_ARR,extra_pairs=[(CSENT,CSENT),(CSENT,CSENT)],reuse={}):
    #setup global values
    rets = namedtuple('rets', ['coord', 'covar', 'returnblk'])
    dimops=len(IFILES)
    coventry=np.zeros((dimops,dimops))

    if EFF_MASS:
        #extra files for effective mass
        IFILES3=extra_pairs[0][0]
        JFILES3=extra_pairs[0][1]
        IFILES4=extra_pairs[1][0]
        JFILES4=extra_pairs[1][1]
        try:
            len(IFILES3)
            len(JFILES3)
        except:
            print("***ERROR***")
            print("Missing time adjacent file(s).")
            sys.exit(1)
        ifiles_chk=[IFILES]
        ifiles_chk.extend([extra_pairs[i][0] for i in range(2)])
        jfiles_chk=[JFILES]
        jfiles_chk.extend([extra_pairs[i][1] for i in range(2)])
        #C(t+1)v=Eigval*C(t_0)v
        CIP_LHS=np.zeros((dimops,dimops),dtype=float)
        CJP_LHS=np.zeros((dimops,dimops),dtype=float)
        CIPP_LHS=np.zeros((dimops,dimops),dtype=float)
        CJPP_LHS=np.zeros((dimops,dimops),dtype=float)

    #C(t)v=Eigval*C(t_0)v
    CI_LHS=np.zeros((dimops,dimops),dtype=float)
    CI_RHS=np.zeros((dimops,dimops),dtype=float)
    CJ_LHS=np.zeros((dimops,dimops),dtype=float)
    CJ_RHS=np.zeros((dimops,dimops),dtype=float)

    try:
        num_configs=len(reuse['i'])
    except:
        num_configs=sum(1 for _ in open(IFILES[0][0]))
        reuse['i']=np.zeros((num_configs,dimops))
        for config in range(num_configs):
            for opa in range(dimops):
                for opb in range(dimops):
                    CI_LHS[opa][opb]=proc_line(getline(IFILES[opa][opb],config+1),IFILES[opa][opb])
                    CI_RHS[opa][opb]=proc_line(getline(IFILES2[opa][opb],config+1),IFILES2[opa][opb])
                    if EFF_MASS:
                        CIP_LHS[opa][opb]=proc_line(getline(IFILES3[opa][opb],config+1),IFILES3[opa][opb])
                        CIPP_LHS[opa][opb]=proc_line(getline(IFILES4[opa][opb],config+1),IFILES4[opa][opb])
            if EFF_MASS:
                eigvalsI,eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,check_finite=False)
                eigvalsIP,eigvecsIP=eig(CIP_LHS,CI_RHS,overwrite_a=True,check_finite=False)
                eigvalsIPP,eigvecsIPP=eig(CIPP_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
                reuse['i'][config]=np.array([proc_MEFF(eigvalsI[op].real,eigvalsIP[op].real,eigvalsIPP[op].real) for op in range(dimops)])
            else:
                reuse['i'][config],eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
    avgI=np.mean(reuse['i'],axis=0)
    if gevp_proc.CONFIGSENT != 0:
        print("Number of configurations to average over:",num_configs)
        gevp_proc.CONFIGSENT = 0
    for test in avgI:
        if test.imag != 0:
            print("***ERROR***")
            print("GEVP has negative eigenvalues.")
            sys.exit(1)
    if np.array_equal(IFILES,JFILES):
        reuse['j']=reuse['i']
    try:
        if not num_configs==len(reuse['j']):
            print("***ERROR***")
            print("Number of configs not equal for i and j")
            print("GEVP covariance matrix entry:",TIME_ARR)
            sys.exit(1)
    except:
        reuse['j']=np.zeros((num_configs,dimops))
        for config in range(num_configs):
            for opa in range(dimops):
                for opb in range(dimops):
                    CJ_LHS[opa][opb]=proc_line(getline(JFILES[opa][opb],config+1),JFILES[opa][opb])
                    CJ_RHS[opa][opb]=proc_line(getline(JFILES2[opa][opb],config+1),JFILES2[opa][opb])
                    if EFF_MASS:
                        CJP_LHS[opa][opb]=proc_line(getline(JFILES3[opa][opb],config+1),JFILES3[opa][opb])
                        CJPP_LHS[opa][opb]=proc_line(getline(JFILES4[opa][opb],config+1),JFILES4[opa][opb])
            #print(CJ_LHS,CJ_RHS)
            if EFF_MASS:
                eigvalsJ,eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,check_finite=False)
                #print(eigvalsJ,TIME_ARR[2])
                eigvalsJP,eigvecsJP=eig(CJP_LHS,CJ_RHS,overwrite_a=True,check_finite=False)
                eigvalsJPP,eigvecsJPP=eig(CJPP_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
                reuse['j'][config]=np.array([proc_MEFF(eigvalsJ[op].real,eigvalsJP[op].real,eigvalsJPP[op].real) for op in range(dimops)])
            else:
                reuse['j'][config],eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
    avgJ=np.mean(reuse['j'],axis=0)
    for test in avgJ:
        if test.imag != 0:
            print("***ERROR***")
            print("GEVP has negative eigenvalues.")
            sys.exit(1)
    if UNCORR:
        for a in range(dimops):
            coventry[a][a]=np.sum([(avgI[a]-reuse['i'][k][a])*(avgI[a]-reuse['i'][k][a]) for k in range(num_configs)],axis=0)
    else:
        coventry=np.sum([np.outer((avgI-reuse['i'][k]),(avgJ-reuse['j'][k])) for k in range(num_configs)],axis=0)
    return rets(coord=avgI, covar=coventry,returnblk=reuse['j'])
gevp_proc.CONFIGSENT = object()
