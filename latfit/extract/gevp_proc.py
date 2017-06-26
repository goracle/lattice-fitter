import os
import sys
import numpy as np
import re
from collections import namedtuple
from linecache import getline
from scipy.linalg import eig
from warnings import warn

from latfit.config import UNCORR

def gevp_proc(IFILES,IFILES2,IFILES3,JFILES,JFILES2,JFILES3,TIME_ARR,reuse={}):
    #setup global values
    rets = namedtuple('rets', ['coord', 'covar'])
    dimops=len(IFILES)
    coventry=np.zeros((dimops,dimops))

    #C(t)v=Eigval*C(t_0)v
    CI_LHS=np.zeros((dimops,dimops),dtype=float)
    CI_RHS=np.zeros((dimops,dimops),dtype=float)
    CJ_LHS=np.zeros((dimops,dimops),dtype=float)
    CJ_RHS=np.zeros((dimops,dimops),dtype=float)

    #C(t+1)v=Eigval*C(t_0)v
    CIP_LHS=np.zeros((dimops,dimops),dtype=float)
    CJP_LHS=np.zeros((dimops,dimops),dtype=float)

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
                    CIP_LHS[opa][opb]=proc_line(getline(IFILES3[opa][opb],config+1),IFILES3[opa][opb])
            eigvalsI,eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,check_finite=False)
            eigvalsIP,eigvecsIP=eig(CIP_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
            energiesI=np.log(eigvalsI)-np.log(eigvalsIP)
            reuse['i'][config]=energiesI
    avgI=np.sum(reuse['i'],axis=0)/num_configs
    if UNCORR and (IFILES!=JFILES).all():
        return rets(coord=avgI, covar=coventry)
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
                    CJP_LHS[opa][opb]=proc_line(getline(JFILES3[opa][opb],config+1),JFILES3[opa][opb])
            eigvalsJ,eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,check_finite=False)
            eigvalsJP,eigvecsJP=eig(CJP_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
            energiesJ=np.log(eigvalsJ)-np.log(eigvalsJP)
            reuse['j'][config]=energiesJ
    avgJ=np.sum(reuse['j'],axis=0)/num_configs
    for i in range(dimops):
        if avgI[i].imag != 0 or avgJ[i].imag!=0:
            print("***ERROR***")
            print("GEVP has negative eigenvalues.")
            sys.exit(1)
    if UNCORR:
        for a in range(dimops):
            coventry[a][a]=np.sum([(avgI[a]-reuse['i'][k][a])*(avgJ[a]-reuse['j'][k][a]) for k in range(num_configs)],axis=0)
    else:
        coventry=np.sum([np.outer((avgI-reuse['i'][k]),(avgJ-reuse['j'][k])) for k in range(num_configs)],axis=0)
    return rets(coord=avgI, covar=coventry)
