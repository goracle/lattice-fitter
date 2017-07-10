import os
import sys
import numpy as np
import re
from collections import namedtuple
from linecache import getline
from scipy.linalg import eig
from warnings import warn
from itertools import chain

from latfit.mathfun.proc_meff import proc_meff
from latfit.mathfun.elim_jkconfigs import elim_jkconfigs
from latfit.extract.proc_line import proc_line

from latfit.config import UNCORR
from latfit.config import EFF_MASS

CSENT = object()
#def gevp_proc(ifiles,ifiles2,jfiles,jfiles2,time_arr,extra_pairs=[(CSENT,CSENT),(CSENT,CSENT)],reuse={}):
def gevp_proc(ifile_tup, jfile_tup, time_arr,reuse=None):

    if reuse is None:
        reuse = {}
    rets = namedtuple('rets', ['coord', 'covar', 'returnblk'])

    ifile_tup
    jfile_tup
    #setup global values
    dimops=len(ifiles[0])
    coventry=np.zeros((dimops,dimops))

    if EFF_MASS:
        #extra files for effective mass
        ifiles3=extra_pairs[0][0]
        jfiles3=extra_pairs[0][1]
        ifiles4=extra_pairs[1][0]
        jfiles4=extra_pairs[1][1]
        try:
            len(ifiles3)
            len(jfiles3)
        except:
            print("***ERROR***")
            print("Missing time adjacent file(s).")
            sys.exit(1)
        ifiles_chk=[ifiles]
        ifiles_chk.extend([extra_pairs[i][0] for i in range(2)])
        jfiles_chk=[jfiles]
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

    #do i
    try:
        num_configs=len(reuse['i'])
    except TypeError:
        num_configs=sum(1 for _ in open(ifiles_tup[0][0][0]))
        reuse['i']=np.zeros((num_configs,dimops))
        for config in range(num_configs):
            for opa in range(dimops):
                for opb in range(dimops):
                    CI_LHS[opa][opb]=proc_line(getline(ifiles[opa][opb],config+1),ifiles[opa][opb])
                    CI_RHS[opa][opb]=proc_line(getline(ifiles2[opa][opb],config+1),ifiles2[opa][opb])
                    if EFF_MASS:
                        CIP_LHS[opa][opb]=proc_line(getline(ifiles3[opa][opb],config+1),ifiles3[opa][opb])
                        CIPP_LHS[opa][opb]=proc_line(getline(ifiles4[opa][opb],config+1),ifiles4[opa][opb])
            if EFF_MASS:
                eigvalsI,eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,check_finite=False)
                eigvalsIP,eigvecsIP=eig(CIP_LHS,CI_RHS,overwrite_a=True,check_finite=False)
                eigvalsIPP,eigvecsIPP=eig(CIPP_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
                reuse['i'][config]=np.array([proc_meff(eigvalsI[op].real,eigvalsIP[op].real,eigvalsIPP[op].real) for op in range(dimops)])
            else:
                reuse['i'][config],eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
        if ELIM_JKCONF_LIST:
            reuse['i']=elim_jkconfigs(reuse['i'])
            num_configs=len(reuse['i'])
    avgI=np.mean(reuse['i'],axis=0)
    if gevp_proc.CONFIGSENT != 0:
        print("Number of configurations to average over:",num_configs)
        gevp_proc.CONFIGSENT = 0
    for test in avgI:
        if test.imag != 0:
            print("***ERROR***")
            print("GEVP has negative eigenvalues.")
            sys.exit(1)
    if np.array_equal(ifiles,jfiles):
        reuse['j']=reuse['i']
    try:
        if not num_configs==len(reuse['j']):
            print("***ERROR***")
            print("Number of configs not equal for i and j")
            print("GEVP covariance matrix entry:",time_arr)
            sys.exit(1)
    except:
        reuse['j']=np.zeros((num_configs,dimops))
        for config in range(num_configs):
            for opa in range(dimops):
                for opb in range(dimops):
                    CJ_LHS[opa][opb]=proc_line(getline(jfiles[opa][opb],config+1),jfiles[opa][opb])
                    CJ_RHS[opa][opb]=proc_line(getline(jfiles2[opa][opb],config+1),jfiles2[opa][opb])
                    if EFF_MASS:
                        CJP_LHS[opa][opb]=proc_line(getline(jfiles3[opa][opb],config+1),jfiles3[opa][opb])
                        CJPP_LHS[opa][opb]=proc_line(getline(jfiles4[opa][opb],config+1),jfiles4[opa][opb])
            #print(CJ_LHS,CJ_RHS)
            if EFF_MASS:
                eigvalsJ,eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,check_finite=False)
                #print(eigvalsJ,time_arr[2])
                eigvalsJP,eigvecsJP=eig(CJP_LHS,CJ_RHS,overwrite_a=True,check_finite=False)
                eigvalsJPP,eigvecsJPP=eig(CJPP_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
                reuse['j'][config]=np.array([proc_meff(eigvalsJ[op].real,eigvalsJP[op].real,eigvalsJPP[op].real,time_arr=time_arr) for op in range(dimops)])
            else:
                reuse['j'][config],eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
        if ELIM_JKCONF_LIST:
            reuse['j']=elim_jkconfigs(reuse['j'])
            num_configs=len(reuse['j'])
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

if EFF_MASS:
    def getblock(file_tup, reuse, ij_str, num_configs, dimops):
        """Given file tuple (for eff_mass),
        get block, store in reuse[ij_str]
        """
