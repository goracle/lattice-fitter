import os
import sys
import numpy as np
import re
from numpy import ceil,floor
from collections import namedtuple
from linecache import getline
from scipy.linalg import eig
from warnings import warn

from latfit.extract.extract import pre_proc_file
from latfit.extract.proc_folder import proc_folder
from latfit.extract.proc_file import proc_line

from latfit.config import GEVP_DIRS
from latfit.config import JACKKNIFE

def gevp_extract(XMIN,XMAX,XSTEP):
    i = 0
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar'])
    #dimcov is dimensions of the covariance matrix
    dimcov = int((XMAX-XMIN)/XSTEP+1)
    #dimops is the dimension of the correlator matrix
    dimops = len(GEVP_DIRS)
    #cov is the covariance matrix
    COV = np.zeros((dimcov,dimcov,dimops,dimops),dtype=np.complex128)
    #COORDS are the coordinates to be plotted.
    #the ith point with the jth value
    COORDS = np.zeros((dimcov,2),dtype=object)
    #return COORDS, COV
    for timei in np.arange(XMIN, XMAX+1, XSTEP):
        timei2=ceil(float(timei)/2.0/XSTEP)*XSTEP if ceil(float(timei)/2.0)!=timei else max(floor(float(timei)/2.0/XSTEP)*XSTEP,XMIN)
        #set the times coordinate
        COORDS[i][0] = timei
        j=0
        #extract files
        IFILES = [[proc_folder(GEVP_DIRS[op1][op2],timei) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES2 = [[proc_folder(GEVP_DIRS[op1][op2],timei2) for op1 in range(dimops)] for op2 in range(dimops)]
        #check for errors
        IFILES = [[pre_proc_file(IFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES2 = [[pre_proc_file(IFILES2[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        for timej in np.arange(XMIN, XMAX+1, XSTEP):
            timej2=ceil(float(timej)/2.0/XSTEP)*XSTEP if ceil(float(timej)/2.0)!=timej else max(floor(float(timej)/2.0/XSTEP)*XSTEP,XMIN)
            #extract files
            JFILES = [[proc_folder(GEVP_DIRS[op1][op2],timej) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES2 = [[proc_folder(GEVP_DIRS[op1][op2],timej2) for op1 in range(dimops)] for op2 in range(dimops)]
            #check for errors
            JFILES = [[pre_proc_file(JFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES2 = [[pre_proc_file(JFILES2[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            RESRET = gevp_proc(IFILES,IFILES2,JFILES,JFILES2)
            COV[i][j] = RESRET.covar
            #only store coordinates once.  each file is read many times
            if j == 0:
                COORDS[i][1] = RESRET.coord
            j+=1
        i+=1
    return COORDS, COV

def gevp_proc(IFILES,IFILES2,JFILES,JFILES2):
    rets = namedtuple('rets', ['coord', 'covar'])
    #find the averages
    num_configs=sum(1 for _ in open(IFILES[0][0]))
    if JACKKNIFE == 'YES':
        warn("Applying jackknife correction to cov. matrix.")
        prefactor=((num_configs-1.0)*1.0)/(num_configs*1.0)
    elif JACKKNIFE == 'NO':
        prefactor = (1.0)/((num_configs-1.0)*(1.0*num_configs))
    dimops=len(IFILES)
    avgone=np.zeros(dimops)
    avgI=0
    avgJ=0
    CI_LHS=np.zeros((dimops,dimops),dtype=complex)
    CI_RHS=np.zeros((dimops,dimops),dtype=complex)
    CJ_LHS=np.zeros((dimops,dimops),dtype=complex)
    CJ_RHS=np.zeros((dimops,dimops),dtype=complex)
    eig_arr=np.zeros((num_configs),dtype=object)
    warn("Taking the real (first column).")
    for config in range(num_configs):
        for opa in range(dimops):
            for opb in range(dimops):
                CI_LHS[opa][opb]=complex(getline(IFILES2[opa][opb],config+1).split()[0])
                CI_RHS[opa][opb]=complex(getline(IFILES[opa][opb],config+1).split()[0])
                CJ_LHS[opa][opb]=complex(getline(JFILES2[opa][opb],config+1).split()[0])
                CJ_RHS[opa][opb]=complex(getline(JFILES[opa][opb],config+1).split()[0])
        eigvalsI,eigvecsI=eig(CI_LHS,CI_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
        eigvalsJ,eigvecsJ=eig(CJ_LHS,CJ_RHS,overwrite_a=True,overwrite_b=True,check_finite=False)
        avgI+=eigvalsI
        avgJ+=eigvalsJ
        eig_arr[config]=np.array([eigvalsI,eigvalsJ])
    avgI/=num_configs
    avgJ/=num_configs
    coventry=np.zeros((dimops,dimops))
    coventry=np.sum([np.outer((avgI-eig_arr[k][0]),(avgJ-eig_arr[k][1])) for k in range(num_configs)])
    return rets(coord=avgI, covar=prefactor*coventry)
