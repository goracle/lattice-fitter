import sys
import os
from numpy.linalg import inv,det,tensorinv
from numpy import eye,sqrt
from numpy import swapaxes as swap
from collections import namedtuple
import numpy as np
from warnings import warn

#package modules
from latfit.procargs import procargs
from latfit.extract.errcheck.inputexists import inputexists
from latfit.checks.maptomat import maptomat
from latfit.extract.extract import extract
from latfit.extract.gevp_extract import gevp_extract
from latfit.extract.inverse_jk import inverse_jk
from latfit.makemin.DOFerrchk import DOFerrchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr

#import global variables
from latfit.config import FIT
from latfit.config import JACKKNIFE
from latfit.config import JACKKNIFE_FIT
from latfit.config import GEVP
from latfit.config import START_PARAMS
from latfit.mathfun.chi_sq import chi_sq

def singlefit(INPUT, XMIN, XMAX, XSTEP):
    #test to see if file/folder exists
    inputexists(INPUT)

    ####process the file(s)
    if GEVP:
        COORDS, COV, REUSE = gevp_extract(XMIN,XMAX,XSTEP)
    else:
        COORDS, COV, REUSE = extract(INPUT, XMIN, XMAX, XSTEP)
    num_configs=len(REUSE[XMIN])
    #do this so REUSE goes from REUSE[time][config] to more convenient REUSE[config][time]
    time_range=np.arange(XMIN,XMAX+1,XSTEP)
    #REUSE=swap(REUSE,0,1)
    REUSE=np.array([[REUSE[time][config] for time in time_range] for config in range(num_configs)])
    if JACKKNIFE == 'YES':
        #applying jackknife correction of (count-1)^2
        warn("Applying jackknife correction to cov. matrix.")
        prefactor = (num_configs-1.0)/(1.0*num_configs)
    elif JACKKNIFE == 'NO':
        prefactor = (1.0)/((num_configs-1.0)*(1.0*num_configs))
    COV*=prefactor

    #error handling for Degrees of Freedom <= 0 (it should be > 0).
    #number of points plotted = len(COV).
    #DOF = len(COV) - START_PARAMS
    DOFerrchk(len(COV))

    ###we have data 6ab
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    if FIT:
        try:
            dimops=len(COV[0][0])
        except:
            dimops=1
        try:
            if dimops==1:
                COVINV = inv(COV)
            else:
                #swap axes, take inverse, swap back
                COVINV = swap(tensorinv(swap(COV,1,2)),1,2)
        except:
            print("Covariance matrix is singular.")
            print("Check to make sure plot range does not contain a mirror image.")
            RETCOV=maptomat(COV,dimops)
            if dimops>1:
                try:
                    COVINV=inv(RETCOV)
                except:
                    print("Regular matrix inversion also failed")
                    print("rows:")
                    for i in range(len(RETCOV)):
                        print(np.array2string(RETCOV[i],separator=', '))
                    print("columns:")
                    for i in range(len(RETCOV)):
                        print(np.array2string(np.transpose(RETCOV)[i],separator=', '))
                    print("det=",det(RETCOV))
            sys.exit(1)
    print("(Rough) scale of errors in data points = ", sqrt(COV[0][0]))

    if FIT:
        #comment out options{...}, bounds for L-BFGS-B
        ###start minimizer
        RESULT_MIN=namedtuple('min',['x','fun','status'])
        RESULT_MIN.status=0
        if JACKKNIFE_FIT:
            #one fit for every jackknife block (N fits for N configs)
            time_range=np.arange(XMIN,XMAX+1,XSTEP)
            coords_jack=np.copy(COORDS)
            min_arr=np.zeros((num_configs,len(START_PARAMS)))
            if JACKKNIFE_FIT=='FROZEN':
                covinv_jack=COVINV
            elif JACKKNIFE_FIT=='DOUBLE':
                REUSE_INV=inverse_jk(REUSE,time_range,num_configs)
            else:
                print("***ERROR***")
                print("Bad jackknife_fit value specified.")
                sys.exit(1)
            for config_num in range(num_configs):
                #if config_num>160: break #for debugging only
                if dimops>1:
                    for time in range(len(time_range)):
                        coords_jack[time,1]=REUSE[config_num][time]
                else:
                    coords_jack[:,1]=REUSE[config_num]
                if JACKKNIFE_FIT == 'DOUBLE':
                    cov_factor=np.delete(REUSE_INV,config_num,0)-REUSE[config_num]
                    try:
                        if dimops==1:
                            covinv_jack=inv(np.einsum('ai,aj->ij',cov_factor,cov_factor))
                        else:
                            covinv_jack=swap(tensorinv(np.einsum('aim,ajn->imjn',temp,temp)),1,2)
                    except:
                        print("Covariance matrix is singular in jackknife fit.")
                        print("Failing config_num=",config_num)
                        sys.exit(1)
                result_min_jack = mkmin(covinv_jack, coords_jack)
                if result_min_jack.status !=0:
                    RESULT_MIN.status=result_min_jack.status
                print("config",config_num,":",result_min_jack.x)
                min_arr[config_num]=result_min_jack.x
            RESULT_MIN.x=np.mean(min_arr,axis=0)
            PARAM_ERR=np.sqrt(prefactor*np.sum((min_arr-RESULT_MIN.x)**2,0))
            RESULT_MIN.fun=chi_sq(RESULT_MIN.x,COVINV,COORDS)
        else:
            RESULT_MIN = mkmin(COVINV, COORDS)
            ####compute errors 8ab, print results (not needed for plot part)
            PARAM_ERR = geterr(RESULT_MIN, COVINV, COORDS)
        return RESULT_MIN, PARAM_ERR, COORDS, COV
    else:
        return COORDS, COV
