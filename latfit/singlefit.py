import sys
import os
from numpy.linalg import inv,det,tensorinv
from numpy import swapaxes,eye
from collections import namedtuple

#import global variables
from latfit.config import EIGCUT
from latfit.config import METHOD
from latfit.config import FIT
from latfit.config import GEVP
from latfit.config import START_PARAMS
from latfit.mathfun import chi_sq
#package modules
from latfit.procargs import procargs
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.extract.gevp import gevp_extract
from latfit.makemin.DOFerrchk import DOFerrchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr
from numpy import sqrt
import numpy as np

def maptomat(COV,dimops=1):
    if dimops==1:
        return COV
    else:
        Lt=len(COV)
        RETCOV=np.zeros((dimops*Lt,dimops*Lt))
        for i in range(Lt):
            for j in range(Lt):
                for a in range(dimops):
                    for b in range(dimops):
                        try:
                            RETCOV[i*dimops+a][j*dimops+b]=swapaxes(COV,1,2)[i][a][j][b]
                        except:
                            print("***ERROR***")
                            print("Dimension mismatch in mapping covariance tensor to matrix.")
                            print("Make sure time indices (i,j) and operator indices (a,b) are like COV[i][a][j][b].")
                            sys.exit(1)
        return RETCOV

def singlefit(INPUT, XMIN, XMAX, XSTEP):
    #test to see if file/folder exists
    inputexists(INPUT)

    ####process the file(s)
    if JACKKNIFE_FIT:
        pass
    else:
        if GEVP:
            COORDS, COV = gevp_extract(XMIN,XMAX,XSTEP)
        else:
            COORDS, COV, REUSE = extract(INPUT, XMIN, XMAX, XSTEP)
    #print(COORDS)

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
                COVINV = swapaxes(tensorinv(swapaxes(COV,1,2)),1,2)
        except:
            print("Covariance matrix is singular.")
            print("Check to make sure plot range does not contain a mirror image.")
            RETCOV=maptomat(COV,dimops)
            if dimops>1:
                try:
                    COVINV=inv(RETCOV)
                except:
                    print("Regular matrix inversion also failed")
                    count=0
                    for i in RETCOV:
                        print("row",count,"=",RETCOV[i])
                        print("column",count,"=",np.transpose(RETCOV)[i])
                        count+=1
            #print "determinant:",det(COV)
            sys.exit(1)
    #COVINV=eye(len(COORDS)*dimops)
    #COVINV.shape=(len(COORDS),len(COORDS),dimops,dimops)
    print("(Rough) scale of errors in data points = ", sqrt(COV[0][0]))

    #error handling for Degrees of Freedom <= 0 (it should be > 0).
    #number of points plotted = len(COV).
    #DOF = len(COV) - START_PARAMS
    DOFerrchk(len(COV))

    if FIT:
        #BFGS uses first derivatives of function
        #comment out options{...}, bounds for L-BFGS-B
        ###start minimizer
        RESULT_MIN=namedtuple('min',['x','fun','status'])
        RESULT_MIN.status=0
        if JACKKNIFE == 'Yes':
            num_configs=len(REUSE[XMIN])
            prefactor=(num_configs-1.0)/(1.0*num_configs)
            avg_min=np.zeros(len(START_PARAMS))
            avg_err=np.zeros(len(START_PARAMS))
            if FROZEN:
                covinv_jack=COVINV
            for config_num in range(num_configs):
                for time in np.arange(XMIN,XMAX+1,XSTEP):
                    coords_jack[time][1]=REUSE[time][config_num]
                if DOUBLE_JACKKNIFE:
                    for config_num_dj in range(num_configs):
                        if config_num_dj == config_num:
                            continue
                        pass
                result_min_jack = mkmin(covinv_jack, coords_jack)
                if result_min.status !=0:
                    RESULT_MIN.status=result_min_jack.status
                avg_min += result_min_jack.x
                avg_err += geterr(result_min_jack, covinv_jack, coords_jack)
            RESULT_MIN.x=avg_min/num_configs
            PARAM_ERR=prefactor*avg_err/num_configs
            RESULT_MIN.fun=chi_sq(RESULT_MIN.x,COVINV,COORDS)
        elif DOUBLE_JACKKNIFE:
            print("***ERROR***")
            print("Double jackknife not implemented yet.")
            sys.exit(1)
        else:
            RESULT_MIN = mkmin(COVINV, COORDS)
            ####compute errors 8ab, print results (not needed for plot part)
            PARAM_ERR = geterr(RESULT_MIN, COVINV, COORDS)
            #ERR_A0 = sqrt(2*HINV[0][0])
            #ERR_ENERGY = sqrt(2*HINV[1][1])
            #print "a0 = ", RESULT_MIN.x[0], "+/-", ERR_A0
            #print "energy = ", RESULT_MIN.x[1], "+/-", ERR_ENERGY
            ###plot result
            #plot the function and the data, with error bars
        return RESULT_MIN, PARAM_ERR, COORDS, COV
    else:
        return COORDS, COV
