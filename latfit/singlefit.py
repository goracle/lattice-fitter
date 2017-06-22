import sys
import os
from numpy.linalg import inv,det,tensorinv
from numpy import swapaxes,eye

#import global variables
from latfit.config import EIGCUT
from latfit.config import METHOD
from latfit.config import FIT
from latfit.config import GEVP
#package modules
from latfit.procargs import procargs
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.extract.gevp import gevp_extract
from latfit.makemin.DOFerrchk import DOFerrchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr
from numpy import sqrt

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
                            RETCOV[i*dimops+a][j*dimops+b]=COV[i][a][j][b]
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
    if GEVP:
        COORDS, COV = gevp_extract(XMIN,XMAX,XSTEP)
    else:
        COORDS, COV = extract(INPUT, XMIN, XMAX, XSTEP)
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
