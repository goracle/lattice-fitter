import sys
import os
from numpy.linalg import inv

#import global variables
from latfit.config import EIGCUT
from latfit.config import METHOD
#package modules
from latfit.procargs import procargs
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.makemin.DOFerrchk import DOFerrchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.geterr import geterr

def singlefit(INPUT, XMIN, XMAX, XSTEP):
    #test to see if file/folder exists
    inputexists(INPUT)

    ####process the file(s)
    COORDS, COV, DIMCOV = extract(INPUT, XMIN, XMAX, XSTEP)

    ###we have data 6ab
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    COVINV = inv(COV)
    print "(Rough) scale of errors in data points = ", COV[0][0]

    #error handling for Degrees of Freedom <= 0 (it should be > 0).
    #DIMCOV is number of points plotted.
    #DOF = DIMCOV - START_PARAMS
    DOFerrchk(DIMCOV)

    #BFGS uses first derivatives of function
    #comment out options{...}, bounds for L-BFGS-B
    ###start minimizer
    RESULT_MIN = mkmin(COVINV, COORDS, DIMCOV)

    ####compute errors 8ab, print results (not needed for plot part)
    PARAM_ERR = geterr(RESULT_MIN, COVINV, COORDS)
    #ERR_A0 = sqrt(2*HINV[0][0])
    #ERR_ENERGY = sqrt(2*HINV[1][1])
    #print "a0 = ", RESULT_MIN.x[0], "+/-", ERR_A0
    #print "energy = ", RESULT_MIN.x[1], "+/-", ERR_ENERGY

    ###plot result
    #plot the function and the data, with error bars
    return RESULT_MIN, PARAM_ERR, COORDS, COV
