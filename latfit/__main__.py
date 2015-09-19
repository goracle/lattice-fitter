#!/usr/bin/env python

"""Fit function to data.
Compute chi^2 and errors.
Plot fit with error bars.
Save result to pdf.
usage note: MAKE SURE YOU SET THE Y LIMITS of your plot by hand!
usage note(2): MAKE SURE as well that you correct the other "magic"
parts of the graph routine
"""

#install pip2
#probably needs to be refactored for python3...
#then sudo pip install numdifftools

from collections import namedtuple
import sys
import os
from numpy.linalg import inv

#import global variables
from latfit.globs import EIGCUT
from latfit.globs import METHOD
#package modules
from latfit.procargs import procargs
from latfit.extract.errcheck.xlim_err import xlim_err
from latfit.extract.errcheck.xstep_err import xstep_err
from latfit.extract.errcheck.inputexists import inputexists
from latfit.extract.extract import extract
from latfit.makemin.DOFerrchk import DOFerrchk
from latfit.makemin.mkmin import mkmin
from latfit.finalout.printerr import printerr
from latfit.finalout.mkplot import mkplot

def main():
    ####set up 1ab
    OPTIONS = namedtuple('ops', ['xmin', 'xmax', 'xstep'])

    
    ###error processing, parameter extractions
    INPUT, OPTIONS = procargs(sys.argv[1:])
    XMIN, XMAX = xlim_err(OPTIONS.xmin, OPTIONS.xmax)
    XSTEP = xstep_err(OPTIONS.xstep, INPUT)
    
    #test to see if file/folder exists
    inputexists(INPUT)

    ####process the file(s)
    COORDS, COV, DIMCOV = extract(INPUT, XMIN, XMAX)

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
    printerr(RESULT_MIN, COVINV, COORDS)
    #ERR_A0 = sqrt(2*HINV[0][0])
    #ERR_ENERGY = sqrt(2*HINV[1][1])
    #print "a0 = ", RESULT_MIN.x[0], "+/-", ERR_A0
    #print "energy = ", RESULT_MIN.x[1], "+/-", ERR_ENERGY

    ###plot result
    #plot the function and the data, with error bars
    mkplot(COORDS, COV, RESULT_MIN)
    sys.exit(0)

if __name__ == "__main__":
    main()
