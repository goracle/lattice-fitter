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


import matplotlib.pyplot as plt
import sys
import getopt
import numpy as np
from numpy.linalg import inv
from collections import defaultdict
import os, re
from math import fsum
from itertools import izip
from collections import namedtuple
from scipy.optimize import minimize
from math import exp, sqrt
import numdifftools as nd
from matplotlib.backends.backend_pdf import PdfPages

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:", ["ifolder=", "help", "ifile="])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print "Invalid or missing argument."
        main(["-h"])
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            print "usage(2):", sys.argv[0]
            print "-f <folder of blocks to be averaged>"
            sys.exit()
        elif opt in "-i" "--ifile":
            return arg
        elif opt in "-f" "--ifolder":
            return arg

def tree():
    """Return a multidimensional dict"""
    return defaultdict(tree)

#type of function: pade, used for certain types of qcd theories
#be sure to put triple quotes back in afterwards for the docstring
#"""def fit_func(qsq, a_0, a_1, b_1):
#Give result of function computed to fit the data given in <inputfile>
#(See main(argv))
#return qsq*(a_0+a_1/(b_1+qsq))
#"""

#simple exponential
def fit_func(ctime, trial_params):
    """Give result of function computed to fit the data given"""
    #amplitude is trial_params[0], energy is trial_params[1]
    return trial_params[0]*exp(-ctime*trial_params[1])

def chi_sq(trial_params, covinv, coords):
    """Compute chi^2 given a set of trial parameters,
    the inverse covariance matrix, and the x-y coordinates to fit.
    """
    return fsum([(coords[outer][1]-
                  fit_func(coords[outer][0], trial_params))*
                 covinv[outer][inner]*(coords[inner][1]-
                                       fit_func(coords[inner][0],
                                                trial_params))
                 for outer in range(len(COORDS))
                 for inner in range(len(COORDS))])

#delete me
#CHI_SQ = fsum([(COORDS[i][1]-fit_func(COORDS[i][0], trial_params))*
#                   COVINV[i][j]*(COORDS[j][1]-fit_func(COORDS[j][0],
#                                                       a_0, energy))
#                  for i in range(len(COORDS))
#                  for j in range(len(COORDS))])

def proc_folder(folder, ctime):
    """Process folder where blocks to be averaged are stored.
    Return file corresponding to current ensemble (lattice time slice).
    Assumes file is <anything>t<time><anything>
    Assumes only 1 valid file per match, e.g. ...t3... doesn't happen more
    than once
    """
    #build regex as a string
    my_regex = r"t" + str(ctime)
    temp1 = ""
    temp2 = ""
    for root, dirs, files in os.walk(folder):
        for name in files:
            if re.search(my_regex, name):
                return name
            else:
                temp1 = root
                temp2 = dirs
    print temp1
    print temp2
    print folder
    print "***ERROR***"
    print "Can't find file corresponding to time = ", ctime
    sys.exit(1)

def simple_proc_file(kfile):
    """Process file with precomputed covariance matrix."""
    cdict = tree()
    rets = namedtuple('rets', ['coord', 'covar'])
    proccoords = []
    with open(kfile) as opensimp:
        for line in opensimp:
            try:
                cols = [float(p) for p in line.split()]
            except ValueError:
                print "ignored line: '", line, "'"
                continue
            if len(cols) == 2:
                proccoords.append([cols[0], cols[1]])
                #two columns mean coordinate section, 3 covariance section
            elif len(cols) == 3:
                cdict[cols[0]][cols[1]] = cols[2]
            else:
                print "***Error***"
                print "mangled file:"
                print IFILE
                sys.exit(1)
        ccov = [[cdict[proccoords[ci][0]][proccoords[cj][0]]
                 for ci in range(len(proccoords))]
                for cj in range(len(proccoords))]
        return rets(coord=proccoords, covar=ccov)
    sys.exit(1)

CSENT = object()
def proc_file(pifile, pjfile=CSENT):
    """Process the current file.
    Return covariance matrix entry I,indexj in the case of multi-file structure.
    Return the covariance matrix for single file.
    """
    #initialize return value named tuple. in other words:
    #create a type of object, rets, to hold return values
    #instantiate it with return values, then return that instantiation
    rets = namedtuple('rets', ['coord', 'covar'])
    if pjfile == CSENT:
        print "***ERROR***"
        print "Missing secondary file."
        sys.exit(1)
    #within true cond. of test, we assume number of columns is one
    with open(pifile) as ithfile:
        avgone = 0
        avgtwo = 0
        count = 0
        for line in ithfile:
            avgone += float(line)
            count += 1
        avgone /= count
        with open(pjfile) as jthfile:
            counttest = 0
            for line in jthfile:
                avgtwo += float(line)
                counttest += 1
            if not counttest == count:
                print "***ERROR***"
                print "Number of rows in paired files doesn't match"
                print count, counttest
                print "Offending files:", pifile, "and", pjfile
                sys.exit(1)
            else:
                avgtwo /= count
            #cov[I][indexj]=return value for folder-style
            #applying jackknife correction of (count-1)^2
            coventry = (count-1)/(1.0*count)*fsum([
                (float(l1)-avgone)*(float(l2)-avgtwo)
                for l1, l2 in izip(open(pifile), open(pjfile))])
            return rets(coord=avgone,
                        covar=coventry)
    print "***Unexpted Error***"
    print "If you\'re seeing this program has a bug that needs fixing"
    sys.exit(1)
    #delete me (if working)
            #append at position i,j to the covariance matrix entry
        #store precomputed covariance matrix (if it exists)
        #in convenient form
        #try:
        #COV = [[CDICT[PROCCOORDS[i][0]][PROCCOORDS[j][0]]
        #        for i in range(len(PROCCOORDS))]
        #       for j in range(len(PROCCOORDS))]
    #delete me end (if working)

#main part
if __name__ == "__main__":
    #re.match(r'',input part from file) // ignore this
    INPUT = main(sys.argv[1:])
    #error handling
    #test to see if file/folder exists
    if not (os.path.isfile(INPUT) or os.path.isdir(INPUT)):
        print "File:", INPUT, "not found"
        print "Folder:", INPUT, "also not found."
        main(["h"])
    #test to see if input is file, then process the file
    #result is returnd as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar'])
    if os.path.isfile(INPUT):
        RESRET = simple_proc_file(INPUT)
        COV = RESRET.covar
        COORDS = RESRET.coord
    #test if directory
    #then find out domain of files to process
    elif os.path.isdir(INPUT):
        print "Now, input valid time domain (abscissa)."
        print "time min<=t<=time max"
        print "time min="
        TMIN = int(raw_input())
        print "time max="
        TMAX = int(raw_input())
        #now process individual files
        #error handling, test to see if time value goes out of range,
        #i.e. if data isn't available to match the requested time domain
        #i,j are new indices, shifting TMIN to the origin
        #j = 0 # initialized below
        i = 0
        #DIMCOV is dimensions of the covariance matrix
        DIMCOV = (TMAX+1)-TMIN
        #cov is the covariance matrix
        COV = [[[0] for k in range(DIMCOV)] for j in range(DIMCOV)]
        #COORDS are the coordinates to be plotted.
        #the ith point with the jth value
        COORDS = [[[0] for k in range(2)] for j in range(DIMCOV)]
        for time in range(TMIN, TMAX+1):
            COORDS[i][0] = time
            j = 0
            for time2 in range(TMIN, TMAX+1):
                IFILE = proc_folder(INPUT, time)
                JFILE = proc_folder(INPUT, time2)
                IFILE = INPUT + "/" + IFILE
                JFILE = INPUT + "/" + JFILE
                try:
                    TRIAL = open(IFILE, "r")
                    TRIAL2 = open(JFILE, "r")
                except TypeError:
                    STR1 = "Either time range is invalid,"
                    print STR1, "or folder is invalid."
                    print "Double check contents of folder."
                    print "Offending file(s):"
                    print IFILE
                    print JFILE
                    sys.exit(1)
                RESRET = proc_file(IFILE, JFILE)
                #fill in the covariance matrix
                COV[i][j] = RESRET.covar
                #only store coordinates once.  each file is read many times
                if j == 0:
                    COORDS[i][1] = RESRET.coord
                j += 1
            i += 1
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    COVINV = inv(COV)
    print COVINV[2][3]
    #minimize chi squared

    #todo:generalize this
    START_A_0 = 6
    START_ENERGY = 0.7
    START_PARAMS = [START_A_0, START_ENERGY]
    BINDS = ((None, None), (0, None))
    #end todo: generalize this
    #minimize chi squared
    #plan (delete me later):
    #compute derivatives of each parameter of the fit function
    #call sage to solve system of equationss???
    #"""CHI_SQ = fsum([(COORDS[i][1]-fit_func(COORDS[i][0], trial_params))*
    #               COVINV[i][j]*(COORDS[j][1]-fit_func(COORDS[j][0],
    #                                                   a_0, energy))
    #              for i in range(len(COORDS))
    #              for j in range(len(COORDS))])"""

    #def chi_sq(COVINV, trial_params, COORDS):
    #BFGS uses first derivatives of function
    RESULT_MIN = minimize(chi_sq, [1, 1], (COVINV, COORDS),
                          bounds=BINDS, method='L-BFGS-B',
                          options={'disp': True})
    print "minimized params = ", RESULT_MIN.x
    print "successfully minimized = ", RESULT_MIN.success
    print "status of optimizer = ", RESULT_MIN.status
    print "message of optimizer = ", RESULT_MIN.message
    print "number of iterations = ", RESULT_MIN.nit
    print "chi^2 minimized = ", RESULT_MIN.fun
    print "chi^2 reduced = ", RESULT_MIN.fun/(DIMCOV-len(START_PARAMS))
    #compute hessian matrix
    HFUNC = lambda xrray: chi_sq(xrray, COVINV, COORDS)
    HFUN = nd.Hessian(HFUNC)
    #compute hessian inverse
    HINV = inv(HFUN(RESULT_MIN.x))
    #HESSINV = inv(HESS)
    #compute errors in fit parameters
    ERR_A0 = sqrt(2*HINV[0][0])
    ERR_ENERGY = sqrt(2*HINV[1][1])
    print "a0 = ", RESULT_MIN.x[0], "+/-", ERR_A0
    print "energy = ", RESULT_MIN.x[1], "+/-", ERR_ENERGY
    #plot the function and the data, with error bars
    with PdfPages('foo.pdf') as pdf:
        XCOORD = np.arange(TMIN, TMAX+1, 1)
        YCOORD = [COORDS[i][1] for i in range(len(COORDS))]
        ER2 = np.array([COV[i][i] for i in range(len(COORDS))])
        plt.errorbar(XCOORD, YCOORD, yerr=ER2, linestyle='None')
        XFIT = np.arange(TMIN-1, TMAX+1, 0.1)
        YFIT = np.array([fit_func(XFIT[i], RESULT_MIN.x)
                         for i in range(len(XFIT))])
        plt.plot(XFIT, YFIT)
        plt.xlim([TMIN-1, TMAX+1])
        #magic numbers for the problem you're solving
        plt.ylim([0, 0.1])
        #add labels, more magic numbers
        plt.title('Some Correlation function vs. time')
        STRIKE1 = "Energy = " + str(RESULT_MIN.x[1]) + "+/-" + str(
            ERR_ENERGY)
        STRIKE2 = "Amplitude = " + str(RESULT_MIN.x[0]) + "+/-" + str(
            ERR_A0)
        X_POS_OF_FIT_RESULTS = 8
        plt.text(X_POS_OF_FIT_RESULTS, 0.07, STRIKE1)
        plt.text(X_POS_OF_FIT_RESULTS, .065, STRIKE2)
        plt.xlabel('time (?)')
        plt.ylabel('the function')
        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    #extraneous notes below
    sys.exit()
