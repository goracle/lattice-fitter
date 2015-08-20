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

from __future__ import division
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
from numpy.linalg import cholesky as posdefexcept
########SOURCE CODE NAVIGATION#######
########for the part starting with
#if __name__ == "__main__":
####set up 1ab
####error handling 2ab
####process the files 3ab
####input time domain for folders 4ab
####process individual files in dir 5ab
####we have data 6ab
####minimize 7ab
####compute errors 8ab
####plot result 9ab

#function definitions
def main(argv):
    """Parse the command line.
    Give usage information or set the input file.
    """
    try:
        opts = getopt.getopt(argv, "f:hi:s:",
                             ["ifolder=", "help", "ifile=",
                              "switch=", "xmin=", "xmax=", 'xstep=',
                              'nextra='])[0]
        if opts == []:
            raise NameError("NoArgs")
    except (getopt.GetoptError, NameError):
        print "Invalid or missing argument."
        main(["-h"])
    switch = -1
    cxmin = object()
    cxmax = object()
    cxstep = object()
    cnextra = object()
    options = namedtuple('ops', ['xmin', 'xmax', 'xstep', 'nextra'])
    #Get environment variables from command line.
    for opt, arg in opts:
        if opt == '-h':
            print "usage:", sys.argv[0], "-i <inputfile>"
            print "usage(2):", sys.argv[0]
            print "-f <folder of blocks to be averaged>"
            print "Required aruments:"
            print "-s <fit function to use>"
            print "fit function options are:"
            print "0: Pade"
            print "Optional argument for Pade:"
            print "--nextra=<number of extra arguments/2>"
            print "E.g., if you enter --nextra=3, 6 additional fit"
            print "parameters will be used."
            print "1: Exponential"
            print "Optional Arguments"
            print "--xmin=<domain lower bound>"
            print "--xmax=<domain upper bound>"
            print "--xstep=<domain step size>"
            sys.exit()
        if opt in "-s" "--switch":
            switch = arg
        if opt in "--xmin":
            cxmin = arg
        if opt in "--xstep":
            cxstep = arg
        if opt in "--xmax":
            cxmax = arg
        if opt in "--nextra":
            cnextra = arg
    if not switch in set(['0', '1']):
        print "You need to pick a fit function."
        main(["-h"])
    #exiting loop
    for opt, arg in opts:
        if opt in "-i" "--ifile" "-f" "--ifolder":
            return arg, switch, options(xmin=cxmin, xmax=cxmax,
                                        xstep=cxstep, nextra=cnextra)

def tree():
    """Return a multidimensional dict"""
    return defaultdict(tree)

def fit_func(ctime, trial_params, switch):
    """Give result of function computed to fit the data given in <inputfile>
    (See main(argv))
    """
    if switch == '0':
        #pade function
        return trial_params[0]+ctime*(trial_params[1]+trial_params[2]/(
            trial_params[3]+ctime)+fsum([trial_params[i]/(trial_params[i+1]+ctime) for i in np.arange(4, len(trial_params), 2)]))
    #return (trial_params[0]+trial_params[1]*ctime+
    #trial_params[3]*ctime*ctime)/(
    #           1+trial_params[2]*ctime)
    if switch == '1':
        #simple exponential
        return trial_params[0]*exp(-ctime*trial_params[1])

def chi_sq(trial_params, covinv, coords, switch):
    """Compute chi^2 given a set of trial parameters,
    the inverse covariance matrix, and the x-y coordinates to fit.
    """
    return fsum([(coords[outer][1]-
                  fit_func(coords[outer][0], trial_params, switch))*
                 covinv[outer][inner]*(coords[inner][1]-
                                       fit_func(coords[inner][0],
                                                trial_params, switch))
                 for outer in range(len(coords))
                 for inner in range(len(coords))])

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
    than once.
    Both the int and float versions of ctime are treated the same.
    """
    #build regex as a string
    my_regex = r"t" + str(ctime)
    flag = 0
    flag2 = 0
    temp4 = object()
    retname = temp4
    if int(str(ctime-int(ctime))[2:]) == 0:
        my_regex2 = r"t" + str(int(ctime))
        flag = 1
    temp1 = ""
    temp2 = ""
    for root, dirs, files in os.walk(folder):
        for name in files:
            #logic: if the search matches either int or float ctime
            if re.search(my_regex, name) or (
                    re.search(my_regex2, name) and flag == 1):
                #logic: if we found another matching file in
                #the folder already
                if not retname == temp4:
                    flag2 = 1
                #logic: else save the file name to return after folder walk
                retname = name
            else:
                temp1 = root
                temp2 = dirs
    #logic: if we found at least one match
    if not retname == temp4:
        if flag2 == 1:
            print "***ERROR***"
            print "File name collision."
            print "Two files match the search."
            print "Amend your file names."
            sys.exit(1)
        return retname
    print temp1
    print temp2
    print folder
    print "***ERROR***"
    print "Can't find file corresponding to time = ", ctime
    sys.exit(1)

def simple_proc_file(kfile, cxmin, cxmax):
    """Process file with precomputed covariance matrix."""
    cdict = tree()
    rets = namedtuple('rets', ['coord', 'covar', 'numblocks'])
    proccoords = []
    with open(kfile) as opensimp:
        for line in opensimp:
            try:
                cols = [float(p) for p in line.split()]
            except ValueError:
                print "ignored line: '", line, "'"
                continue
            if len(cols) == 2 and cxmin <= cols[0] <= cxmax:
                #only store coordinates in the valid range
                proccoords.append([cols[0], cols[1]])
                #two columns mean coordinate section, 3 covariance section
            elif len(cols) == 3:
                cdict[cols[0]][cols[1]] = cols[2]
            elif (not len(cols) == 2) and (not len(cols(3))):
                print "***Error***"
                print "mangled file:"
                print kfile
                print "Expecting either two or three numbers per line."
                print len(cols), "found instead."
                sys.exit(1)
        ccov = [[cdict[proccoords[ci][0]][proccoords[cj][0]]
                 for ci in range(len(proccoords))]
                for cj in range(len(proccoords))]
        #perform a symmetry check on the covariance matrix, just in case
        #Note, pos def => symmetric.
        #I don't know why the covariance matrix would ever be non-symmetric
        #unless the data were mangled.
        for ciii in range(len(ccov)):
            for cjjj in range(ciii+1, len(ccov)):
                if ccov[ciii][cjjj] == ccov[cjjj][ciii]:
                    pass
                else:
                    print "***ERROR***"
                    print "The provided covariance matrix is not symmetric."
                    print "Good fits need a symmetric covariance matrix."
                    print "Please provide different data."
                    print "Exiting."
                    print sys.exit(1)
        #check to see if (cov) matrix is positive definite.  If it is, then
        #it must have a Cholesky decomposition.
        #The posdefexcept finds this decomposition, and raises a LinAlgError
        #if the matrix is not positive definite.
        #The program then tells the user to select a different domain.
        #The data may still be useable.
        #Some people on the internet suggest this is faster, and I was going
        #to use a canned routine anyway, so this one won.
        try:
            posdefexcept(ccov)
        except np.linalg.linalg.LinAlgError:
            print "***ERROR***"
            print "Covariance matrix is not positive definite."
            print "Choose a different domain to fit."
            print "The data may still be useable."
            sys.exit(1)
        return rets(coord=proccoords, covar=ccov, numblocks=len(ccov))
    print "simple proc error"
    sys.exit(1)

CSENT = object()
def proc_file(pifile, pjfile=CSENT):
    """Process the current file.
    Return covariance matrix entry I,indexj in the case of multi-file
    structure.
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
    print "***Unexpected Error***"
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
    ####set up 1ab
    SENT1 = object()
    SENT2 = object()
    SENT3 = object()
    SENT4 = object()
    XMIN = SENT1
    XMAX = SENT2
    XSTEP = SENT3
    NUMPEXTRA = SENT4
    OPTIONS = namedtuple('ops', ['xmin', 'xmax', 'xstep', 'nextra'])
    INPUT, SWITCH, OPTIONS = main(sys.argv[1:])
    if isinstance(OPTIONS.xmax, str):
        try:
            XMAX = float(OPTIONS.xmax)
        except ValueError:
            print "***ERROR***"
            print "Invalid xmax."
            main(["h"])
    if isinstance(OPTIONS.xmin, str):
        try:
            XMIN = float(OPTIONS.xmin)
        except ValueError:
            print "***ERROR***"
            print "Invalid xmin."
            main(["h"])
    if isinstance(OPTIONS.nextra, str):
        try:
            OPSTEMP = OPTIONS.nextra
            OPSTEMP = int(OPSTEMP)
        except ValueError:
            print "***ERROR***"
            print "Invalid number of extra parameters."
            print "Expecting an int >= 0."
            main(["h"])
    if OPSTEMP >= 0:
        NUMPEXTRA = OPSTEMP
    else:
        NUMPEXTRA = -1
    ####error handling 2ab
    #test to see if file/folder exists
    if not (os.path.isfile(INPUT) or os.path.isdir(INPUT)):
        print "File:", INPUT, "not found"
        print "Folder:", INPUT, "also not found."
        main(["h"])
    #test to see if input is file, then process the file
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar', 'numblocks'])
    ####input time domain for folders 4ab
    if XMIN == SENT1 or XMAX == SENT2:
        print "Now, input valid domain (abscissa)."
        print "xmin<=x<=xmax"
        if XMIN == SENT1:
            print "x min="
            XMIN = float(raw_input())
        if XMAX == SENT2:
            print "time max="
            XMAX = float(raw_input())
    if XMAX < XMIN:
        print "Assuming you swapped xmin for xmax on command line."
        print "Swapping xmin for xmax."
        XMIN, XMAX = XMAX, XMIN
    #We only care about step size for multi file setup
    if XSTEP == SENT3 and os.path.isdir(INPUT):
        print "Assuming domain step size is 1 (int)."
        XSTEP = 1
    ####process the file(s) 3ab
    #process individual file (single file setup)
    if os.path.isfile(INPUT):
        RESRET = simple_proc_file(INPUT, XMIN, XMAX)
        COV = RESRET.covar
        COORDS = RESRET.coord
        #DIMCOV is dimensions of the covariance matrix
        DIMCOV = RESRET.numblocks
        #then find out domain of files to process
    ####process individual files in dir 5ab
    #error handling, test to see if time value goes out of range,
    #i.e. if data isn't available to match the requested time domain
    #i,j are new indices, shifting XMIN to the origin
    #j = 0 # initialized below
    #test if directory
    elif os.path.isdir(INPUT):
        i = 0
        #DIMCOV is dimensions of the covariance matrix
        DIMCOV = int((XMAX-XMIN)/XSTEP+1)
        #cov is the covariance matrix
        COV = [[[0] for _ in range(DIMCOV)] for _ in range(DIMCOV)]
        #COORDS are the coordinates to be plotted.
        #the ith point with the jth value
        COORDS = [[[0] for _ in range(2)] for _ in range(DIMCOV)]
        for time in np.arange(XMIN, XMAX+1, XSTEP):
            COORDS[i][0] = time
            j = 0
            for time2 in np.arange(XMIN, XMAX+1, XSTEP):
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
####we have data 6ab
    #at this point we have the covariance matrix, and coordinates
    #compute inverse of covariance matrix
    COVINV = inv(COV)
    print "(Rough) scale of errors in data points = ", COV[0][0]
####minimize 7ab
    #minimize chi squared
    #todo:generalize this
    if SWITCH == '0':
        print "This method is highly questionable."
        print "Most likely causes of failure:"
        print "(1): Pade definition is wrong."
        print "(2): Starting point is ill-considered."
        if NUMPEXTRA == -1:
            print "Input the number of extra fit parameters desired / 2"
            print "Two extra parameters will be used for each Extra"
            print "E.g., if you input 3, the number of extra parameters will"
            print "be 6.  See definition of pade given in source."
            print "Please fix to be more general:"
            print "each b_i value is taken to be greater than 4*(m_pi)^2"
            print "Extra = "
            NUMPEXTRA = int(raw_input())
            if NUMPEXTRA < 0 or (not isinstance(NUMPEXTRA, int)):
                print "***ERROR***"
                print "Expecting an int >= 0"
                sys.exit(2)
        #m_rho = 770 MeV
        #m_pi = 140 MeV
        #b_i>2.3716
        start_params = [-.18, 0.09405524, 1.21877187, 2.4]
        if NUMPEXTRA == 0:
            ADDPARAMS = []
        else:
            ADDPARAMS = [1.21877187+i/1000.0 if i%2 == 0
                         else 2.4+i/1000.0 for i in range(NUMPEXTRA*2)]
        for i in ADDPARAMS:
            start_params.append(i)
        MBOUND = 0.0779
        binds = [(None, None), (None, None), (None, None),
                 (MBOUND, None)]
        ADDBINDS = [(None, None) if i%2 ==0 else (MBOUND, None)
                    for i in range(NUMPEXTRA*2)]
        for i in ADDBINDS:
            binds.append(i)
        binds = tuple(binds)
        METHOD = 'L-BFGS-B'
    if SWITCH == '1':
        START_A_0 = 20
        START_ENERGY = 2
        start_params = [START_A_0, START_ENERGY]
        binds = ((None, None), (0, None))
        METHOD = 'L-BFGS-B'
    #error handling for Degrees of Freedom <= 0.
    #DIMCOV is number of points plotted.
    if len(start_params) >= DIMCOV:
        print "***ERROR***"
        print "Degrees of freedom <= 0."
        print "Rerun with a different number of fit parameters."
        sys.exit(1)
    #BFGS uses first derivatives of function
    #comment out options{...}, bounds for L-BFGS-B
    if not METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (COVINV, COORDS, SWITCH),
                              method=METHOD)
                          #method='BFGS')
                          #method='L-BFGS-B',
                          #bounds=binds,
                          #options={'disp': True})
    if METHOD in set(['L-BFGS-B']):
        RESULT_MIN = minimize(chi_sq, start_params, (COVINV, COORDS, SWITCH),
                              method=METHOD, bounds=binds,
                              options={'disp': True})
        print "number of iterations = ", RESULT_MIN.nit
    print "minimized params = ", RESULT_MIN.x
    print "successfully minimized = ", RESULT_MIN.success
    print "status of optimizer = ", RESULT_MIN.status
    print "message of optimizer = ", RESULT_MIN.message
    print "chi^2 minimized = ", RESULT_MIN.fun
    if RESULT_MIN.fun < 0:
        print "***ERROR***"
        print "Chi^2 minimizer failed. Chi^2 found to be less than zero."
    print "chi^2 reduced = ", RESULT_MIN.fun/(DIMCOV-len(start_params))
####compute errors 8ab
    #compute hessian matrix
    if RESULT_MIN.fun > 0 and RESULT_MIN.status == 0:
        HFUNC = lambda xrray: chi_sq(xrray, COVINV, COORDS, SWITCH)
        HFUN = nd.Hessian(HFUNC)
        #compute hessian inverse
        HINV = inv(HFUN(RESULT_MIN.x))
        #compute errors in fit parameters
        for i in range(len(HINV)):
            print "Statistical error in parameter #", i, " = "
            try:
                print sqrt(2*HINV[i][i])
            except ValueError:
                print "***ERROR***"
                print "Examine fit domain.  Hessian inverse has negative"
                print "diagonal entries, leading to complex errors in some"
                print "or all of the fit parameters."
                sys.exit(1)
        #ERR_A0 = sqrt(2*HINV[0][0])
        #ERR_ENERGY = sqrt(2*HINV[1][1])
        #print "a0 = ", RESULT_MIN.x[0], "+/-", ERR_A0
        #print "energy = ", RESULT_MIN.x[1], "+/-", ERR_ENERGY
####plot result 9ab
    #plot the function and the data, with error bars
    with PdfPages('foo.pdf') as pdf:
        XCOORD = [COORDS[i][0] for i in range(len(COORDS))]
        YCOORD = [COORDS[i][1] for i in range(len(COORDS))]
        ER2 = np.array([COV[i][i] for i in range(len(COORDS))])
        plt.errorbar(XCOORD, YCOORD, yerr=ER2, linestyle='None')
        #the fit function is plotted on a scale 1000x more fine
        #than the original data points
        XFIT = np.arange(XCOORD[0],
                         XCOORD[len(XCOORD)-1],
                         abs((XCOORD[len(
                             XCOORD)-1]-XCOORD[0]))/1000.0/len(XCOORD))
        YFIT = np.array([fit_func(XFIT[i], RESULT_MIN.x, SWITCH)
                         for i in range(len(XFIT))])
        #only plot fit function if minimizer result makes sense
        if RESULT_MIN.status == 0:
            print "Minimizer thinks that it worked.  Plotting fit."
            plt.plot(XFIT, YFIT)
        #todo: figure out a way to generally assign limits to plot
        #plt.xlim([XCOORD[0], XMAX+1])
        #magic numbers for the problem you're solving
        #plt.ylim([0, 0.1])
        #add labels, more magic numbers
        plt.title('Some Correlation function vs. time')
        #todo: figure out a way to generally place text on plot
        #STRIKE1 = "Energy = " + str(RESULT_MIN.x[1]) + "+/-" + str(
        #    ERR_ENERGY)
        #STRIKE2 = "Amplitude = " + str(RESULT_MIN.x[0]) + "+/-" + str(
        #    ERR_A0)
        #X_POS_OF_FIT_RESULTS = XCOORD[3]
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[3], STRIKE1)
        #plt.text(X_POS_OF_FIT_RESULTS, YCOORD[7], STRIKE2)
        plt.xlabel('time (?)')
        plt.ylabel('the function')
        #read out into a pdf
        pdf.savefig()
        #show the plot
        plt.show()
    #extraneous notes below
    sys.exit(0)
