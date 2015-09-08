from __future__ import division
from collections import namedtuple
import numpy as np
import os

from latfit.globs import EIGCUT
from latfit.extract.simple_proc_file import simple_proc_file
from latfit.extract.proc_folder import proc_folder
from latfit.extract.proc_file import proc_file

def extract(INPUT, xmin, xmax):
    """Get covariance matrix, coordinates.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar', 'numblocks'])
    if os.path.isfile(INPUT):
        RESRET = simple_proc_file(INPUT, xmin, xmax, EIGCUT)
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
                    STR1 = "Either domain is invalid,"
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
    return COORDS, COV, DIMCOV
