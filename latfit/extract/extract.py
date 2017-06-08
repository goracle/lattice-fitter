from __future__ import division
from collections import namedtuple
import numpy as np
import os

from latfit.config import EIGCUT
from latfit.config import EFF_MASS
from latfit.extract.simple_proc_file import simple_proc_file
from latfit.extract.proc_folder import proc_folder
from latfit.extract.proc_file import proc_file

def pre_proc_file(IFILE,INPUT):
    IFILE = INPUT + "/" + IFILE
    try:
        TRIAL = open(IFILE, "r")
    except TypeError:
        STR1 = "Either domain is invalid,"
        print STR1, "or folder is invalid."
        print "Double check contents of folder."
        print "Offending file(s):"
        print IFILE
        sys.exit(1)
    return IFILE

def extract(INPUT, xmin, xmax, xstep):
    """Get covariance matrix, coordinates.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar', 'numblocks'])
    #reuse results
    reuse={}
    if os.path.isfile(INPUT):
        RESRET = simple_proc_file(INPUT, xmin, xmax, EIGCUT)
        COV = RESRET.covar
        COORDS = RESRET.coord
        #dimcov is dimensions of the covariance matrix
        dimcov = RESRET.numblocks
        #then find out domain of files to process
    ####process individual files in dir 5ab
    #error handling, test to see if time value goes out of range,
    #i.e. if data isn't available to match the requested time domain
    #i,j are new indices, shifting xmin to the origin
    #j = 0 # initialized below
    #test if directory
    elif os.path.isdir(INPUT):
        i = 0
        #dimcov is dimensions of the covariance matrix
        dimcov = int((xmax-xmin)/xstep+1)
        #cov is the covariance matrix
        COV = [[[0] for _ in range(dimcov)] for _ in range(dimcov)]
        #COORDS are the coordinates to be plotted.
        #the ith point with the jth value
        COORDS = [[[0] for _ in range(2)] for _ in range(dimcov)]
        for time in np.arange(xmin, xmax+1, xstep):
            COORDS[i][0] = time
            j = 0
            for time2 in np.arange(xmin, xmax+1, xstep):
                #extract file
                IFILE = proc_folder(INPUT, time)
                JFILE = proc_folder(INPUT, time2)
                #check for errors
                IFILE = pre_proc_file(IFILE,INPUT)
                JFILE = pre_proc_file(JFILE,INPUT)
                #if plotting effective mass
                if EFF_MASS:
                    ti2 = time+xstep
                    ti3 = time+2*xstep
                    tj2 = time2+xstep
                    tj3 = time2+2*xstep
                    #print time,ti2,ti3,time2,tj2,tj3
                    I2FILE = proc_folder(INPUT, ti2)
                    I3FILE = proc_folder(INPUT, ti3)
                    J2FILE = proc_folder(INPUT, tj2)
                    J3FILE = proc_folder(INPUT, tj3)
                    I2FILE = pre_proc_file(I2FILE,INPUT)
                    I3FILE = pre_proc_file(I3FILE,INPUT)
                    J2FILE = pre_proc_file(J2FILE,INPUT)
                    J3FILE = pre_proc_file(J3FILE,INPUT)
                    #print IFILE,I2FILE,I3FILE,JFILE,J2FILE,J3FILE
                    RESRET = proc_file(IFILE, JFILE,
                                       [(I2FILE,J2FILE),(I3FILE,J3FILE)],reuse)
                else:
                    RESRET = proc_file(IFILE, JFILE)
                #fill in the covariance matrix
                COV[i][j] = RESRET.covar
                #only store coordinates once.  each file is read many times
                if j == 0:
                    COORDS[i][1] = RESRET.coord
                j += 1
            i += 1
    return COORDS, COV
