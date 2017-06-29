from collections import namedtuple
import numpy as np
import os
import sys

from latfit.extract.simple_proc_file import simple_proc_file
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_file import proc_file
from latfit.extract.proc_folder import proc_folder

from latfit.config import EIGCUT
from latfit.config import EFF_MASS

def extract(INPUT, xmin, xmax, xstep):
    """Get covariance matrix, coordinates.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar', 'numblocks','returnblk'])
    #REUSE results
    REUSE={xmin:0}
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
        COV = np.zeros((dimcov,dimcov))
        #COORDS are the coordinates to be plotted.
        #the ith point with the jth value
        COORDS = np.zeros((dimcov,2))
        for timei in np.arange(xmin, xmax+1, xstep):
            if timei in REUSE:
                REUSE['i']=REUSE[timei]
            else:
                REUSE.pop('i')
                if timei!=xmin:
                    #delete me if working!
                    print("***ERROR***")
                    print("Time slice:",timei,", is not being stored for some reason")
                    sys.exit(1)
            #extract file
            IFILE = proc_folder(INPUT, timei)
            #check for errors
            IFILE = pre_proc_file(IFILE,INPUT)
            if EFF_MASS:
                I2FILE = proc_folder(INPUT, timei+xstep)
                I3FILE = proc_folder(INPUT, timei+2*xstep)
                I2FILE = pre_proc_file(I2FILE,INPUT)
                I3FILE = pre_proc_file(I3FILE,INPUT)
            j = 0
            for timej in np.arange(xmin, xmax+1, xstep):
                if timej in REUSE:
                    REUSE['j']=REUSE[timej]
                else:
                    REUSE['j']=0
                JFILE = proc_folder(INPUT, timej)
                JFILE = pre_proc_file(JFILE,INPUT)
                #if plotting effective mass
                if EFF_MASS:
                    J2FILE = proc_folder(INPUT, timej+xstep)
                    J3FILE = proc_folder(INPUT, timej+2*xstep)
                    J2FILE = pre_proc_file(J2FILE,INPUT)
                    J3FILE = pre_proc_file(J3FILE,INPUT)
                    RESRET = proc_file(IFILE, JFILE,
                                       [(I2FILE,J2FILE),(I3FILE,J3FILE)],reuse=REUSE)
                else:
                    RESRET = proc_file(IFILE, JFILE,reuse=REUSE)
                #fill in the covariance matrix
                COV[i][j] = RESRET.covar
                #only store coordinates once.  each file is read many times
                REUSE[timej]=RESRET.returnblk
                if timej==timei:
                    REUSE['i']=REUSE[timej]
                if j == 0:
                    COORDS[i][0] = timei
                    COORDS[i][1] = RESRET.coord
                j += 1
            i += 1
    return COORDS, COV, REUSE
