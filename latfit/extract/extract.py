"""Extract cov. matrix and jackknife blocks."""
from collections import namedtuple
import os
import numpy as np

from latfit.extract.simple_proc_file import simple_proc_file
from latfit.extract.proc_ijfile import proc_ijfile
from latfit.extract.getfiles import getfiles

from latfit.config import GEVP_DIRS
from latfit.config import GEVP
from latfit.config import EIGCUT
from latfit.config import NUM_PENCILS
from latfit.config import STYPE

def extract(input_f, xmin, xmax, xstep):
    """Get covariance matrix, coordinates, jackknife blocks.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: resret
    resret = namedtuple('ret', ['coord', 'covar', 'numblocks', 'returnblk'])

    #if simple file, do that extraction
    if os.path.isfile(input_f) and STYPE == 'ascii':
        resret = simple_proc_file(input_f, xmin, xmax, EIGCUT)
        cov = resret.covar
        coords = resret.coord

    #test if directory
    elif os.path.isdir(input_f) or STYPE == 'hdf5':

        #reuse results
        reuse = {xmin:0}

        ##allocate space for return values

        coords, cov = allocate(xmin, xmax, xstep)

        tij = [None, None]

        for i, timei in enumerate(np.arange(xmin, xmax+1, xstep)):

            #setup the reuse block for 'i' so proc_ijfile can remain
            #time agnostic
            reuse['i'] = reuse_ij(reuse, timei)

            #tell the processor function which time slices we are on
            tij[0] = timei

            #get the ifile(s)
            ifile_tup = getfiles(timei, xstep, xmin, input_f)

            for j, timej in enumerate(np.arange(xmin, xmax+1, xstep)):

                #same for j
                reuse['j'] = reuse_ij(reuse, timej)
                jfile_tup = getfiles(timej, xstep, xmin, input_f)
                tij[1] = timej

                #get the cov entry and the block
                resret = proc_ijfile(ifile_tup, jfile_tup, reuse=reuse, timeij=tij)

                #fill in the covariance matrix
                cov[i][j] = resret.covar

                #fill in dictionary for reusing already extracted blocks
                #with the newest block
                if i == 0:
                    reuse[timej] = resret.returnblk

                if j == 0:
                    #only when j=0 does the i block need updating
                    reuse['i'] = reuse[timei]
                    #only store coordinates once.
                    coords[i][0] = timei
                    coords[i][1] = resret.coord

    return coords, cov, reuse


#side effects warning
def reuse_ij(reuse, time):
    """Prepare reuse container with proper block for i or j
    This allows proc_ijfile to remain agnostic as to where it is in the
    structure of the covariance matrix.
    """
    if time in reuse:
        retblk = reuse[time]
    else:
        retblk = 0
    return retblk

if GEVP:
    def allocate(xmin, xmax, xstep):
        """Allocate blank coords, covariance matrix (GEVP)."""
        #dimcov is dimensions of the covariance matrix
        dimcov = int((xmax-xmin)/xstep+1)
        #dimops is the dimension of the correlator matrix
        dimops = len(GEVP_DIRS)
        #cov is the covariance matrix
        cov = np.zeros((dimcov, dimcov, dimops*(NUM_PENCILS+1),
                        dimops*(NUM_PENCILS+1)), dtype=np.float)
        #coords are the coordinates to be plotted.
        #the ith point with the jth value
        coords = np.zeros((dimcov, 2), dtype=object)
        return coords, cov

else:
    def allocate(xmin, xmax, xstep):
        """Allocate blank coords, covariance matrix."""
        #dimcov is dimensions of the covariance matrix
        dimcov = int((xmax-xmin)/xstep+1)
        #cov is the covariance matrix
        cov = np.zeros((dimcov, dimcov))
        #coords are the coordinates to be plotted.
        #the ith point with the jth value
        coords = np.zeros((dimcov, 2))
        return coords, cov
