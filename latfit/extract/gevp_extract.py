"""Extract cov. matrix and jackknife blocks."""
from collections import namedtuple
import numpy as np
import sys

from latfit.extract.gevp_proc import gevp_proc_ijfile
from latfit.extract.gevp_getfiles import gevp_getfiles
from latfit.extract.extract import reuse_ij

from latfit.config import GEVP_DIRS
from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS

def gevp_extract(xmin,xmax,xstep):
    """Get covariance matrix, coordinates, jackknife blocks.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: resret
    resret = namedtuple('ret', ['coord', 'covar'])

    #dimcov is dimensions of the covariance matrix
    dimcov = int((xmax-xmin)/xstep+1)

    #Reuse results (store all read-in data)
    reuse={xmin:0}
    
    #dimops is the dimension of the correlator matrix
    dimops = len(GEVP_DIRS)

    #cov is the covariance matrix
    cov = np.zeros((dimcov,dimcov,dimops*(NUM_PENCILS+1),dimops*(NUM_PENCILS+1)),dtype=np.complex128)

    #coords are the coordinates to be plotted.
    #the ith point with the jth value
    coords = np.zeros((dimcov, 2, dimops))

    for i, timei in enumerate(np.arange(xmin, xmax+1, xstep)):

        #set the times coordinate
        #coords[i][0] = timei

        reuse_ij(reuse, timei, 'i')
        timei2, ifiles_tup=gevp_getfiles(timei, xstep, xmin)

        for j, timej in enumerate(np.arange(xmin, xmax+1, xstep)):

            reuse_ij(reuse, timej, 'j')
            timej2,jfile_tup=gevp_getfiles(timej, xstep, xmin)
            time_arr=[timei,timei2,timej,timej2,xstep]

            #get the cov entry and the block
            resret = gevp_proc_ijfile(ifile_tup, jfile_tup,
                                      time_arr, reuse=reuse)

            #fill in the covariance matrix
            cov[i][j] = resret.covar

            #fill in dictionary for reusing already extracted blocks
            #with the newest block
            if i == 0:
                reuse[timej]=resret.returnblk

            if j == 0:
                #only when j=0 does the i block need updating
                reuse['i'] = reuse[timej]
                #only store coordinates once.
                coords[i][0] = timei
                coords[i][1] = resret.coord

    return coords, cov, reuse
