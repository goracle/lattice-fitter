"""Extract cov. matrix and jackknife blocks."""
from collections import namedtuple
import os
import numpy as np

from latfit.extract.simple_proc_file import simple_proc_file
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_ijfile import proc_ijfile
from latfit.extract.proc_folder import proc_folder

from latfit.config import EIGCUT
from latfit.config import EFF_MASS

def extract(input_f, xmin, xmax, xstep):
    """Get covariance matrix, coordinates, jackknife blocks.
    This is the meta-extractor.  It processes both individual files and
    folders.
    """
    #result is returned as a named tuple: resret
    resret = namedtuple('ret', ['coord', 'covar', 'numblocks', 'returnblk'])

    #if simple file, do that extraction
    if os.path.isfile(input_f):
        resret = simple_proc_file(input_f, xmin, xmax, EIGCUT)
        cov = resret.covar
        coords = resret.coord
        dimcov = resret.numblocks

    #test if directory
    elif os.path.isdir(input_f):

        #dimcov is dimensions of the covariance matrix
        dimcov = int((xmax-xmin)/xstep+1)

        #reuse results
        reuse = {xmin:0}

        ##allocate space for return values

        #cov is the covariance matrix
        cov = np.zeros((dimcov, dimcov))

        #coords are the coordinates to be plotted.
        #the ith point with the jth value
        coords = np.zeros((dimcov, 2))

        for i, timei in enumerate(np.arange(xmin, xmax+1, xstep)):

            #setup the reuse block for 'i' so proc_ijfile can remain
            #time agnostic
            reuse_ij(reuse, timei, 'i')

            #get the ifile(s)
            ifile_tup = ij_file_prep(timei, input_f, xstep)

            for j, timej in enumerate(np.arange(xmin, xmax+1, xstep)):

                #same for j
                reuse_ij(reuse, timej, 'j')
                jfile_tup = ij_file_prep(timej, input_f, xstep)

                #get the cov entry and the block
                resret = proc_ijfile(ifile_tup, jfile_tup, reuse=reuse)

                #fill in the covariance matrix
                cov[i][j] = resret.covar

                #fill in dictionary for reusing already extracted blocks
                #with the newest block
                if i == 0:
                    reuse[timej] = resret.returnblk

                if j == 0:
                    #only when j=0 does the i block need updating
                    reuse['i'] = reuse[timej]
                    #only store coordinates once.
                    coords[i][0] = timei
                    coords[i][1] = resret.coord

    return coords, cov, reuse

def ij_file_prep(time, input_f, xstep):
    """Get files for a given time slice."""
    #extract file
    ijfile = proc_folder(input_f, time)
    #check for errors
    ijfile = pre_proc_file(ijfile, input_f)
    if EFF_MASS:
        ij2file = proc_folder(input_f, time+xstep)
        ij3file = proc_folder(input_f, time+2*xstep)
        ij2file = pre_proc_file(ij2file, input_f)
        ij3file = pre_proc_file(ij3file, input_f)
        ijfile_tup = (ijfile, ij2file, ij3file)
    else:
        ijfile_tup = (ijfile)
    return ijfile_tup

#side effects warning
def reuse_ij(reuse, time, ij_str):
    """Prepare reuse container with proper block for i or j
    This allows proc_ijfile to remain agnostic as to where it is in the
    structure of the covariance matrix.
    """
    if time in reuse:
        reuse[ij_str] = reuse[time]
    else:
        reuse[ij_str] = 0
