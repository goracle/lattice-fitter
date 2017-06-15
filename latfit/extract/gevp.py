import os
import numpy as np
from os.path import isfile, join
import re
from os import listdir
from latfit.extract import extract

from latfit.config import GEVP_DIRS

def gevp_extract(XMIN,XMAX,XSTEP):
    #dimcov is dimensions of the covariance matrix
    dimcov = int((XMAX-XMIN)/XSTEP+1)
    #dimops is the dimension of the correlator matrix
    dimops = len(GEVP_DIRS)
    #cov is the covariance matrix
    COV = np.zeros((dimops,dimops,dimcov,dimcov))
    #COORDS are the coordinates to be plotted.
    #the ith point with the jth value
    COORDS = np.zeros((dimops,dimops,dimcov,2))
    ret_coords, ret_cov = extract(XMIN,XMAX,XSTEP,INPUT,INPUT2)
    #return COORDS, COV
