from collections import namedtuple
import numpy as np
import sys

from latfit.extract.gevp_proc import gevp_proc
from latfit.extract.gevp_getfiles import gevp_getfiles

from latfit.config import GEVP_DIRS
from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS

def gevp_extract(xmin,xmax,XSTEP):
    i = 0
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar'])
    #Reuse results (store all read-in data)
    REUSE={xmin:0}
    #dimcov is dimensions of the covariance matrix
    dimcov = int((xmax-xmin)/XSTEP+1)
    #dimops is the dimension of the correlator matrix
    dimops = len(GEVP_DIRS)
    #cov is the covariance matrix
    COV = np.zeros((dimcov,dimcov,dimops*(NUM_PENCILS+1),dimops*(NUM_PENCILS+1)),dtype=np.complex128)
    #COORDS are the coordinates to be plotted.
    #the ith point with the jth value
    COORDS = np.zeros((dimcov,2),dtype=object)
    for timei in np.arange(xmin, xmax+1, XSTEP):
        #set the times coordinate
        COORDS[i][0] = timei
        if timei in REUSE:
            REUSE['i']=REUSE[timei]
        else:
            REUSE['i']=0
            if timei!=xmin:
                #delete me if working!
                print("***ERROR***")
                print("Time slice:",timei,", is not being stored for some reason")
                sys.exit(1)
        if EFF_MASS:
            timei2,IFILES,IFILES2,IFILES3,IFILES4=gevp_getfiles(timei,XSTEP,xmin)
        else:
            timei2,IFILES,IFILES2=gevp_getfiles(timei,XSTEP,xmin)
        j=0
        for timej in np.arange(xmin, xmax+1, XSTEP):
            if timej in REUSE:
                REUSE['j']=REUSE[timej]
            else:
                REUSE['j']=0
            if EFF_MASS:
                timej2,JFILES,JFILES2,JFILES3,JFILES4=gevp_getfiles(timej,XSTEP,xmin)
                TIME_ARR=[timei,timei2,timej,timej2,XSTEP]
                RESRET = gevp_proc(IFILES,IFILES2,JFILES,JFILES2,TIME_ARR,[(IFILES3,JFILES3),(IFILES4,JFILES4)],reuse=REUSE)
            else:
                timej2,JFILES,JFILES2=gevp_getfiles(timej,XSTEP,xmin)
                TIME_ARR=[timei,timei2,timej,timej2,XSTEP]
                RESRET = gevp_proc(IFILES,IFILES2,JFILES,JFILES2,TIME_ARR,reuse=REUSE)
            COV[i][j] = RESRET.covar
            REUSE[timej]=RESRET.returnblk
            if timei==timej:
                REUSE['i']=REUSE[timej]
            #only store coordinates once.  each file is read many times
            if j == 0:
                COORDS[i][1] = RESRET.coord
                COORDS[i][0] = timei
            j+=1
        i+=1
    return COORDS, COV, REUSE
