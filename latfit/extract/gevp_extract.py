from collections import namedtuple
import numpy as np
from numpy import ceil,floor

from latfit.extract.gevp_proc import gevp_proc
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder
from latfit.extract.proc_file import proc_line

from latfit.config import GEVP_DIRS

def gevp_extract(XMIN,XMAX,XSTEP):
    i = 0
    #result is returned as a named tuple: RESRET
    RESRET = namedtuple('ret', ['coord', 'covar'])
    #Reuse results
    REUSE={}
    #dimcov is dimensions of the covariance matrix
    dimcov = int((XMAX-XMIN)/XSTEP+1)
    #dimops is the dimension of the correlator matrix
    dimops = len(GEVP_DIRS)
    #cov is the covariance matrix
    COV = np.zeros((dimcov,dimcov,dimops,dimops),dtype=np.complex128)
    #COORDS are the coordinates to be plotted.
    #the ith point with the jth value
    COORDS = np.zeros((dimcov,2),dtype=object)
    for timei in np.arange(XMIN, XMAX+1, XSTEP):
        if timei in REUSE:
            REUSE['i']=REUSE[timei]
        else:
            REUSE.pop('i')
            if timei!=xmin:
                #delete me if working!
                print("***ERROR***")
                print("Time slice:",timei,", is not being stored for some reason")
                sys.exit(1)
        timei2=ceil(float(timei)/2.0/XSTEP)*XSTEP if ceil(float(timei)/2.0)!=timei else max(floor(float(timei)/2.0/XSTEP)*XSTEP,XMIN)
        #set the times coordinate
        COORDS[i][0] = timei
        j=0
        #extract files
        IFILES = [[proc_folder(GEVP_DIRS[op1][op2],timei) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES2 = [[proc_folder(GEVP_DIRS[op1][op2],timei2) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES3 = [[proc_folder(GEVP_DIRS[op1][op2],timei+XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
        #check for errors
        IFILES = [[pre_proc_file(IFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES2 = [[pre_proc_file(IFILES2[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        IFILES3 = [[pre_proc_file(IFILES3[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        for timej in np.arange(XMIN, XMAX+1, XSTEP):
            if timej in REUSE:
                REUSE['j']=REUSE[timej]
            else:
                REUSE.pop('j')
            timej2=ceil(float(timej)/2.0/XSTEP)*XSTEP if ceil(float(timej)/2.0)!=timej else max(floor(float(timej)/2.0/XSTEP)*XSTEP,XMIN)
            TIME_ARR=[timei,timei2,timej,timej2,XSTEP]
            #extract files
            JFILES = [[proc_folder(GEVP_DIRS[op1][op2],timej) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES2 = [[proc_folder(GEVP_DIRS[op1][op2],timej2) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES3 = [[proc_folder(GEVP_DIRS[op1][op2],timej+XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
            #check for errors
            JFILES = [[pre_proc_file(JFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES2 = [[pre_proc_file(JFILES2[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES3 = [[pre_proc_file(JFILES3[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            RESRET = gevp_proc(IFILES,IFILES2,IFILES3,JFILES,JFILES2,JFILES3,TIME_ARR,reuse=REUSE)
            COV[i][j] = RESRET.covar
            #only store coordinates once.  each file is read many times
            if j == 0:
                COORDS[i][1] = RESRET.coord
            j+=1
        i+=1
    return COORDS, COV, REUSE
