from collections import namedtuple
import numpy as np
from numpy import ceil,floor
import sys

from latfit.extract.gevp_proc import gevp_proc
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder
from latfit.extract.proc_file import proc_line

from latfit.config import GEVP_DIRS
from latfit.config import EFF_MASS

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
    COV = np.zeros((dimcov,dimcov,dimops,dimops),dtype=np.complex128)
    #COORDS are the coordinates to be plotted.
    #the ith point with the jth value
    COORDS = np.zeros((dimcov,2),dtype=object)
    for timei in np.arange(xmin, xmax+1, XSTEP):
        if timei in REUSE:
            REUSE['i']=REUSE[timei]
        else:
            REUSE.pop('i')
            if timei!=xmin:
                #delete me if working!
                print("***ERROR***")
                print("Time slice:",timei,", is not being stored for some reason")
                sys.exit(1)
        timei2=ceil(float(timei)/2.0/XSTEP)*XSTEP if ceil(float(timei)/2.0)!=timei else max(floor(float(timei)/2.0/XSTEP)*XSTEP,xmin)
        #set the times coordinate
        COORDS[i][0] = timei
        #extract files
        IFILES = [[proc_folder(GEVP_DIRS[op1][op2],timei) for op1 in range(dimops)] for op2 in range(dimops)]
        I2FILES = [[proc_folder(GEVP_DIRS[op1][op2],timei2) for op1 in range(dimops)] for op2 in range(dimops)]
        #check for errors
        IFILES = [[pre_proc_file(IFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        I2FILES = [[pre_proc_file(I2FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        if EFF_MASS:
            I3FILES = [[proc_folder(GEVP_DIRS[op1][op2],timei+XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
            I4FILES = [[proc_folder(GEVP_DIRS[op1][op2],timei+2*XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
            I3FILES = [[pre_proc_file(I3FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            I4FILES = [[pre_proc_file(I4FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
        j=0
        for timej in np.arange(xmin, xmax+1, XSTEP):
            if timej in REUSE:
                REUSE['j']=REUSE[timej]
            else:
                REUSE['j']=0
            timej2=ceil(float(timej)/2.0/XSTEP)*XSTEP if ceil(float(timej)/2.0)!=timej else max(floor(float(timej)/2.0/XSTEP)*XSTEP,xmin)
            TIME_ARR=[timei,timei2,timej,timej2,XSTEP]
            JFILES = [[proc_folder(GEVP_DIRS[op1][op2],timej) for op1 in range(dimops)] for op2 in range(dimops)]
            JFILES = [[pre_proc_file(JFILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            J2FILES = [[proc_folder(GEVP_DIRS[op1][op2],timej2) for op1 in range(dimops)] for op2 in range(dimops)]
            J2FILES = [[pre_proc_file(J2FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
            if EFF_MASS:
                J3FILES = [[proc_folder(GEVP_DIRS[op1][op2],timej+XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
                J4FILES = [[proc_folder(GEVP_DIRS[op1][op2],timej+2*XSTEP) for op1 in range(dimops)] for op2 in range(dimops)]
                J3FILES = [[pre_proc_file(J3FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
                J4FILES = [[pre_proc_file(J4FILES[op1][op2],GEVP_DIRS[op1][op2]) for op1 in range(dimops)] for op2 in range(dimops)]
                RESRET = gevp_proc(IFILES,I2FILES,JFILES,J2FILES,TIME_ARR,[(I3FILES,J3FILES),(I4FILES,J4FILES)],reuse=REUSE)
            else:
                RESRET = gevp_proc(IFILES,I2FILES,JFILES,J2FILES,TIME_ARR,reuse=REUSE)
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
