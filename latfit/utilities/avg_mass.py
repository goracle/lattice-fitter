#!/usr/bin/python

import numpy as np
import scipy
from scipy.optimize import curve_fit
import os.path
import re
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import sys

def f(t,A,E):
    return A*np.exp(-E*t)

def fit_this(filename,lb,ub, yp=False):
    xdata=[]
    ydata=[]
    count=lb
    fn=open(filename,'r')
    for line in fn:
        l=line.split()
        xdata=np.append(xdata,count)
        ydata=np.append(ydata,complex(l[1]).real)
        count+=1
        if count>ub:
            break
    popt, pcov = curve_fit(f, xdata, ydata)
    A,E=popt
    print filename, A,E
    if(yp):
        plt.plot(xdata, f(xdata, *popt), 'g--', label='fit-> A*exp(-E*t)')
        plt.title(filename+" Exp Fit")
        plt.xlabel('t')
        plt.ylabel('Normed Corr')
        plt.plot(xdata, ydata, 'b.', label='data')
        plt.legend()
        plt.show()
    return A,E

def main():
    yp=False 
    if len(sys.argv) == 2:
        if sys.argv[1] == '1':
            yp=True
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))] 
    count = 0
    A=0
    E=0
    Err=[]
    for fn in onlyfiles:
        if not re.search("traj", fn):
            continue
        count+=1
        A1,E1=fit_this(fn,5,16,yp)
        A+=A1
        E+=E1
        Err=np.append(Err,E1)
    A/=count 
    E/=count 
    print "Done analyzing.  Fit params A*exp(-E*t); A=", A, "E=", E
    print "std err in E=",np.std(Err)/np.sqrt(count)

if __name__ == "__main__":
    main()
