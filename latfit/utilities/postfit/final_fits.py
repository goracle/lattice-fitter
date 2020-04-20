#!/usr/bin/python3
"""Perform final fits"""
import sys
import glob
import pickle
import subprocess
import os
import numpy as np

def fitwin_from_include(include):
    """Find the necessary fit window"""
    tmin = np.inf
    tmax = 0
    for i in include:
        tmin = min(min(i), tmin)
        tmax = max(max(i), tmax)
    return (tmin, tmax)


def main():
    """main"""
    link = 'final_fit.p'
    for fil in glob.glob('fit_*'):
        if 'log' in fil:
            continue
        if os.path.exists(link):
            os.unlink(link)
        assert not os.path.exists(link), link
        os.symlink(fil, link)
        assert os.path.exists(link), link
        arr = pickle.load(open(fil, 'rb'))
        print("using:", fil)
        print(arr)
        try:
            fitmin, fitmax = arr[-4]
        except ValueError:
            print(arr)
            raise
        ens = arr[4]
        fitmin, fitmax = fitwin_from_include(arr[0])

        # old method for xmin/xmax
        xmin = 1
        if ens == '24c':
            xmax = 16
        elif ens == '32c':
            xmax = 22

        # processing
        xmin, xmax = fitmin, fitmax
        dim = arr[2]
        dimr = dim
        if len(sys.argv) > 1:
            dimr = int(sys.argv[1])
        if dim != dimr:
            continue
        xmin, xmax = xmin, str(xmax)
        fitmin = str(fitmin)
        fitmax = str(fitmax)
        # xmin, xmax = fitmin, fitmax
        flag = 1
        xmin = 1
        xmax = '22'
        call = 'mpirun -np 4 latfit -f . --xmin='+str(xmin)+' --xmax='+\
            xmax+' --fitmin='+fitmin+' --fitmax='+fitmax
        print("call =", call)
        flag = subprocess.call(call, shell=True)
        if flag:
            xmin = int(fitmin)
        while flag:
            break
            call = 'latfit -f . --xmin='+str(xmin)+' --xmax='+\
                xmax+' --fitmin='+fitmin+' --fitmax='+fitmax
            print("call =", call)
            flag = subprocess.call(call, shell=True)
            if flag and xmin:
                xmin -= 1
            else:
                break

if __name__ == '__main__':
    main()
