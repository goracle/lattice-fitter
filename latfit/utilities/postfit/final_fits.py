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
            xmin, xmax = arr[-4]
        except ValueError:
            print(arr)
            raise
        xmin, xmax = fitwin_from_include(arr[0])
        xmin = str(xmin)
        xmax = str(xmax)
        call = 'latfit -f . --xmin='+xmin+' --xmax='+xmax
        subprocess.call(call, shell=True)

if __name__ == '__main__':
    main()
