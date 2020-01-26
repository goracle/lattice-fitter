#!/usr/bin/python3
"""Perform final fits"""
import sys
import glob
import pickle
import subprocess
import os

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
        xmin, xmax = arr[-3]
        xmin = str(xmin)
        xmax = str(xmax)
        call = 'latfit -f . --xmin='+xmin+' --xmax='+xmax
        subprocess.call(call, shell=True)

if __name__ == '__main__':
    main()
