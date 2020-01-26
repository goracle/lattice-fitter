#!/usr/bin/python3
"""Perform final fits"""

import sys
import glob
import pickle
import subprocess

def main():
    """main"""
    for fil in glob.glob('fit_*'):
        if 'log' in fil:
            continue
        os.symlink(fil, 'final_fit.p')
        arr = pickle.load(open(fil, 'rb'))
        xmin, xmax = arr[-1]
        xmin = str(xmin)
        xmax = str(xmax)
        subprocess.check_output(['latfit', '-f',
                                 '.', '--xmin', xmin,
                                 '--xmax', xmax])




if __name__ == '__main__':
    main()
