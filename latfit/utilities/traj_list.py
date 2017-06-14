#!/usr/bin/python3

import read_file as rf
import os.path
import numpy as np
import linecache as lc
from os import listdir
import os.path
from os.path import isfile, join
import re

def traj_list(onlyfiles=None):
    trajl = set()
    if not onlyfiles:
        onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn2 in onlyfiles:
        trajl.add(rf.traj(fn2))
    trajl-=set([None])
    print("Done getting trajectory list. N trajectories = "+str(len(trajl)))
    return trajl

def main():
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    print(traj_list(onlyfiles))

if __name__ == "__main__":
    main()

