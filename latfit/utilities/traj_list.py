#!/usr/bin/python3
"""get traj list"""
from os import listdir
from os.path import isfile, join
import read_file as rf

def traj_list(onlyfiles=None):
    """get list of trajectories from file list
    """
    trajl = set()
    if not onlyfiles:
        onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    for fn2 in onlyfiles:
        trajl.add(rf.traj(fn2))
    trajl -= set([None])
    print("Done getting trajectory list. N trajectories = "+str(len(trajl)))
    return trajl

def main():
    """get traj list main"""
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    print(traj_list(onlyfiles))

if __name__ == "__main__":
    main()
