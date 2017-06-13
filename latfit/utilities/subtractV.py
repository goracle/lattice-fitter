#!/usr/bin/python

from . import read_file as rf
from .traj_list import traj_list
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re

def procV(filen):
    Lt = sum(1 for line in open(filen))
    ar = np.zeros(shape=(Lt),dtype=complex)
    fn = open(filen, 'r')
    for line in fn:
        l=line.split()
        tsrc = int(l[0])
        if len(l)==3:
            ar.itemset(tsrc,complex(float(l[1]),float(l[2])))
        elif len(l)==2:
            ar.itemset(tsrc,complex(l[1]))
        else:
            print("Not a disconnected or summed diagram.  exiting.")
            print("cause:",filen)
            exit(1)
    return ar

def AvgVdis():
    d='summed_tsrc_diagrams/'
    if not os.path.isdir(d):
        os.makedirs(d)
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    tlist=traj_list(onlyfiles)
    for fn in onlyfiles:
        fign = rf.figure(fn)
        if fign in ['Vdis', 'scalar-bubble']:
            mom=rf.mom(fn)
            try:
                p1,p2 = mom
                momstr="_mom1"+rf.ptostr(p1)+"_mom2"+rf.ptostr(p2)
            except:
                p1 = mom
                momstr="_mom"+rf.ptostr(p1)
            sep = rf.sep(fn)
            if sep:
                sepstr="_sep"+str(sep)
            else:
                sepstr=""
            outfile = d+"Avg_"+fign+sepstr+momstr
            if os.path.isfile(outfile):
                continue
            avg = np.array(procV(fn))
            numt=1
            #use this first file to bootstrap the rest of the files (traj substitution)
            for traj in tlist:
                if traj == rf.traj(fn):
                    continue
                #slightly less stringent checking here on substitute:  no checking of anything following second underscore. probably fine since user warned above.
                fn2 = re.sub('traj_([B0-9]+)_','traj_'+traj+'_',fn)
                try:
                    open(fn2,'r')
                except:
                    continue
                avg+=np.array(procV(fn2))
                numt+=1
            print("Number of configs to average over:",numt,"for outfile:",outfile)
            rf.write_vec_str(avg/numt,outfile)
    print("Done writing Vdis averaged over trajectories.")
    return

def main():
    AvgVdis()

if __name__ == "__main__":
    main()
