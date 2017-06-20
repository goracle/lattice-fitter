#!/usr/bin/pypy

import read_file as rf
import os.path
import numpy as np
import linecache as lc
from os import listdir
import os.path
from os.path import isfile, join
import sys
import re

#write jackknife blocks for this basename
def write_blocks(trajl,outdir,basename,Lt, FixNorm = False, Isospin = 0):
    basename2='_'+basename
    #number of trajectories - 1 (to avg over)
    n = len(trajl)-1
    #loop over lines in the base, each gives a separate jk block of trajs
    for time in range(Lt): 
        #block file name to append to
        #(a2a since this was written to do all-to-all analyses)
        outfile=outdir+"/a2a."+"jkblk.t"+str(time)
        if(os.path.isfile(outfile)):
            print("Block exists.  Skipping.")
            continue
        print("Writing block:",time,"for diagram:",basename)
        #trajectory to exclude this line (loop over lines in the block)
        for excl in trajl:
            avg = 0
            #avg over non-excluded trajectories
            for t in trajl:
                if t == excl:
                    continue
                #current file
                #readf = "traj_"+str(t)+"_"+basename
                #grab current file's line corresponding to the block index,
                #block index is called time
                line = lc.getline("traj_"+str(t)+basename2, time+1).split()
                l = len(line)
                if l == 2:
                    avg += complex(line[1])
                elif l == 3:
                    avg += complex(float(line[1]),float(line[2]))
                elif l != 3 and l != 2: #if not summed over tsrc, for example
                    readf = "traj_"+str(t)+basename2
                    if not line:
                        print("Error: file '"+readf+"' not found")
                    else:
                        print("Error:  Bad filename:'"+readf+"', needs 2 or 3 columns:")
                        print("only",l,"columns found.")
                    exit(1)
                else:
                    print("How did you get here?")
                    exit(1)
            #line to write in the block file
            outl=complex('{0:.{1}f}'.format(avg/n,sys.float_info.dig))
            outl = str(outl.real)+" "+str(outl.imag)+"\n"
            with open(outfile, "a") as myfile:
               myfile.write(outl)

#get basename of file
def baseN(fn):
    m = re.search('traj_[B0-9]+_(.*)',fn)
    if not m:
        return None
    return m.group(1)

#return value: True if diagram is in every trajectory, otherwise False: we need a smaller trajectory list otherwise
def allTp(base,trajl):
    for t in trajl:
        test="traj_"+str(t)+"_"+base
        if not (os.path.isfile(test)):
            print("Missing:",test)
            return False
    return True

#gets the trajectory list for an individual base
def tlist(base,onlyfiles=None):
    trajl = set()
    if not onlyfiles:
        onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    for fn2 in onlyfiles:
        if baseN(fn2) == base:
            trajl.add(rf.traj(fn2))
    trajl-=set([None])
    print("Done getting trajectory list. N trajectories = "+str(len(trajl)))
    return trajl

def main():
    baseList = set()
    #setup the directory
    d='jackknifed_diagrams/'
    if not os.path.isdir(d):
        os.makedirs(d)
    #get the file list, exclude directories
    onlyfiles=[f for f in listdir('.') if isfile(join('.',f))]
    trajl = set()
    #get the max trajectory list
    for fn in onlyfiles:
        trajl.add(rf.traj(fn))
    trajl-=set([None])
    trajl=sorted([int(a) for a in trajl])
    print("Done getting max trajectory list. N trajectories = "+str(len(trajl)))
    for fn in onlyfiles:
        #get the basename of the file (non-trajectory information)
        base=baseN(fn)
        #continue if bad filename base (skip files that aren't data files), or if we've already hit this file's base
        if not base or base in baseList:
            continue
        print("Processing base:", base)
        baseList.add(base)

        #output directory for the jackknife blocks for this basename
        outdir=d+base
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        #Lt
        numlines = sum(1 for line in open(fn))

        #does base exist in all trajectories?
        if not allTp(base, trajl):
            print("Missing file(s); Attempting to write blocks with remainder.")
            write_blocks(tlist(base,onlyfiles),outdir,base,numlines)
        else:
            write_blocks(trajl,outdir,base,numlines)

    print("Done writing jackknife blocks.")
    exit(0)
    
if __name__ == "__main__":
    main()
