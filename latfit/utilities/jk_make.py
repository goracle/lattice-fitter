#!/usr/bin/python3
"""Write jackknife blocks from summed tsrc diagram files"""
import linecache as lc
from os import listdir
import os.path
from os.path import isfile, join
import sys
import re
import numpy as np
import read_file as rf
#from traj_list import traj_list
import traj_list as tl

def write_blocks(trajl, basename, outfiles):
    """write jackknife blocks for this basename
    """
    basename2 = '_'+basename
    #loop over lines in the base, each gives a separate jk block of trajs
    for time, outfile in enumerate(outfiles):
        #block file name to append to
        #(a2a since this was written to do all-to-all analyses)
        if os.path.isfile(outfile):
            print("Block exists.  Skipping.")
            continue
        print("Writing block:", time, "for diagram:", basename)
        #trajectory to exclude this line (loop over lines in the block)
        outarr = np.zeros((len(trajl)), dtype=object)
        data = rf.get_linejk(traj, basename2, time, trajl)
        for i, _ in enumerate(trajl):
            #avg = 0
            avg=np.mean(np.delete(data,i))
            #line to write in the block file
            avg = complex('{0:.{1}f}'.format(avg, sys.float_info.dig))
            avg = str(avg.real)+" "+str(avg.imag)+'\n'
            outarr[i] = avg
        rf.write_block(outarr, outfile, already_checked=True)

def main():
    """Make jackknife blocks (main)"""
    baselist = set()
    #setup the directory
    dur = 'jackknifed_diagrams/'
    if not os.path.isdir(dur):
        os.makedirs(dur)
    #get the file list, exclude directories
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    trajl = set()
    #get the max trajectory list
    for filen in onlyfiles:
        trajl.add(rf.traj(filen))
    trajl -= set([None])
    trajl = sorted([int(a) for a in trajl])
    trajl_set = set(trajl)
    print("Done getting max trajectory list. N trajectories = "+str(len(trajl)))
    lookup = {}
    lookup_t = {}
    for filen in onlyfiles:
        #get the basename of the file (non-trajectory information)
        base = rf.basename(filen)
        if not base:
            continue
        traj = rf.traj(filen)
        if not traj:
            continue
        lookup_t.setdefault(base,set()).add(int(traj))
        #output directory for the jackknife blocks for this basename
        lookup.setdefault(base, []).append(filen)
    for base in lookup:
        numlines = rf.numlines(lookup[base][0])
        break
    base_blks = ['/a2a.jkblk.t'+str(i) for i in range(numlines)]
    lenb = len(lookup)
    for ibase, base in enumerate(lookup):
        outdir = dur+base
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        #len_t
        #continue if bad filename base
        #(skip files that aren't data files),
        #or if we've already hit this file's base
        print("Processing base:", base, ibase, "of", lenb)

        #does base exist in all trajectories?
        outfiles = [outdir+i for i in base_blks]
        if not lookup_t[base] == trajl_set:
            print("Missing file(s); Attempting to write blocks with remainder.")
            write_blocks(tl.traj_list(onlyfiles=lookup[base], base=base),
                         base, outfiles)
        else:
            write_blocks(trajl, base, outfiles)

    print("Done writing jackknife blocks.")
    sys.exit(0)

if __name__ == "__main__":
    main()
