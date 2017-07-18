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

def write_blocks(trajl, outfiles, basename):
    """write jackknife blocks for this basename
    """
    basename2 = '_'+basename
    #number of trajectories - 1 (to avg over)
    num_configs = len(trajl)-1
    outfile2 = outdir+"/a2a.jkblk.t"
    #loop over lines in the base, each gives a separate jk block of trajs
    for time in range(len_t):
        #block file name to append to
        #(a2a since this was written to do all-to-all analyses)
        outfile=outfile2+str(time)
        if os.path.isfile(outfile):
            print("Block exists.  Skipping.")
            continue
        print("Writing block:", time, "for diagram:", basename)
        #trajectory to exclude this line (loop over lines in the block)
        outarr = np.zeros((len(trajl)), dtype=object)
        data = np.array([
            complex(lc.getline("traj_"+str(
                traj)+basename2, time+1).split()[1])
                for traj in trajl])
        
        for i, _ in enumerate(trajl):
            #avg = 0
            avg=np.mean(np.delete(data,i))
            #line to write in the block file
            avg = complex('{0:.{1}f}'.format(avg, sys.float_info.dig))
            avg = str(avg.real)+" "+str(avg.imag)+'\n'
            outarr[i] = avg
        with open(outfile, "a") as myfile:
            for line in outarr:
                myfile.write(line)

def base_name(filen):
    """get basename of file
    """
    mat = re.search('traj_[B0-9]+_(.*)', filen)
    if not mat:
        return None
    return mat.group(1)

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
        base = base_name(filen)
        if not base:
            continue
        traj = rf.traj(filen)
        if not traj:
            continue
        lookup_t.setdefault(base,set()).add(int(traj))
        #output directory for the jackknife blocks for this basename
        lookup.setdefault(base, []).append(filen)
    for base in lookup:
        numlines = sum(1 for line in open(lookup[base][0]))
        break
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
        if not lookup_t[base] == trajl_set:
            print("Missing file(s); Attempting to write blocks with remainder.")
            write_blocks(tl.traj_list(onlyfiles=lookup[base], base=base),
                         outdir, base, numlines)
        else:
            write_blocks(trajl, outdir, base, numlines)

    print("Done writing jackknife blocks.")
    sys.exit(0)

if __name__ == "__main__":
    main()
