#!/usr/bin/python3
"""Write jackknife blocks from summed tsrc diagram files"""
import linecache as lc
from os import listdir
import os.path
from os.path import isfile, join
import sys
import re
import read_file as rf

def write_blocks(trajl, outdir, basename, len_t):
    """write jackknife blocks for this basename
    """
    basename2 = '_'+basename
    #number of trajectories - 1 (to avg over)
    num_configs = len(trajl)-1
    #loop over lines in the base, each gives a separate jk block of trajs
    for time in range(len_t):
        #block file name to append to
        #(a2a since this was written to do all-to-all analyses)
        outfile = outdir+"/a2a."+"jkblk.t"+str(time)
        if os.path.isfile(outfile):
            print("Block exists.  Skipping.")
            continue
        print("Writing block:", time, "for diagram:", basename)
        #trajectory to exclude this line (loop over lines in the block)
        for excl in trajl:
            avg = 0
            #avg over non-excluded trajectories
            for traj in trajl:
                if traj == excl:
                    continue
                #current file
                #readf = "traj_"+str(t)+"_"+basename
                #grab current file's line corresponding to the block index,
                #block index is called time
                line = lc.getline("traj_"+str(traj)+basename2, time+1).split()
                lsp = len(line)
                if lsp == 2:
                    avg += complex(line[1])
                elif lsp == 3:
                    avg += complex(float(line[1]), float(line[2]))
                elif lsp != 3 and lsp != 2: #if not summed over tsrc, for example
                    readf = "traj_"+str(traj)+basename2
                    if not line:
                        print("Error: file '"+readf+"' not found")
                    else:
                        print("Error:  Bad filename:'"+readf+"', needs 2 or 3 columns:")
                        print("only", lsp, "columns found.")
                    exit(1)
                else:
                    print("How did you get here?")
                    exit(1)
            #line to write in the block file
            avg = complex('{0:.{1}f}'.format(avg/num_configs,
                                             sys.float_info.dig))
            avg = str(avg.real)+" "+str(avg.imag)+"\n"
            with open(outfile, "a") as myfile:
                myfile.write(avg)

def base_name(filen):
    """get basename of file
    """
    mat = re.search('traj_[B0-9]+_(.*)', filen)
    if not mat:
        return None
    return mat.group(1)

def alltp(base, trajl):
    """return value: True if diagram is in every trajectory,
    otherwise False: we need a smaller trajectory list otherwise
    """
    for traj in trajl:
        test = "traj_"+str(traj)+"_"+base
        if not os.path.isfile(test):
            print("Missing:", test)
            return False
    return True

def tlist(base, onlyfiles=None):
    """gets the trajectory list for an individual base
    """
    trajl = set()
    if not onlyfiles:
        onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    for filen2 in onlyfiles:
        if base_name(filen2) == base:
            trajl.add(rf.traj(filen2))
    trajl -= set([None])
    print("Done getting trajectory list. N trajectories = "+str(len(trajl)))
    return trajl

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
    print("Done getting max trajectory list. N trajectories = "+str(len(trajl)))
    for filen in onlyfiles:
        #get the basename of the file (non-trajectory information)
        base = base_name(filen)
        #continue if bad filename base
        #(skip files that aren't data files),
        #or if we've already hit this file's base
        if not base or base in baselist:
            continue
        print("Processing base:", base)
        baselist.add(base)

        #output directory for the jackknife blocks for this basename
        outdir = dur+base
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        #len_t
        numlines = sum(1 for line in open(filen))

        #does base exist in all trajectories?
        if not alltp(base, trajl):
            print("Missing file(s); Attempting to write blocks with remainder.")
            write_blocks(tlist(base, onlyfiles), outdir, base, numlines)
        else:
            write_blocks(trajl, outdir, base, numlines)

    print("Done writing jackknife blocks.")
    exit(0)

if __name__ == "__main__":
    main()
