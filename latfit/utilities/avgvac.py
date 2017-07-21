#!/usr/bin/python3
"""Get avgerage of disconnected bubbles."""
import sys
from collections import deque
import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np
import read_file as rf
from traj_list import traj_list


def avg_vdis():
    """Do the averaging over trajecetories of the bubbles."""
    dur = 'summed_tsrc_diagrams/'
    if not os.path.isdir(dur):
        os.makedirs(dur)
    onlyfiles = [f for f in listdir('.') if isfile(join('.', f))]
    for filen in onlyfiles:
        fign = rf.figure(filen)
        if not fign in ['Vdis', 'scalar-bubble']:
            continue
        flag = 0
        outfile = get_outfile(filen, fign, dur)
        if os.path.isfile(outfile):
            flag = 1
        outerr = outfile+'_err'
        flag2 = 0
        if os.path.isfile(outfile):
            flag2 = 1
            if flag2 == flag:
                continue
        avg = np.array(proc_vac(filen))
        numt = 1
        #use this first file to bootstrap the rest of the files
        #(traj substitution)
        err_fact = deque()
        err_fact.append(avg)
        for traj in traj_list(onlyfiles):
            if str(traj) == rf.traj(filen) or int(traj) == rf.traj(filen):
                continue
            #slightly less stringent checking here on substitute:
            #no checking of anything following second underscore.
            #probably fine since user warned above.
            filen2 = re.sub('traj_([B0-9]+)_', 'traj_'+str(traj)+'_', filen)
            retarr = np.array(proc_vac(filen2))
            err_fact.append(retarr)
            avg += retarr
            numt += 1
        print("Number of configs to average over:",
              numt, "for outfile:", outfile)
        if flag == 0:
            rf.write_vec_str(avg/numt, outfile)
        if flag2 == 0:
            err_fact = np.array(err_fact)
            err_fact = err_fact-avg/numt
            err_fact = np.sqrt(np.diag(np.einsum(
                'ai,aj->ij', err_fact, err_fact))/(numt*(numt-1)))
            rf.write_vec_str(err_fact, outerr)
    print("Done writing Vdis averaged over trajectories.")
    return

def get_outfile(filen, fign, dur):
    """get output file name"""
    mom = rf.mom(filen)
    try:
        mom1, mom2 = mom
        momstr = "_mom1"+rf.ptostr(mom1)+"_mom2"+rf.ptostr(mom2)
    except ValueError:
        mom1 = mom
        momstr = "_mom"+rf.ptostr(mom1)
    sep = rf.sep(filen)
    if sep:
        sepstr = "_sep"+str(sep)
    else:
        sepstr = ""
    return dur+"Avg_"+fign+sepstr+momstr

def main():
    """Main; get the average bubble values."""
    avg_vdis()

if __name__ == "__main__":
    main()
