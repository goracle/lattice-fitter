#!/usr/bin/python3
"""Unsubtract vacuum bubbles from jackknife blocks
"""
import sys
import os
import os.path
import numpy as np
from avgvac import proc_vac
import re
import read_file as rf

def get_block_data(filen, onlydirs):
    """Get array of jackknife block data (from single time slice file, e.g.)
    """
    retarr = np.array([], dtype=complex)
    filen = open(filen, 'r')
    for line in filen:
        lsp = line.split()
        if len(lsp) == 2:
            retarr = np.append(retarr, complex(float(lsp[0]), float(lsp[1])))
        elif len(lsp) == 1:
            retarr = np.append(retarr, complex(lsp[0]))
        else:
            print("Not a jackknife block.  exiting.")
            print("cause:", filen)
            print(onlydirs)
            sys.exit(1)
    return retarr

def unsub():
    """Do the un-subtraction of vacuum bubbles on jk blks"""
    dur = '.'
    onlydirs = [os.path.join(dur, o)
                for o in os.listdir(dur)
                if os.path.isdir(os.path.join(dur, o))]
    for lindex, datadir in enumerate(onlydirs):
        #get rid of preceding slash
        datadir = datadir[2:]
        subfile = 'AvgVac_'+datadir
        if not os.path.isfile(subfile):
            continue
        subarr = proc_vac(subfile)
        #average buble with respect to time
        mean_bubble = np.repeat(np.mean(subarr), len(subarr))
        outdirs = [datadir+'_sub', datadir+'_avgsub']
        if not os.path.isdir(outdirs[0]):
            os.makedirs(outdirs[0])
        if not os.path.isdir(outdirs[1]):
            os.makedirs(outdirs[1])
        coeffs = [-1, -1]
        subarrs = [subarr, mean_bubble]
        write_blocks_todirs(datadir, outdirs, subarrs,
                            coeffs, onlydirs[:lindex+1])

def write_blocks_todirs(datadir, outdirs, subarrs, coeffs, onlydirs):
    """Get jackknife blocks, then for each one
    write several versions of it"""
    for i, avgs in enumerate(zip(*subarrs)):
        outfile = 'a2a.jkblk.t'+str(i)
        mainarr = get_block_data(datadir+'/'+outfile, onlydirs)
        for avgi, cfm, odir in zip(coeffs, avgs, outdirs):
            wfn = odir+'/'+outfile
            if os.path.isfile(wfn):
                print("Skipping:", wfn)
                print("File exists.")
            else:
                print("Writing block:", i, "for diagram:", odir)
                rf.write_block(mainarr+cfm*avgi, wfn, already_checked=True)

def main():
    """unsub main"""
    unsub()

if __name__ == "__main__":
    main()
