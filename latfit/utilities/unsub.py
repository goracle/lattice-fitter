#!/usr/bin/python3
"""Unsubtract vacuum bubbles from jackknife blocks
"""
import sys
import os
import os.path
import numpy as np
from avg_vac import proc_vac

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
    print("edit this.  think carefully before proceeding")
    sys.exit(0)
    dur = '.'
    cwd = os.getcwd()
    onlydirs = [os.path.join(dur, o)
                for o in os.listdir(dur)
                if os.path.isdir(os.path.join(dur, o))]
    for lindex, pdir in enumerate(onlydirs):
        pdir = pdir[2:]
        os.chdir(cwd)
        subfile = '..'+'/AvgVac_'+pdir
        print("starting subfile:", subfile)
        if not os.path.isfile(subfile):
            continue
        subarr = proc_vac(subfile)
        print("Got unsub array from subfile:", subfile)
        os.chdir(pdir)
        for i, avgi in enumerate(subarr):
            outfile = 'a2a.jkblk.t'+str(i)
            mainarr = get_block_data(outfile, onlydirs[:lindex+1])
            mainarr += avgi #edit this line
            print("rewriting block:", outfile)
            with open(outfile, "w") as myfile:
                for numj in mainarr:
                    outl = complex('{0:.{1}f}'.format(
                        numj, sys.float_info.dig))
                    outl = str(outl.real)+" "+str(outl.imag)+"\n"
                    myfile.write(outl)

def main():
    """unsub main"""
    unsub()

if __name__ == "__main__":
    main()
