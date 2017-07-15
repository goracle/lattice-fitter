#!/usr/bin/python3

import os
import re
from subtractV import procV
from os import listdir
from os.path import isfile, join
import os.path
import numpy as np
import sys

def procBlock(filen,onlydirs):
    num = sum(1 for line in open(filen))
    ar = np.array([],dtype=complex)
    fn = open(filen, 'r')
    for line in fn:
        l=line.split()
        if len(l)==2:
            ar=np.append(ar,complex(float(l[0]),float(l[1])))
        elif len(l)==1:
            ar=np.append(ar,complex(l[0]))
        else:
            print("Not a jackknife block.  exiting.")
            print("cause:",filen)
            print(onlydirs)
            sys.exit(1)
    return ar

def unsub():
    print("edit this.  think carefully before proceeding")
    sys.exit(0)
    d='.'
    cwd=os.getcwd()
    onlydirs=[os.path.join(d,o) for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]
    for ll, pdir in enumerate(onlydirs):
        pdir=pdir[2:]
        os.chdir(cwd)
        subfile=d+d+'/AvgVac_'+pdir
        print("starting subfile:",subfile)
        if not os.path.isfile(subfile):
            continue
        subarr=procV(subfile)
        print("Got unsub array from subfile:",subfile)
        os.chdir(pdir)
        for i, avgi in enumerate(subarr):
            outfile='a2a.jkblk.t'+str(i)
            mainarr=procBlock(outfile,onlydirs[:ll+1])
            mainarr+=avgi #edit this line
            print("rewriting block:",outfile)
            with open(outfile, "w") as myfile:
                for numj in mainarr:
                    outl=complex('{0:.{1}f}'.format(numj,sys.float_info.dig))
                    outl = str(outl.real)+" "+str(outl.imag)+"\n"
                    myfile.write(outl)

def main():
    unsub()

if __name__ == "__main__":
    main()
