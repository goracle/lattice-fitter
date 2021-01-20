#!/usr/bin/python3

import sys
import re
from glob import glob

def read(dim):
    """read line by line"""
    print("\ngetting fit info of energy level:", dim, "\n")
    filen = get_most_recent_slurm()
    fn1 = open(filen, 'r')
    start = False
    fitr = None
    mid = False
    prind = set()
    end = False
    pfit = None
    for line in fn1:
        line = line.rstrip()

        ## prelim lines

        if 'energy' and 'Loaded' in line:
            start = True
            assert not mid
            fitr = line.split("'")[0].split(':')[1][1:-2]
        if start:
            if 'dimension of gevp of interest:' in line:
                if dim in line:
                    mid = True
        if not start or not mid:
            continue

        pfit = fitr
        ## fit quality lines

        #lprin = len(prind)
        if 't^2' in line and 'config' not in line and 'Minimizer' not in line:
            if 'dof' in line and 'dev' not in line:
                continue
            print(line)
            prind.add(line)
        elif 'p-value' in line and 'config' not in line:
            print(line)
            prind.add(line)
        elif 'file name of saved plot' in line:
            start = False
            mid = False
            end = True
        #if lprin < len(prind) and False:
        #    print('line', line)
        #    print('prind:', prind)
    if end:
        print("fit range:", pfit)
        print("fit finished.")
    else:
        print("fit DID NOT finish. fit range:", pfit)


            

def get_most_recent_slurm():
    """Get the most recent slurm log 
    (return log name with highest job id)"""
    mmax = 0
    ret = None
    for item in glob("*slurm-*.out"):
        sub = re.sub('slurm-', '', item)
        sub = re.sub('.out', '', sub)
        val = int(sub)
        mmax = max(val, mmax)
        if str(mmax) in item:
            ret = item
    assert ret is not None, "no slurm logs found"
    return ret
        
    
if __name__ == '__main__':
    for dim in sys.argv[1:]:
        read(dim)
