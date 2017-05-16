#!/usr/bin/python

import sys
import os
from read_file import *

def build_arr(fin):
    """Read the file and build the array"""
    fn = open(fin,'r')
    out = []
    for line in fn:
        l=line.split()
        try:
            out.append(complex(float(l[1]),float(l[2])))
        except:
            out.append(complex(l[1]))
    return out

def find_dim(fin):
    nl = sum(1 for line in open(fin))
    return nl

def comb_dis(fin1,fin2):
    """Combine disconnected diagrams into an array.

    args = filename 1, filename 2
    returns an array indexed by tsrc, tdis
    """
    print "combining", fin1, fin2
    Lt = find_dim(fin1)
    if Lt != find_dim(fin2):
        print "Error: dimension mismatch in combine operation."
        exit(1)
    out = np.zeros(shape=(Lt,Lt),dtype=complex)
    src = build_arr(fin1)
    snk = build_arr(fin2)
    for tsrc in range(Lt):
        for tsnk in range(Lt):
            out.itemset(tsrc,(tsnk-tsrc+Lt)%Lt,src[tsrc]*snk[tsnk])
    return out

def main():
    args=len(sys.argv)
    ar=sys.argv
    if args == 3:
        comb_dis(ar[1],ar[2])
    elif args == 2:
        comb_dis(ar[1],ar[1])
    else:
        print "wrong num of args.  need two files to combine"
        exit(1)

if __name__ == "__main__":
    main()
