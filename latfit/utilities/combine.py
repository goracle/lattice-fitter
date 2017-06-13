#!/usr/bin/python3

import sys
import numpy as np

def build_arr(fin):
    """Read the file and build the array"""
    fn = open(fin,'r')
    out = np.array([],dtype=complex)
    for line in fn:
        l=line.split()
        try:
            out=np.append(out,complex(float(l[1]),float(l[2])))
        except:
            out=np.append(out,complex(l[1]))
    return np.array(out)

def find_dim(fin):
    nl = sum(1 for line in open(fin))
    return nl

def comb_dis(finSrc,finSnk,sep=0,starSnk=False,starSrc=False):
    """Combine disconnected diagrams into an array.

    args = filename 1, filename 2
    returns an array indexed by tsrc, tdis
    """
    print("combining", finSrc, finSnk)
    Lt = find_dim(finSrc)
    if Lt != find_dim(finSnk):
        print("Error: dimension mismatch in combine operation.")
        exit(1)
    out = np.zeros(shape=(Lt,Lt),dtype=complex)
    src = build_arr(finSrc)
    snk = build_arr(finSnk)
    for tsrc in range(Lt):
        for tsnk in range(Lt):
            srcNum=src[tsrc].conjugate() if starSrc==True else src[tsrc]
            snkNum=snk[(tsnk+sep)%Lt] if starSnk==False else snk[(tsnk+sep)%Lt].conjugate()
            out.itemset(tsrc,(tsnk-tsrc+Lt)%Lt,srcNum*snkNum)
    return out

def main():
    args=len(sys.argv)
    ar=sys.argv
    if args == 3:
        comb_dis(ar[1],ar[2])
    elif args == 2:
        comb_dis(ar[1],ar[1])
    else:
        print("wrong num of args.  need two files to combine")
        exit(1)

if __name__ == "__main__":
    main()
