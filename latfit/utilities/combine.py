"""Combine two disconnected bubbles."""
#!/usr/bin/python3

import sys
import numpy as np

def build_arr(fin):
    """Read the file and build the array"""
    filen = open(fin, 'r')
    out = np.array([], dtype=complex)
    for line in filen:
        lsp = line.split()
        try:
            out = np.append(out, complex(float(lsp[1]), float(lsp[2])))
        except IndexError:
            out = np.append(out, complex(lsp[1]))
    return np.array(out)

def find_dim(fin):
    """Find dimensions of the bubble in time direction."""
    return sum(1 for line in open(fin))

def comb_dis(finsrc, finsnk, sep=0, starsnk=False, starsrc=False):
    """Combine disconnected diagrams into an array.

    args = filename 1, filename 2
    returns an array indexed by tsrc, tdis
    """
    print("combining", finsrc, finsnk)
    len_t = find_dim(finsrc)
    if len_t != find_dim(finsnk):
        print("Error: dimension mismatch in combine operation.")
        sys.exit(1)
    out = np.zeros(shape=(len_t, len_t), dtype=complex)
    src = build_arr(finsrc)
    snk = build_arr(finsnk)
    for tsrc in range(len_t):
        for tsnk in range(len_t):
            srcnum = src[tsrc].conjugate() if starsrc else src[tsrc]
            snknum = snk[(tsnk+sep)%len_t] if not starsnk else snk[(tsnk+sep)%len_t].conjugate()
            out.itemset(tsrc, (tsnk-tsrc+len_t)%len_t, srcnum*snknum)
    return out

def main():
    """Main for combine"""
    args = len(sys.argv)
    arr = sys.argv
    if args == 3:
        comb_dis(arr[1], arr[2])
    elif args == 2:
        comb_dis(arr[1], arr[1])
    else:
        print("wrong num of args.  need two files to combine")
        exit(1)

if __name__ == "__main__":
    main()
