#!/usr/bin/python3
"""Combine two disconnected bubbles."""

import sys
import numpy as np
import read_file as rf


def comb_dis(finsrc, finsnk, sep=0, starsnk=True, starsrc=False,):
    """Combine disconnected diagrams into an array.
    args = filename 1, filename 2
    returns an array indexed by tsrc, tdis
    """
    if isinstance(finsrc, str) and isinstance(finsnk, str):
        print("combining", finsrc, finsnk)
        src = rf.proc_vac(finsrc)
        snk = rf.proc_vac(finsnk)
    else:
        src = finsrc
        snk = finsnk
    len_t = len(src)
    if len_t != len(snk):
        print("Error: src snk combo dim mismatch.")
        sys.exit(1)
    out = np.zeros((len_t, len_t), dtype=complex)
    for tsrc in range(len_t):
        for tsnk in range(len_t):
            srcnum = src[tsrc].conjugate() if starsrc else src[tsrc]
            snknum = snk[(tsnk+sep) % len_t] if not starsnk else snk[
                (tsnk+sep) % len_t].conjugate()
            out.itemset(tsrc, (tsnk-tsrc+len_t) % len_t, srcnum*snknum)
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
        sys.exit(1)


if __name__ == "__main__":
    main()
