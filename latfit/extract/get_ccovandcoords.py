"""Extract precomputed cov. mat."""
from collections import defaultdict
import sys

from latfit.config import STYPE

def tree():
    """Return a multidimensional dict"""
    return defaultdict(tree)

def get_ccovandcoords(kfile, cxmin, cxmax):
    """Extract precomputed covariance matrix from file.
    Called by simple_proc_file.
    """
    cdict = tree()
    proccoords = []
    opensimp = open(kfile, 'r')
    try:
        for line in opensimp:
            pass
    except UnicodeDecodeError:
        print("***ERROR***")
        print("decoding error.")
        print("decoding format set to:", STYPE, "(in config)")
        print("Check input file to make sure it matches")
        sys.exit(1)
    for line in opensimp:
        try:
            cols = [float(p) for p in line.split()]
        except ValueError:
            print("ignored line: '", line, "'")
            continue
        if len(cols) == 2 and cxmin <= cols[0] <= cxmax:
            #only store coordinates in the valid range
            proccoords.append([cols[0], cols[1]])
            #two columns mean coordinate section, 3 covariance section
        elif len(cols) == 3:
            cdict[cols[0]][cols[1]] = cols[2]
        elif not len(cols) == 2 and not len(cols) == 3:
            print("***Error***")
            print("mangled file:")
            print(kfile)
            print("Expecting either two or three numbers per line.")
            print(len(cols), "found instead.")
            sys.exit(1)
    ccov = [[cdict[proccoords[ci][0]][proccoords[cj][0]]
             for ci in range(len(proccoords))]
            for cj in range(len(proccoords))]
    return ccov, proccoords
