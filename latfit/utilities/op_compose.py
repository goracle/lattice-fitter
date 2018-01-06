#!/usr/bin/python3
"""Irrep projection."""
from math import sqrt
import os
import sys
import re
import numpy as np
from sum_blks import sum_blks
import read_file as rf

def momstr(psrc, psnk):
    """Take psrc and psnk and return a diagram string of the combination.
    """
    pipi = ''
    if len(psrc) == 2 and len(psnk) == 2:
        vertices = 4
    elif len(psrc) == 3 and isinstance(
            psrc[0], (np.integer, int)) and len(psnk) == 2:
        vertices = 3
        pipi = 'snk'
    elif len(psnk) == 3 and isinstance(
            psnk[0], (np.integer, int)) and len(psrc) == 2:
        vertices = 3
        pipi = 'src'
    elif len(psnk) == len(psrc) and len(psrc) == 3 and isinstance(
            psrc[0], (np.integer, int)):
        vertices = 2
    else:
        pstr = None
    if vertices == 4:
        pstr = 'mom1src'+rf.ptostr(
            psrc[0])+'_mom2src'+rf.ptostr(
                psrc[1])+'_mom1snk'+rf.ptostr(psnk[0])
    elif vertices == 3 and pipi == 'src':
        pstr = 'momsrc'+rf.ptostr(psrc[0])+'_momsnk'+rf.ptostr(psnk)
    elif vertices == 3 and pipi == 'snk':
        pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[0])
    elif vertices == 2:
        if psrc[0] != psnk[0] or psrc[1] != psnk[1] or psrc[2] != psnk[2]:
            pstr = None
        else:
            pstr = 'mom'+rf.ptostr(psrc)
    return pstr

A0 = [
    (1/8, 'pipi', [[1, 1, 1], [-1, -1, -1]]),
    (1/8, 'pipi', [[-1, 1, 1], [1, -1, -1]]),
    (1/8, 'pipi', [[1, -1, 1], [-1, 1, -1]]),
    (1/8, 'pipi', [[1, 1, -1], [-1, -1, 1]]),
    (1/8, 'pipi', [[1, -1, -1], [-1, 1, 1]]),
    (1/8, 'pipi', [[-1, 1, -1], [1, -1, 1]]),
    (1/8, 'pipi', [[-1, -1, 1], [1, 1, -1]]),
    (1/8, 'pipi', [[-1, -1, -1], [1, 1, 1]]),
]

A_1PLUS = [
    (1/sqrt(6), 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, 1], [0, 0, -1]]),
    (1/sqrt(6), 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, -1, 0], [0, 1, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, -1], [0, 0, 1]]),
    (1, 'S_pipi', [[0, 0, 0], [0, 0, 0]]),
    (1, 'sigma', [0, 0, 0]),
    (1/sqrt(12), 'UUpipi', [[0, 1, 1], [0, -1, -1]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, 1], [-1, 0, -1]]),
    (1/sqrt(12), 'UUpipi', [[1, 1, 0], [-1, -1, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, -1], [0, 1, 1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, -1], [1, 0, 1]]),
    (1/sqrt(12), 'UUpipi', [[-1, -1, 0], [1, 1, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, 1], [0, 1, -1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, 1], [1, 0, -1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 1, 0], [1, -1, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, 1, -1], [0, -1, 1]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, -1], [-1, 0, 1]]),
    (1/sqrt(12), 'UUpipi', [[1, -1, 0], [-1, 1, 0]]),
]

#A_1PLUS_sigma = [(1, 'sigma', [0, 0, 0])]

T_1_1MINUS = [
    (-1/2, 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (complex(0, -1/2), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (1/2, 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (complex(0, -1/2), 'pipi', [[0, -1, 0], [0, 1, 0]])]

T_1_3MINUS = [
    (1/2, 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (complex(0, -1/2), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (-1/2, 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (complex(0, 1/2), 'pipi', [[0, -1, 0], [0, 1, 0]])]

T_1_2MINUS = [
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, 0, -1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 0, 1]])]

OPLIST = {'A_1PLUS':A_1PLUS, 'T_1_1MINUS':T_1_1MINUS, 'T_1_3MINUS':T_1_3MINUS, 'T_1_2MINUS':T_1_2MINUS, 'A0':A0}
#OPLIST = {'A0':A0}
PART_LIST = set([])
for opa_out in OPLIST:
    for item in OPLIST[opa_out]:
        PART_LIST.add(item[1])

def partstr(srcpart, snkpart):
    """Get string from particle strings at source and sink"""
    if srcpart == snkpart and srcpart == 'pipi':
        particles = 'pipi'
    else:
        particles = snkpart+srcpart
    return particles

PART_COMBS = set([])
for srcout in PART_LIST:
    for snkout in PART_LIST:
        PART_COMBS.add(partstr(srcout, snkout))

def sepmod(dur):
    if not os.path.isdir(dur):
        if not os.path.isdir('sep4/'+dur):
            print("For op:", opa)
            print("dir", dur, "is missing")
            sys.exit(1)
    else:
        dur = 'sep4/'+dur
    return dur

def op_list(stype='ascii'):
    """Compose irrep operators at source and sink to do irrep projection.
    """
    projlist = {}
    for opa in OPLIST:
        coeffs_tuple = []
        for src in OPLIST[opa]:
            for snk in OPLIST[opa]:
                part_str = partstr(src[1], snk[1])
                coeff = src[0]*snk[0]
                p_str = momstr(src[2], snk[2])
                dur = part_str+"_"+p_str
                dur = re.sub('S_', '', dur)
                dur = re.sub('UU', '', dur)
                dur = re.sub('pipipipi', 'pipi', dur)
                if stype == 'ascii':
                    dur = sepmod(dur)
                coeffs_tuple.append((dur, coeff, part_str))
        coeffs_arr = []
        if stype == 'ascii':
            print("trying", opa)
        for parts in PART_COMBS:
            outdir = parts+"_"+opa
            coeffs_arr = [(tup[0], tup[1])
                          for tup in coeffs_tuple if tup[2] == parts]
            if not coeffs_arr:
                continue
            if stype == 'ascii':
                print("Doing", opa, "for particles", parts)
            if stype == 'ascii':
                sum_blks(outdir, coeffs_arr)
            else:
                projlist[outdir] = coeffs_arr
    if stype == 'ascii':
        print("End of operator list.")
    return projlist

def main():
    """Do irrep projection (main)"""
    op_list()

if __name__ == "__main__":
    main()
