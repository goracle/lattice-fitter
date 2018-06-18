#!/usr/bin/python3
"""Irrep projection."""
from math import sqrt
import os
import sys
import re
import numpy as np
from sum_blks import sum_blks
from write_discon import momtotal
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


A0_mom000 = [
    (1/8, 'pipi', [[1, 1, 1], [-1, -1, -1]]),
    (1/8, 'pipi', [[-1, 1, 1], [1, -1, -1]]),
    (1/8, 'pipi', [[1, -1, 1], [-1, 1, -1]]),
    (1/8, 'pipi', [[1, 1, -1], [-1, -1, 1]]),
    (1/8, 'pipi', [[1, -1, -1], [-1, 1, 1]]),
    (1/8, 'pipi', [[-1, 1, -1], [1, -1, 1]]),
    (1/8, 'pipi', [[-1, -1, 1], [1, 1, -1]]),
    (1/8, 'pipi', [[-1, -1, -1], [1, 1, 1]])
]

A_1PLUS_mom000 = [
    (1/sqrt(6), 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, 1], [0, 0, -1]]),
    (1/sqrt(6), 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, -1, 0], [0, 1, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, -1], [0, 0, 1]]),
    (1, 'S_pipi', [[0, 0, 0], [0, 0, 0]]),
    (1, 'sigma', [0, 0, 0]),
    (1, 'rho', [0, 0, 0]),
    (1/sqrt(12), 'UUpipi', [[0, 1, 1], [0, -1, -1]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, -1], [0, 1, 1]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, 1], [-1, 0, -1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, -1], [1, 0, 1]]),
    (1/sqrt(12), 'UUpipi', [[1, 1, 0], [-1, -1, 0]]),
    (1/sqrt(12), 'UUpipi', [[-1, -1, 0], [1, 1, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, 1], [0, 1, -1]]),
    (1/sqrt(12), 'UUpipi', [[0, 1, -1], [0, -1, 1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, 1], [1, 0, -1]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, -1], [-1, 0, 1]]),
    (1/sqrt(12), 'UUpipi', [[-1, 1, 0], [1, -1, 0]]),
    (1/sqrt(12), 'UUpipi', [[1, -1, 0], [-1, 1, 0]])
]

A1z_mom111 = [
    (1/sqrt(6), 'pipi', [[1, 0, 0], [0, 1, 1]]),
    (1/sqrt(6), 'pipi', [[0, 1, 1], [1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 1, 0], [1, 0, 1]]),
    (1/sqrt(6), 'pipi', [[1, 0, 1], [0, 1, 0]]),
    (1/sqrt(6), 'pipi', [[1, 1, 0], [0, 0, 1]]),
    (1/sqrt(6), 'pipi', [[0, 0, 1], [1, 1, 0]]),
    (1, 'sigma', [1, 1, 1]),
    (1/sqrt(6), 'UUpipi', [[-1, 1, 1], [2, 0, 0]]),
    (1/sqrt(6), 'UUpipi', [[2, 0, 0], [-1, 1, 1]]),
    (1/sqrt(6), 'UUpipi', [[0, 2, 0], [1, -1, 1]]),
    (1/sqrt(6), 'UUpipi', [[1, -1, 1], [0, 2, 0]]),
    (1/sqrt(6), 'UUpipi', [[0, 0, 2], [1, 1, -1]]),
    (1/sqrt(6), 'UUpipi', [[1, 1, -1], [0, 0, 2]]),
    (1/sqrt(12), 'U2pipi', [[0, 2, 1], [1, -1, 0]]),
    (1/sqrt(12), 'U2pipi', [[1, -1, 0], [0, 2, 1]]),
    (1/sqrt(12), 'U2pipi', [[1, 0, 2], [0, 1, -1]]),
    (1/sqrt(12), 'U2pipi', [[0, 1, -1], [1, 0, 2]]),
    (1/sqrt(12), 'U2pipi', [[2, 1, 0], [-1, 0, 1]]),
    (1/sqrt(12), 'U2pipi', [[-1, 0, 1], [2, 1, 0]]),
    (1/sqrt(12), 'U2pipi', [[2, 0, 1], [-1, 1, 0]]),
    (1/sqrt(12), 'U2pipi', [[-1, 1, 0], [2, 0, 1]]),
    (1/sqrt(12), 'U2pipi', [[1, 2, 0], [0, -1, 1]]),
    (1/sqrt(12), 'U2pipi', [[0, -1, 1], [1, 2, 0]]),
    (1/sqrt(12), 'U2pipi', [[0, 1, 2], [1, 0, -1]]),
    (1/sqrt(12), 'U2pipi', [[1, 0, -1], [0, 1, 2]]),
    (1/sqrt(6), 'U3pipi', [[2, 1, 1], [-1, 0, 0]]),
    (1/sqrt(6), 'U3pipi', [[-1, 0, 0], [2, 1, 1]]),
    (1/sqrt(6), 'U3pipi', [[1, 2, 1], [0, -1, 0]]),
    (1/sqrt(6), 'U3pipi', [[0, -1, 0], [1, 2, 1]]),
    (1/sqrt(6), 'U3pipi', [[1, 1, 2], [0, 0, -1]]),
    (1/sqrt(6), 'U3pipi', [[0, 0, -1], [1, 1, 2]])
]



A1x_mom011 = [
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]),
    (1, 'sigma', [0, 1, 1]),
    (0.5, 'UUpipi', [[1, 0, 0], [-1, 1, 1]]),
    (0.5, 'UUpipi', [[-1, 1, 1], [1, 0, 0]]),
    (0.5, 'UUpipi', [[-1, 0, 0], [1, 1, 1]]),
    (0.5, 'UUpipi', [[1, 1, 1], [-1, 0, 0]]),
    (0.5, 'U2pipi', [[1, 1, 0], [-1, 0, 1]]),
    (0.5, 'U2pipi', [[-1, 0, 1], [1, 1, 0]]),
    (0.5, 'U2pipi', [[1, 0, 1], [-1, 1, 0]]),
    (0.5, 'U2pipi', [[-1, 1, 0], [1, 0, 1]])
    (0.5, 'U3pipi', [[0, -1, 1], [0, 2, 0]]),
    (0.5, 'U3pipi', [[0, 2, 0], [0, -1, 1]]),
    (0.5, 'U3pipi', [[0, 0, 2], [0, 1, -1]]),
    (0.5, 'U3pipi', [[0, 1, -1], [0, 0, 2]]),
    (0.5, 'U4pipi', [[0, -1, 0], [0, 2, 1]]),
    (0.5, 'U4pipi', [[0, 2, 1], [0, -1, 0]]),
    (0.5, 'U4pipi', [[0, 0, -1], [0, 1, 2]]),
    (0.5, 'U4pipi', [[0, 1, 2], [0, 0, -1]]),
    (1/sqrt(8), 'U5pipi', [[-1, 2, 0], [1, -1, 1]]),
    (1/sqrt(8), 'U5pipi', [[1, -1, 1], [-1, 2, 0]]),
    (1/sqrt(8), 'U5pipi', [[-1, 0, 2], [1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[1, 1, -1], [-1, 0, 2]]),
    (1/sqrt(8), 'U5pipi', [[1, 2, 0], [-1, -1, 1]]),
    (1/sqrt(8), 'U5pipi', [[-1, -1, 1], [1, 2, 0]]),
    (1/sqrt(8), 'U5pipi', [[1, 0, 2], [-1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[-1, 1, -1], [1, 0, 2]]),
    (1/sqrt(8), 'U6pipi', [[-1, 2, 1], [1, -1, 0]]),
    (1/sqrt(8), 'U6pipi', [[1, -1, 0], [-1, 2, 1]]),
    (1/sqrt(8), 'U6pipi', [[-1, 1, 2], [1, 0, -1]]),
    (1/sqrt(8), 'U6pipi', [[1, 0, -1], [-1, 1, 2]]),
    (1/sqrt(8), 'U6pipi', [[1, 2, 1], [-1, -1, 0]]),
    (1/sqrt(8), 'U6pipi', [[-1, -1, 0], [1, 2, 1]]),
    (1/sqrt(8), 'U6pipi', [[1, 1, 2], [-1, 0, -1]]),
    (1/sqrt(8), 'U6pipi', [[-1, 0, -1], [1, 1, 2]])
]

A1z_mom001 = [
    (1/sqrt(2), 'S_pipi', [[0, 0, 0], [0, 0, 1]]),
    (1/sqrt(2), 'S_pipi', [[0, 0, 1], [0, 0, 0]]),
    (1/sqrt(8), 'pipi', [[-1, 0, 1], [1, 0, 0]]),
    (1/sqrt(8), 'pipi', [[1, 0, 0], [-1, 0, 1]]),
    (1/sqrt(8), 'pipi', [[0, -1, 1], [0, 1, 0]]),
    (1/sqrt(8), 'pipi', [[0, 1, 0], [0, -1, 1]]),
    (1/sqrt(8), 'pipi', [[1, 0, 1], [-1, 0, 0]]),
    (1/sqrt(8), 'pipi', [[-1, 0, 0], [1, 0, 1]]),
    (1/sqrt(8), 'pipi', [[0, 1, 1],[0, -1, 0]]),
    (1/sqrt(8), 'pipi', [[0, -1, 0], [0, 1, 1]]),
    (1, 'sigma', [0, 0, 1]),
    (1/sqrt(2), 'UUpipi', [[0, 0, 2], [0, 0, -1]]),
    (1/sqrt(2), 'UUpipi', [[0, 0, -1], [0, 0, 2]]),
    (1/sqrt(8), 'U2pipi', [[0, -1, 2], [0, 1, -1]]),
    (1/sqrt(8), 'U2pipi', [[0, 1, -1], [0, -1, 2]]),
    (1/sqrt(8), 'U2pipi', [[-1, 0, 2], [1, 0, -1]]),
    (1/sqrt(8), 'U2pipi', [[1, 0, -1], [-1, 0, 2]]),
    (1/sqrt(8), 'U2pipi', [[0, 1, 2], [0, -1, -1]]),
    (1/sqrt(8), 'U2pipi', [[0, -1, -1], [0, 1, 2]]),
    (1/sqrt(8), 'U2pipi', [[1, 0, 2], [-1, 0, -1]]),
    (1/sqrt(8), 'U2pipi', [[-1, 0, -1], [1, 0, 2]]),
    (1/sqrt(8), 'U3pipi', [[-2, 0, 1], [2, 0, 0]]),
    (1/sqrt(8), 'U3pipi', [[2, 0, 0], [-2, 0, 1]]),
    (1/sqrt(8), 'U3pipi', [[0, -2, 1], [0, 2, 0]]),
    (1/sqrt(8), 'U3pipi', [[0, 2, 0], [0, -2, 1]]),
    (1/sqrt(8), 'U3pipi', [[2, 0, 1], [-2, 0, 0]]),
    (1/sqrt(8), 'U3pipi', [[-2, 0, 0], [2, 0, 1]]),
    (1/sqrt(8), 'U3pipi', [[0, 2, 1], [0, -2, 0]]),
    (1/sqrt(8), 'U3pipi', [[0, -2, 0], [0, 2, 1]]),
    (1/sqrt(8), 'U4pipi', [[1, 1, -1], [-1, -1, 2]]),
    (1/sqrt(8), 'U4pipi', [[-1, -1, 2], [1, 1, -1]]),
    (1/sqrt(8), 'U4pipi', [[1, -1, -1], [-1, 1, 2]]),
    (1/sqrt(8), 'U4pipi', [[-1, 1, 2], [1, -1, -1]]),
    (1/sqrt(8), 'U4pipi', [[-1, 1, -1], [1, -1, 2]]),
    (1/sqrt(8), 'U4pipi', [[1, -1, 2], [-1, 1, -1]]),
    (1/sqrt(8), 'U4pipi', [[-1, -1, -1], [1, 1, 2]]),
    (1/sqrt(8), 'U4pipi', [[1, 1, 2], [-1, -1, -1]])
]

A1z = [
    (1/sqrt(2), 'S_pipi', [[0, 0, 0], [0, 0, 1]]),
    (1/sqrt(2), 'S_pipi', [[0, 0, 1], [0, 0, 0]]),
    (1/sqrt(8), 'pipi', [[-1, 0, 1], [1, 0, 0]]),
    (1/sqrt(8), 'pipi', [[0, -1, 1], [0, 1, 0]]),
    (1/sqrt(8), 'pipi', [[1, 0, 1], [-1, 0, 0]]),
    (1/sqrt(8), 'pipi', [[0, 1, 1],[0, -1, 0]]),
    (1/sqrt(8), 'pipi', [[1, 0, 0], [-1, 0, 1]]),
    (1/sqrt(8), 'pipi', [[0, 1, 0], [0, -1, 1]]),
    (1/sqrt(8), 'pipi', [[-1, 0, 0], [1, 0, 1]]),
    (1/sqrt(8), 'pipi', [[0, -1, 0], [0, 1, 1]]),
    (1, 'sigma', [0, 0, 1])
]


A1mz = [
    (0.5, 'S_pipi', [[0, 0, 0], [0, 0, -1]]),
    (0.5, 'S_pipi', [[0, 0, -1], [0, 0, 0]]),
    (0.5, 'pipi', [[1, 0, 0], [-1, 0, -1]]),
    (0.5, 'pipi', [[-1, 0, 0], [1, 0, -1]]),
    (0.5, 'pipi', [[0, -1, 0], [0, 1, -1]]),
    (0.5, 'pipi', [[0, -1, 0], [0, 1, -1]]),
    (1, 'sigma', [0, 0, -1])
]

A1y = [
    (0.5, 'S_pipi', [[0, 0, 0], [0, 1, 0]]),
    (0.5, 'S_pipi', [[0, 1, 0], [0, 0, 0]]),
    (0.5, 'pipi', [[1, 0, 0], [-1, 1, 0]]),
    (0.5, 'pipi', [[-1, 0, 0], [1, 1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [0, 1, 1]]),
    (0.5, 'pipi', [[0, 0, 1], [0, 1, -1]]),
    (1, 'sigma', [0, 1, 0])
]
A1my = [
    (0.5, 'S_pipi', [[0, 0, 0], [0, -1, 0]]),
    (0.5, 'S_pipi', [[0, -1, 0], [0, 0, 0]]),
    (0.5, 'pipi', [[1, 0, 0], [-1, -1, 0]]),
    (0.5, 'pipi', [[-1, 0, 0], [1, -1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [0, -1, 1]]),
    (0.5, 'pipi', [[0, 0, 1], [0, -1, -1]]),
    (1, 'sigma', [0, -1, 0])
]
A1x = [
    (0.5, 'S_pipi', [[1, 0, 0], [0, 0, 0]]),
    (0.5, 'S_pipi', [[0, 0, 0], [1, 0, 0]]),
    (0.5, 'pipi', [[1, -1, 0], [0, 1, 0]]),
    (0.5, 'pipi', [[1, 1, 0], [0, -1, 0]]),
    (0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]),
    (0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]),
    (1, 'sigma', [1, 0, 0])
]
A1mx = [
    (0.5, 'S_pipi', [[-1, 0, 0], [0, 0, 0]]),
    (0.5, 'S_pipi', [[0, 0, 0], [-1, 0, 0]]),
    (0.5, 'pipi', [[-1, -1, 0], [0, 1, 0]]),
    (0.5, 'pipi', [[-1, 1, 0], [0, -1, 0]]),
    (0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]),
    (0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]),
    (1, 'sigma', [-1, 0, 0])
]


A2 = [
    (1/sqrt(6), 'pipi', [[1, 0, 0], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 1, 0], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, 1], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[-1, 0, 0], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, -1, 0], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, -1], [0, 0, 0]]),
    (1, 'sigma', [1, 0, 0]),
    (1, 'sigma', [0, 1, 0]),
    (1, 'sigma', [0, 0, 1]),
    (1, 'sigma', [-1, 0, 0]),
    (1, 'sigma', [0, -1, 0]),
    (1, 'sigma', [0, 0, -1]),
    (1/sqrt(12), 'UUpipi', [[0, 1, 1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, 1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[1, 1, 0], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, -1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, -1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[-1, -1, 0], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, -1, 1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[-1, 0, 1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[-1, 1, 0], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[0, 1, -1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[1, 0, -1], [0, 0, 0]]),
    (1/sqrt(12), 'UUpipi', [[1, -1, 0], [0, 0, 0]])
]


# A_1PLUS_sigma = [(1, 'sigma', [0, 0, 0])]

T_1_1MINUS = [
    (1/sqrt(2), 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (1, 'rho', [0, 0, 0])
]
T_1_2MINUS = [
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 1, 0]]),
    (1, 'rho', [0, 0, 0])
]
T_1_3MINUS = [
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, 0, -1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 0, 1]]),
    (1, 'rho', [0, 0, 0])
]

# specify polarization info here
OPLIST = {'A_1PLUS_mom000': A_1PLUS_mom000,
          'A1z_mom001': A1z_mom001,
          'A1x_mom011': A1x_mom011,
          'A1_mom111': A1_mom111,
          'T_1_1MINUS?pol=1': T_1_1MINUS,
          'T_1_3MINUS?pol=2': T_1_3MINUS,
          'T_1_2MINUS?pol=3': T_1_2MINUS,
          'A0_mom000': A0_mom000,
          'A2': A2,
          'A1x': A1x, 'A1mx': A1mx,
          'A1y': A1y, 'A1my': A1my,
          'A1z': A1z, 'A1mz': A1mz,
}

AVG_ROWS = {
    'T_1_MINUS': ('T_1_1MINUS', 'T_1_2MINUS', 'T_1_3MINUS'),
    'A1': ('A1x', 'A1mx', 'A1y', 'A1my', 'A1z', 'A1mz')
}

# OPLIST = {'A0': A0}
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


def sepmod(dur, opa):
    """make different directory name for different time separations
    (probably what this does.)
    """
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
        momchk = rf.mom(opa) if 'mom' in momchk else None
        assert len(set(OPLIST[opa])) == len(OPLIST[opa]), "Duplicate operator found in "+str(opa)
        for src in OPLIST[opa]:
            for snk in OPLIST[opa]:
                assert cons_mom(src, snk, momchk), "operator does not conserve momentum "+str(opa)
                part_str = partstr(src[1], snk[1])
                coeff = src[0]*snk[0]
                p_str = momstr(src[2], snk[2])
                dur = part_str+"_"+p_str
                dur = re.sub('S_', '', dur)
                dur = re.sub('UU', '', dur)
                dur = re.sub('pipipipi', 'pipi', dur)
                if stype == 'ascii':
                    dur = sepmod(dur, opa)
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
                sum_blks(outdir, coeffs_arr)
            else:
                projlist[outdir] = coeffs_arr
    if stype == 'ascii':
        print("End of operator list.")
    return projlist

def cons_mom(src, snk, momtotal=None):
    """Check for momentum conservation"""
    psrc = momtotal(src[2])
    psnk = momtotal(snk[2])
    conssrcsnk = psrc[0] == psnk[0] and psrc[1] == psnk[1] and psrc[2] == psnk[2]
    if momtotal:
        check = momtotal[0] == psnk[0] and momtotal[1] == psnk[1] and momtotal[2] == psnk[2]
    else:
        check = True
    return check and conssrcsnk
    


def main():
    """Do irrep projection (main)"""
    l = op_list('hdf5')
    for i in l:
        print(i)
    print(l['rhorho_T_1_1MINUS'])


if __name__ == "__main__":
    main()
