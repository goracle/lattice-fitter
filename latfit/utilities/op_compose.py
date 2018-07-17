#!/usr/bin/python3
"""Irrep projection."""
from math import sqrt
import os
import sys
import re
import numpy as np
from sum_blks import sum_blks
import write_discon as wd
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
            #psrc[0])+'_mom2src'+rf.ptostr(
            #    psrc[1])+'_mom1snk'+rf.ptostr(psnk[0])
            psrc[1])+'_mom2src'+rf.ptostr(
                psrc[0])+'_mom1snk'+rf.ptostr(psnk[1])
    elif vertices == 3 and pipi == 'src':
        #pstr = 'momsrc'+rf.ptostr(psrc[0])+'_momsnk'+rf.ptostr(psnk)
        pstr = 'momsrc'+rf.ptostr(psrc[1])+'_momsnk'+rf.ptostr(psnk)
    elif vertices == 3 and pipi == 'snk':
        #pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[0])
        pstr = 'momsrc'+rf.ptostr(psrc)+'_momsnk'+rf.ptostr(psnk[1])
    elif vertices == 2:
        if psrc[0] != psnk[0] or psrc[1] != psnk[1] or psrc[2] != psnk[2]:
            pstr = None
        else:
            pstr = 'mom'+rf.ptostr(psrc)
    return pstr

# A_1PLUS_mom000 dim = 5
# A1_mom1 dim = 4
# A1_mom11 dim = 5
# A1_mom111 dim = 3


A_1PLUS_mom000 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 0, 0]]),
    (1/sqrt(6), 'pipi', [[1, 0, 0], [-1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[-1, 0, 0], [1, 0, 0]]),
    (1/sqrt(6), 'pipi', [[0, 1, 0], [0, -1, 0]]),
    (1/sqrt(6), 'pipi', [[0, -1, 0], [0, 1, 0]]),
    (1/sqrt(6), 'pipi', [[0, 0, 1], [0, 0, -1]]),
    (1/sqrt(6), 'pipi', [[0, 0, -1], [0, 0, 1]]),
    (1, 'sigma', [0, 0, 0]),
    # (1, 'rho', [0, 0, 0]),
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
    (1/sqrt(12), 'UUpipi', [[1, -1, 0], [-1, 1, 0]]),
    (1/sqrt(8), 'U2pipi', [[1, 1, 1], [-1, -1, -1]]),
    (1/sqrt(8), 'U2pipi', [[-1, -1, -1], [1, 1, 1]]),
    (1/sqrt(8), 'U2pipi', [[-1, 1, 1], [1, -1, -1]]),
    (1/sqrt(8), 'U2pipi', [[1, -1, -1], [-1, 1, 1]]),
    (1/sqrt(8), 'U2pipi', [[1, -1, 1], [-1, 1, -1]]),
    (1/sqrt(8), 'U2pipi', [[-1, 1, -1], [1, -1, 1]]),
    (1/sqrt(8), 'U2pipi', [[1, 1, -1], [-1, -1, 1]]),
    (1/sqrt(8), 'U2pipi', [[-1, -1, 1], [1, 1, -1]]),
]

A1z_mom001 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 0, 1]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[1, 0, 0], [-1, 0, 1]]),
    (0.5, 'pipi', [[0, 1, 0], [0, -1, 1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[-1, 0, 0], [1, 0, 1]]),
    (0.5, 'pipi', [[0, -1, 0], [0, 1, 1]]),
    (0.5, 'UUpipi', [[1, 1, 0], [-1, -1, 1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 1, 1]]),
    (0.5, 'UUpipi', [[-1, 1, 0], [1, -1, 1]]),
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 1, 1]]),
    (1, 'sigma', [0, 0, 1]), # sigma ?
]
"""
    (1/sqrt(2), 'U2pipi', [[0, 0, 2], [0, 0, -1]]), # 3 unit rel (.8388)
    (1/sqrt(2), 'U2pipi', [[0, 0, -1], [0, 0, 2]]),
    (1/sqrt(8), 'U3pipi', [[0, -1, 2], [0, 1, -1]]), # 5 unit rel (1)
    (1/sqrt(8), 'U3pipi', [[0, 1, -1], [0, -1, 2]]),
    (1/sqrt(8), 'U3pipi', [[-1, 0, 2], [1, 0, -1]]),
    (1/sqrt(8), 'U3pipi', [[1, 0, -1], [-1, 0, 2]]),
    (1/sqrt(8), 'U3pipi', [[0, 1, 2], [0, -1, -1]]),
    (1/sqrt(8), 'U3pipi', [[0, -1, -1], [0, 1, 2]]),
    (1/sqrt(8), 'U3pipi', [[1, 0, 2], [-1, 0, -1]]),
    (1/sqrt(8), 'U3pipi', [[-1, 0, -1], [1, 0, 2]]),
    (1/sqrt(8), 'U4pipi', [[-2, 0, 1], [2, 0, 0]]), # 5 unit rel (1.14)
    (1/sqrt(8), 'U4pipi', [[2, 0, 0], [-2, 0, 1]]),
    (1/sqrt(8), 'U4pipi', [[0, -2, 1], [0, 2, 0]]),
    (1/sqrt(8), 'U4pipi', [[0, 2, 0], [0, -2, 1]]),
    (1/sqrt(8), 'U4pipi', [[2, 0, 1], [-2, 0, 0]]),
    (1/sqrt(8), 'U4pipi', [[-2, 0, 0], [2, 0, 1]]),
    (1/sqrt(8), 'U4pipi', [[0, 2, 1], [0, -2, 0]]),
    (1/sqrt(8), 'U4pipi', [[0, -2, 0], [0, 2, 1]]),
    ]

    (1/sqrt(8), 'U5pipi', [[1, 1, -1], [-1, -1, 2]]), # 7 unit rel (1.13)
    (1/sqrt(8), 'U5pipi', [[-1, -1, 2], [1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[1, -1, -1], [-1, 1, 2]]),
    (1/sqrt(8), 'U5pipi', [[-1, 1, 2], [1, -1, -1]]),
    (1/sqrt(8), 'U5pipi', [[-1, 1, -1], [1, -1, 2]]),
    (1/sqrt(8), 'U5pipi', [[1, -1, 2], [-1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[-1, -1, -1], [1, 1, 2]]),
    (1/sqrt(8), 'U5pipi', [[1, 1, 2], [-1, -1, -1]])
]
"""

A1z_mom00_1 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 0, -1]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[1, 0, 0], [-1, 0, -1]]),
    (0.5, 'pipi', [[0, 1, 0], [0, -1, -1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[-1, 0, 0], [1, 0, -1]]),
    (0.5, 'pipi', [[0, -1, 0], [0, 1, -1]]),
    (0.5, 'UUpipi', [[1, 1, 0], [-1, -1, -1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 1, -1]]),
    (0.5, 'UUpipi', [[-1, 1, 0], [1, -1, -1]]),
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 1, -1]]),
    (1, 'sigma', [0, 0, -1]), # sigma ?
]

A1y_mom010 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 1, 0]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[1, 0, 0], [-1, 1, 0]]),
    (0.5, 'pipi', [[0, 0, 1], [0, 1, -1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[-1, 0, 0], [1, 1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [0, 1, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [-1, 1, -1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[1, 0, -1], [-1, 1, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [1, 1, -1]]),
    (0.5, 'UUpipi', [[-1, 0, -1], [1, 1, 1]]),
    (1, 'sigma', [0, 1, 0]), # sigma ?
]

A1y_mom0_10 = [
    (1, 'S_pipi', [[0, 0, 0], [0, -1, 0]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[1, 0, 0], [-1, -1, 0]]),
    (0.5, 'pipi', [[0, 0, 1], [0, -1, -1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[-1, 0, 0], [1, -1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [0, -1, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [-1, -1, -1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[1, 0, -1], [-1, -1, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [1, -1, -1]]),
    (0.5, 'UUpipi', [[-1, 0, -1], [1, -1, 1]]),
    (1, 'sigma', [0, -1, 0]), # sigma ?
]


A1x_mom100 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 0, 0]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[0, 1, 0], [1, -1, 0]]),
    (0.5, 'pipi', [[0, 0, 1], [1, 0, -1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[0, -1, 0], [1, 1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[0, 1, 1], [1, -1, -1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[0, 1, -1], [1, -1, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [1, 1, -1]]),
    (0.5, 'UUpipi', [[0, -1, -1], [1, 1, 1]]),
    (1, 'sigma', [1, 0, 0]), # sigma ?
]

A1x_mom_100 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 0, 0]]), # 1 unit rel (.3)
    (0.5, 'pipi', [[0, 1, 0], [-1, -1, 0]]),
    (0.5, 'pipi', [[0, 0, 1], [-1, 0, -1]]), # 3 unit rel (.693)
    (0.5, 'pipi', [[0, -1, 0], [-1, 1, 0]]),
    (0.5, 'pipi', [[0, 0, -1], [-1, 0, 1]]),
    (0.5, 'UUpipi', [[0, 1, 1], [-1, -1, -1]]), # 6 unit rel (.870)
    (0.5, 'UUpipi', [[0, 1, -1], [-1, -1, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 1, -1]]),
    (0.5, 'UUpipi', [[0, -1, -1], [-1, 1, 1]]),
    (1, 'sigma', [-1, 0, 0]), # sigma ?
]




A1x_mom011 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 1, 1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]),
    (1, 'sigma', [0, 1, 1]), # sigma
    (1/sqrt(2), 'UUpipi', [[1, 0, 0], [-1, 1, 1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[-1, 0, 0], [1, 1, 1]]),
    (0.5, 'U2pipi', [[1, 1, 0], [-1, 0, 1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, 1], [1, 1, 0]]),
    (0.5, 'U2pipi', [[1, 0, 1], [-1, 1, 0]]),
    (0.5, 'U2pipi', [[-1, 1, 0], [1, 0, 1]]),
]
"""    (0.5, 'U3pipi', [[0, -1, 1], [0, 2, 0]]), # 4 units rel (.9378)
    (0.5, 'U3pipi', [[0, 2, 0], [0, -1, 1]]),
    (0.5, 'U3pipi', [[0, 0, 2], [0, 1, -1]]),
    (0.5, 'U3pipi', [[0, 1, -1], [0, 0, 2]]),
    (0.5, 'U4pipi', [[0, -1, 0], [0, 2, 1]]), # 4 units rel (.897)
    (0.5, 'U4pipi', [[0, 2, 1], [0, -1, 0]]),
    (0.5, 'U4pipi', [[0, 0, -1], [0, 1, 2]]),
    (0.5, 'U4pipi', [[0, 1, 2], [0, 0, -1]]),
]

    (1/sqrt(8), 'U5pipi', [[-1, 2, 0], [1, -1, 1]]), # 6 units rel (1.07)
    (1/sqrt(8), 'U5pipi', [[1, -1, 1], [-1, 2, 0]]),
    (1/sqrt(8), 'U5pipi', [[-1, 0, 2], [1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[1, 1, -1], [-1, 0, 2]]),
    (1/sqrt(8), 'U5pipi', [[1, 2, 0], [-1, -1, 1]]),
    (1/sqrt(8), 'U5pipi', [[-1, -1, 1], [1, 2, 0]]),
    (1/sqrt(8), 'U5pipi', [[1, 0, 2], [-1, 1, -1]]),
    (1/sqrt(8), 'U5pipi', [[-1, 1, -1], [1, 0, 2]]),
    (1/sqrt(8), 'U6pipi', [[-1, 2, 1], [1, -1, 0]]), # 6 units rel (1.05)
    (1/sqrt(8), 'U6pipi', [[1, -1, 0], [-1, 2, 1]]),
    (1/sqrt(8), 'U6pipi', [[-1, 1, 2], [1, 0, -1]]),
    (1/sqrt(8), 'U6pipi', [[1, 0, -1], [-1, 1, 2]]),
    (1/sqrt(8), 'U6pipi', [[1, 2, 1], [-1, -1, 0]]),
    (1/sqrt(8), 'U6pipi', [[-1, -1, 0], [1, 2, 1]]),
    (1/sqrt(8), 'U6pipi', [[1, 1, 2], [-1, 0, -1]]),
    (1/sqrt(8), 'U6pipi', [[-1, 0, -1], [1, 1, 2]])
"""

A1x_mom0_11 = [
    (1, 'S_pipi', [[0, 0, 0], [0, -1, 1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, 1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, -1, 0]]),
    (1, 'sigma', [0, -1, 1]), # sigma
    (1/sqrt(2), 'UUpipi', [[1, 0, 0], [-1, -1, 1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[-1, 0, 0], [1, -1, 1]]),
    (0.5, 'U2pipi', [[1, -1, 0], [-1, 0, 1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, 1], [1, -1, 0]]),
    (0.5, 'U2pipi', [[1, 0, 1], [-1, -1, 0]]),
    (0.5, 'U2pipi', [[-1, -1, 0], [1, 0, 1]]),
]

A1x_mom01_1 = [
    (1, 'S_pipi', [[0, 0, 0], [0, 1, -1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, -1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, -1], [0, 1, 0]]),
    (1, 'sigma', [0, 1, -1]), # sigma
    (1/sqrt(2), 'UUpipi', [[1, 0, 0], [-1, 1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[-1, 0, 0], [1, 1, -1]]),
    (0.5, 'U2pipi', [[1, 1, 0], [-1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, -1], [1, 1, 0]]),
    (0.5, 'U2pipi', [[1, 0, -1], [-1, 1, 0]]),
    (0.5, 'U2pipi', [[-1, 1, 0], [1, 0, -1]]),
]

A1x_mom0_1_1 = [
    (1, 'S_pipi', [[0, 0, 0], [0, -1, -1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, -1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, -1], [0, -1, 0]]),
    (1, 'sigma', [0, -1, -1]), # sigma
    (1/sqrt(2), 'UUpipi', [[1, 0, 0], [-1, -1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[-1, 0, 0], [1, -1, -1]]),
    (0.5, 'U2pipi', [[1, -1, 0], [-1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, -1], [1, -1, 0]]),
    (0.5, 'U2pipi', [[1, 0, -1], [-1, -1, 0]]),
    (0.5, 'U2pipi', [[-1, -1, 0], [1, 0, -1]]),
]


A1y_mom101 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 0, 1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, 1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, 1], [1, 0, 0]]),
    (1, 'sigma', [1, 0, 1]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 1, 0], [1, -1, 1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, -1, 0], [1, 1, 1]]),
    (0.5, 'U2pipi', [[1, 1, 0], [0, -1, 1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[0, -1, 1], [1, 1, 0]]),
    (0.5, 'U2pipi', [[0, 1, 1], [1, -1, 0]]),
    (0.5, 'U2pipi', [[1, -1, 0], [0, 1, 1]]),
]

A1y_mom_101 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 0, 1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, 1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, 1], [-1, 0, 0]]),
    (1, 'sigma', [-1, 0, 1]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 1, 0], [-1, -1, 1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, -1, 0], [-1, 1, 1]]),
    (0.5, 'U2pipi', [[-1, 1, 0], [0, -1, 1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[0, -1, 1], [-1, 1, 0]]),
    (0.5, 'U2pipi', [[0, 1, 1], [-1, -1, 0]]),
    (0.5, 'U2pipi', [[-1, -1, 0], [0, 1, 1]]),
]

A1y_mom10_1 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 0, -1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, -1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, -1], [1, 0, 0]]),
    (1, 'sigma', [1, 0, -1]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 1, 0], [1, -1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, -1, 0], [1, 1, -1]]),
    (0.5, 'U2pipi', [[1, 1, 0], [0, -1, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[0, -1, -1], [1, 1, 0]]),
    (0.5, 'U2pipi', [[0, 1, -1], [1, -1, 0]]),
    (0.5, 'U2pipi', [[1, -1, 0], [0, 1, -1]]),
]

A1y_mom_10_1 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 0, -1]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, -1]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[0, 0, -1], [-1, 0, 0]]),
    (1, 'sigma', [-1, 0, -1]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 1, 0], [-1, -1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, -1, 0], [-1, 1, -1]]),
    (0.5, 'U2pipi', [[-1, 1, 0], [0, -1, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[0, -1, -1], [-1, 1, 0]]),
    (0.5, 'U2pipi', [[0, 1, -1], [-1, -1, 0]]),
    (0.5, 'U2pipi', [[-1, -1, 0], [0, 1, -1]]),
]

A1z_mom110 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 1, 0]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, 1, 0], [1, 0, 0]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, 1, 0]]),
    (1, 'sigma', [1, 1, 0]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 0, 1], [1, 1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, 0, -1], [1, 1, 1]]),
    (0.5, 'U2pipi', [[0, 1, 1], [1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[1, 0, -1], [0, 1, 1]]),
    (0.5, 'U2pipi', [[1, 0, 1], [0, 1, -1]]),
    (0.5, 'U2pipi', [[0, 1, -1], [1, 0, 1]]),
]

A1z_mom_110 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 1, 0]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, 1, 0], [-1, 0, 0]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 1, 0]]),
    (1, 'sigma', [-1, 1, 0]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 0, 1], [-1, 1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, 0, -1], [-1, 1, 1]]),
    (0.5, 'U2pipi', [[0, 1, 1], [-1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, -1], [0, 1, 1]]),
    (0.5, 'U2pipi', [[-1, 0, 1], [0, 1, -1]]),
    (0.5, 'U2pipi', [[0, 1, -1], [-1, 0, 1]]),
]

A1z_mom1_10 = [
    (1, 'S_pipi', [[0, 0, 0], [1, -1, 0]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, -1, 0], [1, 0, 0]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, -1, 0]]),
    (1, 'sigma', [1, -1, 0]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 0, 1], [1, -1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, 0, -1], [1, -1, 1]]),
    (0.5, 'U2pipi', [[0, -1, 1], [1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[1, 0, -1], [0, -1, 1]]),
    (0.5, 'U2pipi', [[1, 0, 1], [0, -1, -1]]),
    (0.5, 'U2pipi', [[0, -1, -1], [1, 0, 1]]),
]

A1z_mom_1_10 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, -1, 0]]), # 2 units rel (efree_24c=.396)
    (1/sqrt(2), 'pipi', [[0, -1, 0], [-1, 0, 0]]), # 2 units rel (.594)
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, -1, 0]]),
    (1, 'sigma', [-1, -1, 0]), # sigma
    (1/sqrt(2), 'UUpipi', [[0, 0, 1], [-1, -1, -1]]), # 4 units rel (efree_24=.77)
    (1/sqrt(2), 'UUpipi', [[0, 0, -1], [-1, -1, 1]]),
    (0.5, 'U2pipi', [[0, -1, 1], [-1, 0, -1]]), # 4 units rel (.792)
    (0.5, 'U2pipi', [[-1, 0, -1], [0, -1, 1]]),
    (0.5, 'U2pipi', [[-1, 0, 1], [0, -1, -1]]),
    (0.5, 'U2pipi', [[0, -1, -1], [-1, 0, 1]]),
]


# 0 neg
A1_mom111 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 1, 1]]), # 3 units rel (.614)
    (1/sqrt(3), 'pipi', [[1, 0, 0], [0, 1, 1]]), # 3 units rel (.692)
    (1/sqrt(3), 'pipi', [[0, 1, 0], [1, 0, 1]]),
    (1/sqrt(3), 'pipi', [[0, 0, 1], [1, 1, 0]]),
    (1, 'sigma', [1, 1, 1]), # sigma ?
]
# 1 neg
A1x_mom_111 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 1, 1]]), # 3 units rel (.614)
    (1/sqrt(3), 'pipi', [[-1, 0, 0], [0, 1, 1]]), # 3 units rel (.692)
    (1/sqrt(3), 'pipi', [[0, 1, 0], [-1, 0, 1]]),
    (1/sqrt(3), 'pipi', [[0, 0, 1], [-1, 1, 0]]),
    (1, 'sigma', [-1, 1, 1]), # sigma
]
A1y_mom1_11 = [
    (1, 'S_pipi', [[0, 0, 0], [1, -1, 1]]),
    (0.5773502691896258, 'pipi', [[0, -1, 0], [1, 0, 1]]),
    (0.5773502691896258, 'pipi', [[1, 0, 0], [0, -1, 1]]),
    (0.5773502691896258, 'pipi', [[0, 0, 1], [1, -1, 0]]),
    (1, 'sigma', [1, -1, 1])
]

A1z_mom11_1 = [
    (1, 'S_pipi', [[0, 0, 0], [1, 1, -1]]),
    (0.5773502691896258, 'pipi', [[0, 0, -1], [1, 1, 0]]),
    (0.5773502691896258, 'pipi', [[0, 1, 0], [1, 0, -1]]),
    (0.5773502691896258, 'pipi', [[1, 0, 0], [0, 1, -1]]),
    (1, 'sigma', [1, 1, -1])
]
# 2 neg
A1x_mom1_1_1 = [
    (1, 'S_pipi', [[0, 0, 0], [1, -1, -1]]), # 3 units rel (.614)
    (1/sqrt(3), 'pipi', [[1, 0, 0], [0, -1, -1]]), # 3 units rel (.692)
    (1/sqrt(3), 'pipi', [[0, -1, 0], [1, 0, -1]]),
    (1/sqrt(3), 'pipi', [[0, 0, -1], [1, -1, 0]]),
    (1, 'sigma', [1, -1, -1]), # sigma
]

A1y_mom_11_1 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, 1, -1]]),
    (0.5773502691896258, 'pipi', [[0, 1, 0], [-1, 0, -1]]),
    (0.5773502691896258, 'pipi', [[-1, 0, 0], [0, 1, -1]]),
    (0.5773502691896258, 'pipi', [[0, 0, -1], [-1, 1, 0]]),
    (1, 'sigma', [-1, 1, -1]),
]

A1z_mom_1_11 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, -1, 1]]),
    (0.5773502691896258, 'pipi', [[0, 0, 1], [-1, -1, 0]]),
    (0.5773502691896258, 'pipi', [[0, -1, 0], [-1, 0, 1]]),
    (0.5773502691896258, 'pipi', [[-1, 0, 0], [0, -1, 1]]),
    (1, 'sigma', [-1, -1, 1])
]


# all neg
A1_mom_1_1_1 = [
    (1, 'S_pipi', [[0, 0, 0], [-1, -1, -1]]), # 3 units rel (.614)
    (1/sqrt(3), 'pipi', [[-1, 0, 0], [0, -1, -1]]), # 3 units rel (.692)
    (1/sqrt(3), 'pipi', [[0, -1, 0], [-1, 0, -1]]),
    (1/sqrt(3), 'pipi', [[0, 0, -1], [-1, -1, 0]]),
    (1, 'sigma', [-1, -1, -1]), # sigma ?
]


"""
    (1/sqrt(6), 'UUpipi', [[2, 1, 1], [-1, 0, 0]]), # 5 units rel (.953)
#    (1/sqrt(6), 'UUpipi', [[-1, 0, 0], [2, 1, 1]]),
    (1/sqrt(6), 'UUpipi', [[1, 2, 1], [0, -1, 0]]),
#    (1/sqrt(6), 'UUpipi', [[0, -1, 0], [1, 2, 1]]),
    (1/sqrt(6), 'UUpipi', [[1, 1, 2], [0, 0, -1]]),
#    (1/sqrt(6), 'UUpipi', [[0, 0, -1], [1, 1, 2]])
    (1/sqrt(6), 'U3pipi', [[-1, 1, 1], [2, 0, 0]]), # 5 units rel (1.02)
    (1/sqrt(6), 'U3pipi', [[2, 0, 0], [-1, 1, 1]]),
    (1/sqrt(6), 'U3pipi', [[0, 2, 0], [1, -1, 1]]),
    (1/sqrt(6), 'U3pipi', [[1, -1, 1], [0, 2, 0]]),
    (1/sqrt(6), 'U3pipi', [[0, 0, 2], [1, 1, -1]]),
    (1/sqrt(6), 'U3pipi', [[1, 1, -1], [0, 0, 2]]),
]
    (1/sqrt(12), 'U2pipi', [[0, 2, 1], [1, -1, 0]]), # 5 units rel (1)
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
    ]

]
"""

# unused/unnecessary/redundant gevp lists below

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

def write_mom_comb():
    """Write the momentum combination vml
    (list of allowed two particle momenta)"""
    twoplist = {}
    for irrep in OPLIST:
        for _, _, mom_comb in OPLIST[irrep]:
            if len(mom_comb) == 2: # skip the sigma
                twoplist[str(mom_comb)] = mom_comb
    begin = 'Array moms[2] = {\nArray moms[0] = {\nArray p[3] = {\n'
    middle = '}\n}\nArray moms[1] = {\nArray p[3] = {\n'
    end = '}\n'*4
    with open('mom_comb.vml', 'w') as fn1:
        fn1.write('class allowedCombP mom_comb = {\n')
        fn1.write('Array momcomb['+str(len(twoplist))+'] = {\n')
        for i, comb in enumerate(sorted(twoplist)):
            fn1.write('Array momcomb['+str(i)+'] = {\n')
            fn1.write(begin)
            fn1.write(ptonewlinelist(twoplist[comb][0]))
            fn1.write(middle)
            fn1.write(ptonewlinelist(twoplist[comb][1]))
            fn1.write(end)
        fn1.write('}\n}')

def ptonewlinelist(mom):
   """make mom into a new line separated momentum string
   """ 
   return 'int p[0] = '+str(mom[0])+'\nint p[1] = '+\
       str(mom[1])+'\nint p[2] = '+str(mom[2])+'\n'

def free_energies(irrep, pionmass, lbox):
    """return a list of free energies."""
    retlist = []
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    opprev = ''
    for _, opa, mom in OPLIST[irrep]:
        if opa == opprev:
            continue
        if len(mom) != 2:
            continue
        opprev = opa
        energy = 0
        for pin in mom:
            # print(pionmass, pin, lbox)
            energy += np.sqrt(pionmass**2+(2*np.pi/lbox)**2*rf.norm2(pin))
        retlist.append(energy)
    return sorted(retlist)

def get_comp_str(irrep):
    """Get center of mass momentum of an irrep, return as a string for latfit"""
    retlist = []
    if irrep in AVG_ROWS:
        for irr in AVG_ROWS[irrep]:
            irrep = irr
            break
    opprev = ''
    momtotal = np.array([0, 0, 0])
    for _, _, mom in OPLIST[irrep]:
        if len(mom) != 2:
            continue
        for pin in mom:
            momtotal += np.array(pin)
        break
    return 'mom'+rf.ptostr(momtotal)


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

def generateChecksums(isospin):
    """Generate a sum of expected diagrams for each operator"""
    isospin = int(isospin)
    checks = {}
    for oplist in OPLIST:
        newl = len(OPLIST[oplist])
        for coeff, opa, mom in OPLIST[oplist]:
            if 'sigma' in opa and isospin != 0:
                newl -= 1
            elif 'rho' in opa and isospin != 1:
                newl -= 1
        checks[oplist] = newl**2
    return checks

# specify polarization info here
OPLIST = {'A_1PLUS_mom000': A_1PLUS_mom000,
          'A1x_mom100': A1x_mom100,
          'A1x_mom_100': A1x_mom_100,
          'A1y_mom010': A1y_mom010,
          'A1y_mom0_10': A1y_mom0_10,
          'A1z_mom001': A1z_mom001,
          'A1z_mom00_1': A1z_mom00_1,
          'A1x_mom011': A1x_mom011,
          'A1x_mom0_11': A1x_mom0_11,
          'A1x_mom01_1': A1x_mom01_1,
          'A1x_mom0_1_1': A1x_mom0_1_1,
          'A1y_mom101': A1y_mom101,
          'A1y_mom_101': A1y_mom_101,
          'A1y_mom10_1': A1y_mom10_1,
          'A1y_mom_10_1': A1y_mom_10_1,
          'A1z_mom110': A1z_mom110,
          'A1z_mom_110': A1z_mom_110,
          'A1z_mom1_10': A1z_mom1_10,
          'A1z_mom_1_10': A1z_mom_1_10,
          'A1_mom111': A1_mom111,
          'A1x_mom_111': A1x_mom_111,
          'A1y_mom1_11': A1y_mom1_11,
          'A1z_mom11_1': A1z_mom11_1,
          'A1x_mom1_1_1': A1x_mom1_1_1,
          'A1y_mom_11_1': A1y_mom_11_1,
          'A1z_mom_1_11': A1z_mom_1_11,
          'A1_mom_1_1_1': A1_mom_1_1_1,
          # 'T_1_1MINUS?pol=1': T_1_1MINUS,
          # 'T_1_3MINUS?pol=2': T_1_3MINUS,
          # 'T_1_2MINUS?pol=3': T_1_2MINUS,
          # 'A0_mom000': A0_mom000,
          # 'A2': A2,
          # 'A1x': A1x,
          # 'A1mx': A1mx,
          # 'A1y': A1y,
          # 'A1my': A1my,
          # 'A1z': A1z,
          #'A1mz': A1mz,
}

AVG_ROWS = {
    'T_1_MINUS': ('T_1_1MINUS', 'T_1_2MINUS', 'T_1_3MINUS'),
    'A1_mom1': ('A1x_mom100',
                'A1x_mom_100',
                'A1y_mom010',
                'A1y_mom0_10',
                'A1z_mom001',
                'A1z_mom00_1',
    ),
    'A1_mom11': ('A1x_mom011',
                 'A1x_mom0_11',
                 'A1x_mom01_1',
                 'A1x_mom0_1_1',
                 'A1y_mom101',
                 'A1y_mom_101',
                 'A1y_mom10_1',
                 'A1y_mom_10_1',
                 'A1z_mom110'
                 'A1z_mom_110'
                 'A1z_mom1_10'
                 'A1z_mom_1_10'
    ),
    'A1_avg_mom111': ('A1_mom111',
                      'A1_mom_1_1_1',
                      'A1x_mom_111',
                      'A1y_mom1_11',
                      'A1z_mom11_1',
                      'A1x_mom1_1_1',
                      'A1y_mom_11_1',
                      'A1z_mom_1_11'),
    # 'A1': ('A1x', 'A1mx', 'A1y', 'A1my', 'A1z', 'A1mz')
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
        momchk = rf.mom(opa) if 'mom' in opa else None
        for chkidx, src in enumerate(OPLIST[opa]):
            for chkidx2, snk in enumerate(OPLIST[opa]):
                if src[1] == snk[1]:
                    dup_flag = True
                    for pcheck, pcheck2 in zip(src[2], snk[2]):
                        if isinstance(pcheck, int):
                            dup_flag = pcheck == pcheck2
                        elif rf.ptostr(pcheck) != rf.ptostr(pcheck2):
                            dup_flag = False
                    if dup_flag:
                        assert chkidx == chkidx2, "Duplicate operator found in "+str(opa)+" "+str(src)+" "+str(snk)
                assert cons_mom(src, snk, momchk), "operator does not conserve momentum "+str(opa)
                part_str = partstr(src[1], snk[1])
                coeff = src[0]*snk[0]
                p_str = momstr(src[2], snk[2])
                dur = part_str+"_"+p_str
                dur = re.sub('S_', '', dur)
                dur = re.sub('UU', '', dur)
                for i in range(10):
                    dur = re.sub('U'+str(i), '', dur)
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
    psrc = wd.momtotal(src[2])
    psnk = wd.momtotal(snk[2])
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
        #print(l[i])

    print(l['pipisigma_A_1PLUS_mom000'])
    print(l['rhorho_T_1_1MINUS?pol=1'])
    generateOperatorMomenta()


if __name__ == "__main__":
    main()
