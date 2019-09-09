"""All the irreps for the I=1 moving frames"""
## I=1 moving frames

## p1 == A1^+ \circleplus B
import sys
import numpy as np
from math import sqrt
import read_file as rf
from latfit.utilities import exactmean as em

current_module = sys.modules[__name__]
cmod = current_module


A_1PLUS_mom_100 = [
    (-1.0, 'S_pipi', [[-1, 0, 0], [0, 0, 0]]),
]

A_1PLUS_mom010 = [
    (-1.0, 'S_pipi', [[0, 1, 0], [0, 0, 0]]),
]

A_1PLUS_mom00_1 = [
    (-1.0, 'S_pipi', [[0, 0, -1], [0, 0, 0]]),
]

A_1PLUS_mom100 = [
    (-1.0, 'S_pipi', [[1, 0, 0], [0, 0, 0]]),
]

A_1PLUS_mom0_10 = [
    (-1.0, 'S_pipi', [[0, -1, 0], [0, 0, 0]]),
]

A_1PLUS_mom001 = [
    (-1.0, 'S_pipi', [[0, 0, 1], [0, 0, 0]]),
]

A_1PLUS_mom_100 += [
    (-0.5, 'pipi', [[-1, -1, 0], [0, 1, 0]]),
    (-0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]),
    (-0.5, 'pipi', [[-1, 1, 0], [0, -1, 0]]),
]

A_1PLUS_mom010 += [
    (-0.5, 'pipi', [[-1, 1, 0], [1, 0, 0]]),
    (-0.5, 'pipi', [[0, 1, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[0, 1, 1], [0, 0, -1]]),
    (-0.5, 'pipi', [[1, 1, 0], [-1, 0, 0]]),
]

A_1PLUS_mom00_1 += [
    (-0.5, 'pipi', [[-1, 0, -1], [1, 0, 0]]),
    (-0.5, 'pipi', [[0, -1, -1], [0, 1, 0]]),
    (-0.5, 'pipi', [[0, 1, -1], [0, -1, 0]]),
    (-0.5, 'pipi', [[1, 0, -1], [-1, 0, 0]]),
]

A_1PLUS_mom100 += [
    (-0.5, 'pipi', [[1, -1, 0], [0, 1, 0]]),
    (-0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]),
    (-0.5, 'pipi', [[1, 1, 0], [0, -1, 0]]),
]

A_1PLUS_mom0_10 += [
    (-0.5, 'pipi', [[-1, -1, 0], [1, 0, 0]]),
    (-0.5, 'pipi', [[0, -1, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[0, -1, 1], [0, 0, -1]]),
    (-0.5, 'pipi', [[1, -1, 0], [-1, 0, 0]]),
]

A_1PLUS_mom001 += [
    (-0.5, 'pipi', [[-1, 0, 1], [1, 0, 0]]),
    (-0.5, 'pipi', [[0, -1, 1], [0, 1, 0]]),
    (-0.5, 'pipi', [[0, 1, 1], [0, -1, 0]]),
    (-0.5, 'pipi', [[1, 0, 1], [-1, 0, 0]]),
]

A_1PLUS_mom_100 += [
    (-0.5, 'UUpipi', [[-1, -1, -1], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[-1, -1, 1], [0, 1, -1]]),
    (-0.5, 'UUpipi', [[-1, 1, -1], [0, -1, 1]]),
    (-0.5, 'UUpipi', [[-1, 1, 1], [0, -1, -1]]),
]

A_1PLUS_mom010 += [
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, 0, 1]]),
    (-0.5, 'UUpipi', [[-1, 1, 1], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, 1, -1], [-1, 0, 1]]),
    (-0.5, 'UUpipi', [[1, 1, 1], [-1, 0, -1]]),
]

A_1PLUS_mom00_1 += [
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, -1, 0]]),
    (-0.5, 'UUpipi', [[1, -1, -1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[1, 1, -1], [-1, -1, 0]]),
]

A_1PLUS_mom100 += [
    (-0.5, 'UUpipi', [[1, -1, -1], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[1, -1, 1], [0, 1, -1]]),
    (-0.5, 'UUpipi', [[1, 1, -1], [0, -1, 1]]),
    (-0.5, 'UUpipi', [[1, 1, 1], [0, -1, -1]]),
]

A_1PLUS_mom0_10 += [
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 0, 1]]),
    (-0.5, 'UUpipi', [[-1, -1, 1], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, -1, -1], [-1, 0, 1]]),
    (-0.5, 'UUpipi', [[1, -1, 1], [-1, 0, -1]]),
]

A_1PLUS_mom001 += [
    (-0.5, 'UUpipi', [[-1, -1, 1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[-1, 1, 1], [1, -1, 0]]),
    (-0.5, 'UUpipi', [[1, -1, 1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[1, 1, 1], [-1, -1, 0]]),
]




B_0_mom00_1 = [
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, -1], [-1, 1, 0]]),
]

B_0_mom_100 = [
    (-0.5, 'UUpipi', [[-1, -1, -1], [0, 1, 1]]),
    (0.5, 'UUpipi', [[-1, 1, 1], [0, -1, -1]]),
]

B_1_mom_100 = [
    (0.5, 'UUpipi', [[-1, -1, 1], [0, 1, -1]]),
    (-0.5, 'UUpipi', [[-1, 1, -1], [0, -1, 1]]),
]

B_1_mom100 = [
    (0.5, 'UUpipi', [[1, -1, -1], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[1, 1, 1], [0, -1, -1]]),
]

B_0_mom0_10 = [
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[1, -1, 1], [-1, 0, -1]]),
]

B_0_mom010 = [
    (-0.5, 'UUpipi', [[-1, 1, 1], [1, 0, -1]]),
    (0.5, 'UUpipi', [[1, 1, -1], [-1, 0, 1]]),
]

B_1_mom010 = [
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[1, 1, 1], [-1, 0, -1]]),
]

B_0_mom001 = [
    (-0.5, 'UUpipi', [[-1, -1, 1], [1, 1, 0]]),
    (0.5, 'UUpipi', [[1, 1, 1], [-1, -1, 0]]),
]

B_1_mom001 = [
    (-0.5, 'UUpipi', [[-1, 1, 1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 1], [-1, 1, 0]]),
]

B_1_mom0_10 = [
    (-0.5, 'UUpipi', [[-1, -1, 1], [1, 0, -1]]),
    (0.5, 'UUpipi', [[1, -1, -1], [-1, 0, 1]]),
]

B_0_mom100 = [
    (-0.5, 'UUpipi', [[1, -1, 1], [0, 1, -1]]),
    (0.5, 'UUpipi', [[1, 1, -1], [0, -1, 1]]),
]

B_1_mom00_1 = [
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 1, 0]]),
    (0.5, 'UUpipi', [[1, 1, -1], [-1, -1, 0]]),
]


B_0_mom00_1 += [
    (-0.5, 'pipi', [[-1, 0, -1], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 0, -1], [-1, 0, 0]]),
]

B_0_mom_100 += [
    (-0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]),
    (0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]),
]

B_1_mom_100 += [
    (-0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]),
    (0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]),
]

B_1_mom100 += [
    (0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]),
]

B_0_mom0_10 += [
    (-0.5, 'pipi', [[-1, -1, 0], [1, 0, 0]]),
    (0.5, 'pipi', [[1, -1, 0], [-1, 0, 0]]),
]

B_0_mom010 += [
    (-0.5, 'pipi', [[-1, 1, 0], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 1, 0], [-1, 0, 0]]),
]

B_1_mom010 += [
    (-0.5, 'pipi', [[-1, 1, 0], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 1, 0], [-1, 0, 0]]),
]

B_0_mom001 += [
    (-0.5, 'pipi', [[-1, 0, 1], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 0, 1], [-1, 0, 0]]),
]

B_1_mom001 += [
    (-0.5, 'pipi', [[-1, 0, 1], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 0, 1], [-1, 0, 0]]),
]

B_1_mom0_10 += [
    (-0.5, 'pipi', [[-1, -1, 0], [1, 0, 0]]),
    (0.5, 'pipi', [[1, -1, 0], [-1, 0, 0]]),
]

B_0_mom100 += [
    (0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]),
    (-0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]),
]

B_1_mom00_1 += [
    (-0.5, 'pipi', [[-1, 0, -1], [1, 0, 0]]),
    (0.5, 'pipi', [[1, 0, -1], [-1, 0, 0]]),
]


## p11 == A1^+ (12) \circleplus A2^ + \circleplus A2^-

A_1PLUS_mom0_1_1 = [
    (-1.0, 'S_pipi', [[0, -1, -1], [0, 0, 0]]),
]

A_1PLUS_mom01_1 = [
    (-1.0, 'S_pipi', [[0, 1, -1], [0, 0, 0]]),
]

A_1PLUS_mom_101 = [
    (-1.0, 'S_pipi', [[-1, 0, 1], [0, 0, 0]]),
]

A_1PLUS_mom011 = [
    (-1.0, 'S_pipi', [[0, 1, 1], [0, 0, 0]]),
]

A_1PLUS_mom10_1 = [
    (-1.0, 'S_pipi', [[1, 0, -1], [0, 0, 0]]),
]

A_1PLUS_mom101 = [
    (-1.0, 'S_pipi', [[1, 0, 1], [0, 0, 0]]),
]

A_1PLUS_mom1_10 = [
    (-1.0, 'S_pipi', [[1, -1, 0], [0, 0, 0]]),
]

A_1PLUS_mom_10_1 = [
    (-1.0, 'S_pipi', [[-1, 0, -1], [0, 0, 0]]),
]

A_1PLUS_mom110 = [
    (-1.0, 'S_pipi', [[1, 1, 0], [0, 0, 0]]),
]

A_1PLUS_mom0_11 = [
    (-1.0, 'S_pipi', [[0, -1, 1], [0, 0, 0]]),
]

A_1PLUS_mom_1_10 = [
    (-1.0, 'S_pipi', [[-1, -1, 0], [0, 0, 0]]),
]

A_1PLUS_mom_110 = [
    (-1.0, 'S_pipi', [[-1, 1, 0], [0, 0, 0]]),
]


A_1PLUS_mom_1_10 += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [-1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, -1, 0]]),
]

A_1PLUS_mom0_1_1 += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, -1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, -1]]),
]

A_1PLUS_mom1_10 += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, -1, 0]]),
]

A_1PLUS_mom10_1 += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, -1]]),
]

A_1PLUS_mom0_11 += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, -1, 0]]),
]

A_1PLUS_mom_110 += [
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [-1, 0, 0]]),
]

A_1PLUS_mom01_1 += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, -1]]),
]

A_1PLUS_mom011 += [
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]),
]

A_1PLUS_mom_101 += [
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [-1, 0, 0]]),
]

A_1PLUS_mom110 += [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [1, 0, 0]]),
]

A_1PLUS_mom_10_1 += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [-1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, -1]]),
]

A_1PLUS_mom101 += [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [1, 0, 0]]),
]

A_2PLUS_mom_1_10 = [
    (1/sqrt(2), 'pipi', [[0, -1, 0], [-1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, -1, 0]]),
]

A_2PLUS_mom10_1 = [
    (1/sqrt(2), 'pipi', [[0, 0, -1], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, -1]]),
]

A_2PLUS_mom_101 = [
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [-1, 0, 0]]),
]

A_2PLUS_mom110 = [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 1, 0]]),
    (1/sqrt(2), 'pipi', [[0, 1, 0], [1, 0, 0]]),
]

A_2PLUS_mom0_11 = [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, -1, 0]]),
]

A_2PLUS_mom01_1 = [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 1, 0]]),
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, -1]]),
]

A_2PLUS_mom0_1_1 = [
    (1/sqrt(2), 'pipi', [[0, 0, -1], [0, -1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, -1]]),
]

A_2PLUS_mom_110 = [
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [-1, 0, 0]]),
]

A_2PLUS_mom101 = [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[0, 0, 1], [1, 0, 0]]),
]

A_2PLUS_mom1_10 = [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, -1, 0]]),
]

A_2PLUS_mom_10_1 = [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [-1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, -1]]),
]

A_2PLUS_mom011 = [
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]),
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]),
]

A_1PLUS_mom0_1_1 += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [1, 0, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [-1, 0, 0]]),
]

A_1PLUS_mom01_1 += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [1, 0, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [-1, 0, 0]]),
]

A_1PLUS_mom_101 += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [0, 1, 0]]),
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [0, -1, 0]]),
]

A_1PLUS_mom011 += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [1, 0, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [-1, 0, 0]]),
]

A_1PLUS_mom10_1 += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [0, 1, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [0, -1, 0]]),
]

A_1PLUS_mom101 += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [0, 1, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [0, -1, 0]]),
]

A_1PLUS_mom1_10 += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [0, 0, 1]]),
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [0, 0, -1]]),
]

A_1PLUS_mom_10_1 += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [0, 1, 0]]),
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [0, -1, 0]]),
]

A_1PLUS_mom110 += [
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [0, 0, 1]]),
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [0, 0, -1]]),
]

A_1PLUS_mom0_11 += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [1, 0, 0]]),
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [-1, 0, 0]]),
]

A_1PLUS_mom_1_10 += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [0, 0, 1]]),
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [0, 0, -1]]),
]

A_1PLUS_mom_110 += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [0, 0, 1]]),
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [0, 0, -1]]),
]

A_2MINUS_mom_110 = [
    (-1/sqrt(2), 'pipi', [[-1, 1, -1], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[-1, 1, 1], [0, 0, -1]]),
]

A_2MINUS_mom0_1_1 = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[1, -1, -1], [-1, 0, 0]]),
]

A_2MINUS_mom_1_10 = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [0, 0, -1]]),
]

A_2MINUS_mom110 = [
    (-1/sqrt(2), 'pipi', [[1, 1, -1], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[1, 1, 1], [0, 0, -1]]),
]

A_2MINUS_mom10_1 = [
    (-1/sqrt(2), 'pipi', [[1, -1, -1], [0, 1, 0]]),
    (1/sqrt(2), 'pipi', [[1, 1, -1], [0, -1, 0]]),
]

A_2MINUS_mom1_10 = [
    (-1/sqrt(2), 'pipi', [[1, -1, -1], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[1, -1, 1], [0, 0, -1]]),
]

A_2MINUS_mom0_11 = [
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, -1, 1], [-1, 0, 0]]),
]

A_2MINUS_mom101 = [
    (1/sqrt(2), 'pipi', [[1, -1, 1], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 1, 1], [0, -1, 0]]),
]

A_2MINUS_mom01_1 = [
    (-1/sqrt(2), 'pipi', [[-1, 1, -1], [1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[1, 1, -1], [-1, 0, 0]]),
]

A_2MINUS_mom_10_1 = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [0, 1, 0]]),
    (1/sqrt(2), 'pipi', [[-1, 1, -1], [0, -1, 0]]),
]

A_2MINUS_mom_101 = [
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[-1, 1, 1], [0, -1, 0]]),
]

A_2MINUS_mom011 = [
    (1/sqrt(2), 'pipi', [[-1, 1, 1], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 1, 1], [-1, 0, 0]]),
]




A_1PLUS_mom_101 += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [0, 1, 1]]),
    (-0.5, 'U2pipi', [[-1, 1, 0], [0, -1, 1]]),
    (-0.5, 'U2pipi', [[0, -1, 1], [-1, 1, 0]]),
    (-0.5, 'U2pipi', [[0, 1, 1], [-1, -1, 0]]),
]

A_1PLUS_mom1_10 += [
    (-0.5, 'U2pipi', [[0, -1, -1], [1, 0, 1]]),
    (-0.5, 'U2pipi', [[0, -1, 1], [1, 0, -1]]),
    (-0.5, 'U2pipi', [[1, 0, -1], [0, -1, 1]]),
    (-0.5, 'U2pipi', [[1, 0, 1], [0, -1, -1]]),
]

A_1PLUS_mom_10_1 += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [0, 1, -1]]),
    (-0.5, 'U2pipi', [[-1, 1, 0], [0, -1, -1]]),
    (-0.5, 'U2pipi', [[0, -1, -1], [-1, 1, 0]]),
    (-0.5, 'U2pipi', [[0, 1, -1], [-1, -1, 0]]),
]

A_1PLUS_mom_1_10 += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [0, -1, 1]]),
    (-0.5, 'U2pipi', [[-1, 0, 1], [0, -1, -1]]),
    (-0.5, 'U2pipi', [[0, -1, -1], [-1, 0, 1]]),
    (-0.5, 'U2pipi', [[0, -1, 1], [-1, 0, -1]]),
]

A_1PLUS_mom01_1 += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [1, 1, 0]]),
    (-0.5, 'U2pipi', [[-1, 1, 0], [1, 0, -1]]),
    (-0.5, 'U2pipi', [[1, 0, -1], [-1, 1, 0]]),
    (-0.5, 'U2pipi', [[1, 1, 0], [-1, 0, -1]]),
]

A_1PLUS_mom110 += [
    (-0.5, 'U2pipi', [[0, 1, -1], [1, 0, 1]]),
    (-0.5, 'U2pipi', [[0, 1, 1], [1, 0, -1]]),
    (-0.5, 'U2pipi', [[1, 0, -1], [0, 1, 1]]),
    (-0.5, 'U2pipi', [[1, 0, 1], [0, 1, -1]]),
]

A_1PLUS_mom101 += [
    (-0.5, 'U2pipi', [[0, -1, 1], [1, 1, 0]]),
    (-0.5, 'U2pipi', [[0, 1, 1], [1, -1, 0]]),
    (-0.5, 'U2pipi', [[1, -1, 0], [0, 1, 1]]),
    (-0.5, 'U2pipi', [[1, 1, 0], [0, -1, 1]]),
]

A_1PLUS_mom_110 += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [0, 1, 1]]),
    (-0.5, 'U2pipi', [[-1, 0, 1], [0, 1, -1]]),
    (-0.5, 'U2pipi', [[0, 1, -1], [-1, 0, 1]]),
    (-0.5, 'U2pipi', [[0, 1, 1], [-1, 0, -1]]),
]

A_1PLUS_mom0_11 += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [1, 0, 1]]),
    (-0.5, 'U2pipi', [[-1, 0, 1], [1, -1, 0]]),
    (-0.5, 'U2pipi', [[1, -1, 0], [-1, 0, 1]]),
    (-0.5, 'U2pipi', [[1, 0, 1], [-1, -1, 0]]),
]

A_1PLUS_mom011 += [
    (-0.5, 'U2pipi', [[-1, 0, 1], [1, 1, 0]]),
    (-0.5, 'U2pipi', [[-1, 1, 0], [1, 0, 1]]),
    (-0.5, 'U2pipi', [[1, 0, 1], [-1, 1, 0]]),
    (-0.5, 'U2pipi', [[1, 1, 0], [-1, 0, 1]]),
]

A_1PLUS_mom10_1 += [
    (-0.5, 'U2pipi', [[0, -1, -1], [1, 1, 0]]),
    (-0.5, 'U2pipi', [[0, 1, -1], [1, -1, 0]]),
    (-0.5, 'U2pipi', [[1, -1, 0], [0, 1, -1]]),
    (-0.5, 'U2pipi', [[1, 1, 0], [0, -1, -1]]),
]

A_1PLUS_mom0_1_1 += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [1, 0, -1]]),
    (-0.5, 'U2pipi', [[-1, 0, -1], [1, -1, 0]]),
    (-0.5, 'U2pipi', [[1, -1, 0], [-1, 0, -1]]),
    (-0.5, 'U2pipi', [[1, 0, -1], [-1, -1, 0]]),
]

A_2MINUS_mom_10_1 += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, -1]]),
    (0.5, 'UUpipi', [[-1, 1, 0], [0, -1, -1]]),
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 1, 0]]),
    (0.5, 'UUpipi', [[0, 1, -1], [-1, -1, 0]]),
]

A_2MINUS_mom01_1 += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, -1]]),
    (0.5, 'UUpipi', [[1, 0, -1], [-1, 1, 0]]),
    (0.5, 'UUpipi', [[1, 1, 0], [-1, 0, -1]]),
]

A_2MINUS_mom110 += [
    (-0.5, 'UUpipi', [[0, 1, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[0, 1, 1], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, 0, -1], [0, 1, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [0, 1, -1]]),
]

A_2MINUS_mom0_1_1 += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, -1]]),
    (0.5, 'UUpipi', [[1, 0, -1], [-1, -1, 0]]),
]

A_2MINUS_mom011 += [
    (0.5, 'UUpipi', [[-1, 0, 1], [1, 1, 0]]),
    (0.5, 'UUpipi', [[-1, 1, 0], [1, 0, 1]]),
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, 1]]),
]

A_2MINUS_mom_1_10 += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, -1, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [0, -1, -1]]),
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 0, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 0, -1]]),
]

A_2MINUS_mom101 += [
    (0.5, 'UUpipi', [[0, -1, 1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[0, 1, 1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[1, 1, 0], [0, -1, 1]]),
]

A_2MINUS_mom0_11 += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [1, -1, 0]]),
    (-0.5, 'UUpipi', [[1, -1, 0], [-1, 0, 1]]),
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, -1, 0]]),
]

A_2MINUS_mom1_10 += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, 0, -1], [0, -1, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [0, -1, -1]]),
]

A_2MINUS_mom_110 += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, 1, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [0, 1, -1]]),
    (-0.5, 'UUpipi', [[0, 1, -1], [-1, 0, 1]]),
    (0.5, 'UUpipi', [[0, 1, 1], [-1, 0, -1]]),
]

A_2MINUS_mom_101 += [
    (0.5, 'UUpipi', [[-1, -1, 0], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[0, 1, 1], [-1, -1, 0]]),
]

A_2MINUS_mom10_1 += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 1, 0]]),
    (0.5, 'UUpipi', [[0, 1, -1], [1, -1, 0]]),
    (-0.5, 'UUpipi', [[1, -1, 0], [0, 1, -1]]),
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, -1]]),
]


A_2PLUS_mom0_11 += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, 1]]),
    (-0.5, 'UUpipi', [[-1, 0, 1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, 1]]),
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, -1, 0]]),
]

A_2PLUS_mom1_10 += [
    (0.5, 'UUpipi', [[0, -1, -1], [1, 0, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, 0, -1], [0, -1, 1]]),
    (-0.5, 'UUpipi', [[1, 0, 1], [0, -1, -1]]),
]

A_2PLUS_mom011 += [
    (0.5, 'UUpipi', [[-1, 0, 1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, 1]]),
]

A_2PLUS_mom101 += [
    (-0.5, 'UUpipi', [[0, -1, 1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[0, 1, 1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, 1]]),
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, 1]]),
]

A_2PLUS_mom_1_10 += [
    (0.5, 'UUpipi', [[-1, 0, -1], [0, -1, 1]]),
    (0.5, 'UUpipi', [[-1, 0, 1], [0, -1, -1]]),
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 0, 1]]),
    (-0.5, 'UUpipi', [[0, -1, 1], [-1, 0, -1]]),
]

A_2PLUS_mom110 += [
    (-0.5, 'UUpipi', [[0, 1, -1], [1, 0, 1]]),
    (-0.5, 'UUpipi', [[0, 1, 1], [1, 0, -1]]),
    (0.5, 'UUpipi', [[1, 0, -1], [0, 1, 1]]),
    (0.5, 'UUpipi', [[1, 0, 1], [0, 1, -1]]),
]

A_2PLUS_mom0_1_1 += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, -1]]),
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, -1]]),
    (-0.5, 'UUpipi', [[1, 0, -1], [-1, -1, 0]]),
]

A_2PLUS_mom10_1 += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[0, 1, -1], [1, -1, 0]]),
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, -1]]),
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, -1]]),
]

A_2PLUS_mom_101 += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, 1]]),
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 1, 0]]),
    (0.5, 'UUpipi', [[0, 1, 1], [-1, -1, 0]]),
]

A_2PLUS_mom01_1 += [
    (0.5, 'UUpipi', [[-1, 0, -1], [1, 1, 0]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, -1]]),
    (0.5, 'UUpipi', [[1, 0, -1], [-1, 1, 0]]),
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, -1]]),
]

A_2PLUS_mom_10_1 += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, -1]]),
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, -1]]),
    (0.5, 'UUpipi', [[0, -1, -1], [-1, 1, 0]]),
    (0.5, 'UUpipi', [[0, 1, -1], [-1, -1, 0]]),
]

A_2PLUS_mom_110 += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, 1, 1]]),
    (-0.5, 'UUpipi', [[-1, 0, 1], [0, 1, -1]]),
    (0.5, 'UUpipi', [[0, 1, -1], [-1, 0, 1]]),
    (0.5, 'UUpipi', [[0, 1, 1], [-1, 0, -1]]),
]


## p111 == A1^+ \circleplus B

A_1PLUS_mom1_1_1 = [
    (-1/sqrt(3), 'pipi', [[0, -1, -1], [1, 0, 0]]),
    (-1/sqrt(3), 'pipi', [[1, -1, 0], [0, 0, -1]]),
    (-1/sqrt(3), 'pipi', [[1, 0, -1], [0, -1, 0]]),
]

A_1PLUS_mom_1_11 = [
    (-1/sqrt(3), 'pipi', [[-1, -1, 0], [0, 0, 1]]),
    (-1/sqrt(3), 'pipi', [[-1, 0, 1], [0, -1, 0]]),
    (-1/sqrt(3), 'pipi', [[0, -1, 1], [-1, 0, 0]]),
]

A_1PLUS_mom11_1 = [
    (-1/sqrt(3), 'pipi', [[0, 1, -1], [1, 0, 0]]),
    (-1/sqrt(3), 'pipi', [[1, 0, -1], [0, 1, 0]]),
    (-1/sqrt(3), 'pipi', [[1, 1, 0], [0, 0, -1]]),
]

A_1PLUS_mom_111 = [
    (-1/sqrt(3), 'pipi', [[-1, 0, 1], [0, 1, 0]]),
    (-1/sqrt(3), 'pipi', [[-1, 1, 0], [0, 0, 1]]),
    (-1/sqrt(3), 'pipi', [[0, 1, 1], [-1, 0, 0]]),
]

A_1PLUS_mom_1_1_1 = [
    (-1/sqrt(3), 'pipi', [[-1, -1, 0], [0, 0, -1]]),
    (-1/sqrt(3), 'pipi', [[-1, 0, -1], [0, -1, 0]]),
    (-1/sqrt(3), 'pipi', [[0, -1, -1], [-1, 0, 0]]),
]

A_1PLUS_mom_11_1 = [
    (-1/sqrt(3), 'pipi', [[-1, 0, -1], [0, 1, 0]]),
    (-1/sqrt(3), 'pipi', [[-1, 1, 0], [0, 0, -1]]),
    (-1/sqrt(3), 'pipi', [[0, 1, -1], [-1, 0, 0]]),
]

A_1PLUS_mom111 = [
    (-1/sqrt(3), 'pipi', [[0, 1, 1], [1, 0, 0]]),
    (-1/sqrt(3), 'pipi', [[1, 0, 1], [0, 1, 0]]),
    (-1/sqrt(3), 'pipi', [[1, 1, 0], [0, 0, 1]]),
]

A_1PLUS_mom1_11 = [
    (-1/sqrt(3), 'pipi', [[0, -1, 1], [1, 0, 0]]),
    (-1/sqrt(3), 'pipi', [[1, -1, 0], [0, 0, 1]]),
    (-1/sqrt(3), 'pipi', [[1, 0, 1], [0, -1, 0]]),
]


A_1PLUS_mom111 += [
    (-1.0, 'S_pipi', [[1, 1, 1], [0, 0, 0]]),
]

A_1PLUS_mom_1_11 += [
    (-1.0, 'S_pipi', [[-1, -1, 1], [0, 0, 0]]),
]

A_1PLUS_mom11_1 += [
    (-1.0, 'S_pipi', [[1, 1, -1], [0, 0, 0]]),
]

A_1PLUS_mom_111 += [
    (-1.0, 'S_pipi', [[-1, 1, 1], [0, 0, 0]]),
]

A_1PLUS_mom_11_1 += [
    (-1.0, 'S_pipi', [[-1, 1, -1], [0, 0, 0]]),
]

A_1PLUS_mom_1_1_1 += [
    (-1.0, 'S_pipi', [[-1, -1, -1], [0, 0, 0]]),
]

A_1PLUS_mom1_11 += [
    (-1.0, 'S_pipi', [[1, -1, 1], [0, 0, 0]]),
]

A_1PLUS_mom1_1_1 += [
    (-1.0, 'S_pipi', [[1, -1, -1], [0, 0, 0]]),
]



# next p111 irrep, B

B_1_mom11_1 = [
    (1/sqrt(2), 'pipi', [[1, 0, -1], [0, 1, 0]]),
    (-1/sqrt(2), 'pipi', [[1, 1, 0], [0, 0, -1]]),
]

B_0_mom1_11 = [
    (-1/sqrt(6), 'pipi', [[0, -1, 1], [1, 0, 0]]),
    (-1/sqrt(6), 'pipi', [[1, -1, 0], [0, 0, 1]]),
    (sqrt(2/3), 'pipi', [[1, 0, 1], [0, -1, 0]]),
]

B_0_mom_1_1_1 = [
    (sqrt(2/3), 'pipi', [[-1, -1, 0], [0, 0, -1]]),
    (-1/sqrt(6), 'pipi', [[-1, 0, -1], [0, -1, 0]]),
    (-1/sqrt(6), 'pipi', [[0, -1, -1], [-1, 0, 0]]),
]

B_1_mom_1_1_1 = [
    (-1/sqrt(2), 'pipi', [[-1, 0, -1], [0, -1, 0]]),
    (1/sqrt(2), 'pipi', [[0, -1, -1], [-1, 0, 0]]),
]

B_0_mom_1_11 = [
    (-1/sqrt(6), 'pipi', [[-1, -1, 0], [0, 0, 1]]),
    (sqrt(2/3), 'pipi', [[-1, 0, 1], [0, -1, 0]]),
    (-1/sqrt(6), 'pipi', [[0, -1, 1], [-1, 0, 0]]),
]

B_1_mom111 = [
    (-1/sqrt(2), 'pipi', [[0, 1, 1], [1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[1, 0, 1], [0, 1, 0]]),
]

B_1_mom_1_11 = [
    (-1/sqrt(2), 'pipi', [[-1, -1, 0], [0, 0, 1]]),
    (1/sqrt(2), 'pipi', [[0, -1, 1], [-1, 0, 0]]),
]

B_0_mom111 = [
    (-1/sqrt(6), 'pipi', [[0, 1, 1], [1, 0, 0]]),
    (-1/sqrt(6), 'pipi', [[1, 0, 1], [0, 1, 0]]),
    (sqrt(2/3), 'pipi', [[1, 1, 0], [0, 0, 1]]),
]

B_1_mom_111 = [
    (-1/sqrt(2), 'pipi', [[-1, 0, 1], [0, 1, 0]]),
    (1/sqrt(2), 'pipi', [[0, 1, 1], [-1, 0, 0]]),
]

B_1_mom_11_1 = [
    (-1/sqrt(2), 'pipi', [[-1, 1, 0], [0, 0, -1]]),
    (1/sqrt(2), 'pipi', [[0, 1, -1], [-1, 0, 0]]),
]

B_1_mom1_1_1 = [
    (1/sqrt(2), 'pipi', [[0, -1, -1], [1, 0, 0]]),
    (-1/sqrt(2), 'pipi', [[1, -1, 0], [0, 0, -1]]),
]

B_1_mom1_11 = [
    (-1/sqrt(2), 'pipi', [[0, -1, 1], [1, 0, 0]]),
    (1/sqrt(2), 'pipi', [[1, -1, 0], [0, 0, 1]]),
]

B_0_mom_11_1 = [
    (sqrt(2/3), 'pipi', [[-1, 0, -1], [0, 1, 0]]),
    (-1/sqrt(6), 'pipi', [[-1, 1, 0], [0, 0, -1]]),
    (-1/sqrt(6), 'pipi', [[0, 1, -1], [-1, 0, 0]]),
]

B_0_mom1_1_1 = [
    (-1/sqrt(6), 'pipi', [[0, -1, -1], [1, 0, 0]]),
    (-1/sqrt(6), 'pipi', [[1, -1, 0], [0, 0, -1]]),
    (sqrt(2/3), 'pipi', [[1, 0, -1], [0, -1, 0]]),
]

B_0_mom11_1 = [
    (sqrt(2/3), 'pipi', [[0, 1, -1], [1, 0, 0]]),
    (-1/sqrt(6), 'pipi', [[1, 0, -1], [0, 1, 0]]),
    (-1/sqrt(6), 'pipi', [[1, 1, 0], [0, 0, -1]]),
]

B_0_mom_111 = [
    (-1/sqrt(6), 'pipi', [[-1, 0, 1], [0, 1, 0]]),
    (sqrt(2/3), 'pipi', [[-1, 1, 0], [0, 0, 1]]),
    (-1/sqrt(6), 'pipi', [[0, 1, 1], [-1, 0, 0]]),
]


def row(irr):
    ret = irr.split('_')[1]
    return int(ret)


def pol_coeff(comp):
    """Get polarization coefficients
    from center of mass momentum"""
    isnzero = int(sum([abs(i) for i in comp]))
    if isnzero == 3:
        ret = np.cross(comp, [1, 0, 0])
    elif isnzero == 2:
        assert None, "fix this"
        ret = []
    else:
        assert None, "comp should be p111 or p11"
    return list(ret)

def lstr(arr):
    """Make sure the string will convert back to a list"""
    ret = list(arr)
    ret = str(ret)
    return ret

def sumabs(mom):
    """Sum the absolute value of the momentum
    [1,-1,0] => 2
    """
    ret = em.acsum(np.abs(mom))
    return ret

def sortmom(irrvar, irr):
    """Enforce Luchang's condition
    that inner pions should be higher energy
    """
    ret = []
    for i in irrvar:
        moms = list(i[2])
        if len(moms) == 2:
            if sumabs(moms[0]) > sumabs(moms[1]):
                moms[0], moms[1] = moms[1], moms[0]
                toapp = (i[0], i[1], moms)
                assert str(toapp) != str(i)
                if toapp in irrvar:
                    ret = irrvar
                    print(irr, "has reverse")
                    break
            else:
                toapp = i
        ret.append(toapp)
    assert len(ret) == len(irrvar), "bad return length:"+\
        str(ret)+" "+str(irrvar)
    return ret

# add the rho operator via this hack
OPLIST = {}
for irr in dir(cmod):
    if 'mom' not in irr:
        continue
    else:
        irrvar = getattr(cmod, irr)
        if not hasattr(irrvar, '__iter__'):
            continue
        irrvar = sortmom(irrvar, irr)
        mom = rf.mom(irr)
        assert len(mom) == 3, "bad momentum specified:"+str(mom)
        for i in mom:
            assert isinstance(i, int), "momentum has non-int value:"+str(mom)
        toadd = [(1, 'rho', list(mom))]
        irrvar.insert(0, *toadd)
        key = str(irr)+'?pol='
        tarr = [bool(i) for i in mom]

        #pols for p1
        if 'A_1PLUS' in irr and sum(np.abs(mom)) == 1:
            key += str(tarr.index(True)+1)
        elif 'B' in irr and sum(np.abs(mom)) == 1:
            if row(irr):
                tarr[tarr.index(False)] = True
            pol = int(tarr.index(False)) + 1
            key += str(pol)

        #pols for p11
        elif 'A_1PLUS' in irr and sum(np.abs(mom)) == 2:
            key += lstr(mom)
        elif 'A_2PLUS' in irr and sum(np.abs(mom)) == 2:
            key += str(tarr.index(False)+1)
        elif 'A_2MINUS' in irr and sum(np.abs(mom)) == 2:
            key += lstr(np.cross(mom, [(1 if not i else 0) for i in tarr]))

        #pols for p111
        elif 'A_1PLUS' in irr and sum(np.abs(mom)) == 3:
            key += lstr(mom)
        elif 'B' in irr and sum(np.abs(mom)) == 3:
            pollist = pol_coeff(mom)
            if row(irr):
                key += lstr(np.cross(pollist, mom))
            else:
                key += lstr(pollist)

        # default
        else:
            assert None, "bad irrep specified:"+str(irr)
        OPLIST[key] = irrvar

AVG_ROWS = {
    'A_1PLUS_mom1': ('A_1PLUS_mom00_1',
                     'A_1PLUS_mom001',
                     'A_1PLUS_mom010',
                     'A_1PLUS_mom0_10',
                     'A_1PLUS_mom_100',
                     'A_1PLUS_mom100',),
    'B_mom1': ('B_1_mom00_1',
               'B_1_mom001',
               'B_1_mom010',
               'B_1_mom0_10',
               'B_1_mom_100',
               'B_1_mom100',
               'B_0_mom00_1',
               'B_0_mom001',
               'B_0_mom010',
               'B_0_mom0_10',
               'B_0_mom_100',
               'B_0_mom100',),
    'A_1PLUS_mom11': ('A_1PLUS_mom011',
                      'A_1PLUS_mom0_11',
                      'A_1PLUS_mom01_1',
                      'A_1PLUS_mom0_1_1',
                      'A_1PLUS_mom101',
                      'A_1PLUS_mom_101',
                      'A_1PLUS_mom10_1',
                      'A_1PLUS_mom_10_1',
                      'A_1PLUS_mom110',
                      'A_1PLUS_mom_110',
                      'A_1PLUS_mom1_10',
                      'A_1PLUS_mom_1_10',),
    'A_2PLUS_mom11': ('A_2PLUS_mom011',
                      'A_2PLUS_mom0_11',
                      'A_2PLUS_mom01_1',
                      'A_2PLUS_mom0_1_1',
                      'A_2PLUS_mom101',
                      'A_2PLUS_mom_101',
                      'A_2PLUS_mom10_1',
                      'A_2PLUS_mom_10_1',
                      'A_2PLUS_mom110',
                      'A_2PLUS_mom_110',
                      'A_2PLUS_mom1_10',
                      'A_2PLUS_mom_1_10',),
    'A_2MINUS_mom11': ('A_2MINUS_mom011',
                       'A_2MINUS_mom0_11',
                       'A_2MINUS_mom01_1',
                       'A_2MINUS_mom0_1_1',
                       'A_2MINUS_mom101',
                       'A_2MINUS_mom_101',
                       'A_2MINUS_mom10_1',
                       'A_2MINUS_mom_10_1',
                       'A_2MINUS_mom110',
                       'A_2MINUS_mom_110',
                       'A_2MINUS_mom1_10', 'A_2MINUS_mom_1_10',),
    'B_mom111': ('B_0_mom111',
                 'B_0_mom_1_1_1',
                 'B_0_mom_111',
                 'B_0_mom1_11',
                 'B_0_mom11_1',
                 'B_0_mom1_1_1',
                 'B_0_mom_11_1',
                 'B_0_mom_1_11',
                 'B_1_mom111',
                 'B_1_mom_1_1_1',
                 'B_1_mom_111',
                 'B_1_mom1_11',
                 'B_1_mom11_1',
                 'B_1_mom1_1_1',
                 'B_1_mom_11_1',
                 'B_1_mom_1_11',),
    'A_1PLUS_avg_mom111': ('A_1PLUS_mom111',
                           'A_1PLUS_mom_1_1_1',
                           'A_1PLUS_mom_111',
                           'A_1PLUS_mom1_11',
                           'A_1PLUS_mom11_1',
                           'A_1PLUS_mom1_1_1',
                           'A_1PLUS_mom_11_1',
                           'A_1PLUS_mom_1_11'),}

for i in AVG_ROWS:
    assert len(set(AVG_ROWS[i])) == len(list(AVG_ROWS[i])),\
        "duplicate found:"+str(AVG_ROWS[i])
    count = 0
    for j in AVG_ROWS[i]:
        assert j in dir(cmod), "unknown irrep:"+str(j)
        irrvar = getattr(cmod, j)
        count = len(irrvar) if not count else count
        if 'B' not in j:
            assert len(irrvar) == count, "bad irrep length:"+str(j)+" "+str(irrvar)
