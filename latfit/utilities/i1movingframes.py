"""All the irreps for the I=1 moving frames"""
## I=1 moving frames

## p1 == A1^+ \circleplus B
import sys
from math import sqrt
import read_file as rf
current_module = sys.modules[__name__]
cmod = current_module

A_1PLUS_mom00_1 = [
    (-0.5, 'pipi', [[-1, 0, -1], [1, 0, 0]]) ,
    (-0.5, 'pipi', [[0, -1, -1], [0, 1, 0]]) ,
    (-0.5, 'pipi', [[0, 1, -1], [0, -1, 0]]) ,
    (-0.5, 'pipi', [[1, 0, -1], [-1, 0, 0]]) ,
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, -1, 0]]) ,
    (-0.5, 'UUpipi', [[1, -1, -1], [-1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[1, 1, -1], [-1, -1, 0]]) ,
]

A_1PLUS_mom0_10 = [
    (-0.5, 'pipi', [[-1, -1, 0], [1, 0, 0]]) ,
    (-0.5, 'pipi', [[0, -1, -1], [0, 0, 1]]) ,
    (-0.5, 'pipi', [[0, -1, 1], [0, 0, -1]]) ,
    (-0.5, 'pipi', [[1, -1, 0], [-1, 0, 0]]) ,
    (-0.5, 'UUpipi', [[-1, -1, -1], [1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[-1, -1, 1], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, -1, -1], [-1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[1, -1, 1], [-1, 0, -1]]) ,
]

A_1PLUS_mom_100 = [
    (-0.5, 'pipi', [[-1, -1, 0], [0, 1, 0]]) ,
    (-0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]) ,
    (-0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]) ,
    (-0.5, 'pipi', [[-1, 1, 0], [0, -1, 0]]) ,
    (-0.5, 'UUpipi', [[-1, -1, -1], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[-1, -1, 1], [0, 1, -1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, -1], [0, -1, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 1], [0, -1, -1]]) ,
]

A_1PLUS_mom100 = [
    (-0.5, 'pipi', [[1, -1, 0], [0, 1, 0]]) ,
    (-0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]) ,
    (-0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]) ,
    (-0.5, 'pipi', [[1, 1, 0], [0, -1, 0]]) ,
    (-0.5, 'UUpipi', [[1, -1, -1], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[1, -1, 1], [0, 1, -1]]) ,
    (-0.5, 'UUpipi', [[1, 1, -1], [0, -1, 1]]) ,
    (-0.5, 'UUpipi', [[1, 1, 1], [0, -1, -1]]) ,
]

A_1PLUS_mom010 = [
    (-0.5, 'pipi', [[-1, 1, 0], [1, 0, 0]]) ,
    (-0.5, 'pipi', [[0, 1, -1], [0, 0, 1]]) ,
    (-0.5, 'pipi', [[0, 1, 1], [0, 0, -1]]) ,
    (-0.5, 'pipi', [[1, 1, 0], [-1, 0, 0]]) ,
    (-0.5, 'UUpipi', [[-1, 1, -1], [1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 1], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, 1, -1], [-1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[1, 1, 1], [-1, 0, -1]]) ,
]

A_1PLUS_mom001 = [
    (-0.5, 'pipi', [[-1, 0, 1], [1, 0, 0]]) ,
    (-0.5, 'pipi', [[0, -1, 1], [0, 1, 0]]) ,
    (-0.5, 'pipi', [[0, 1, 1], [0, -1, 0]]) ,
    (-0.5, 'pipi', [[1, 0, 1], [-1, 0, 0]]) ,
]

B_mom00_1 = [
    (-0.5, 'pipi', [[-1, 0, -1], [1, 0, 0]]) ,
    (0.5, 'pipi', [[1, 0, -1], [-1, 0, 0]]) ,
]

B_mom0_10 = [
    (-0.5, 'pipi', [[-1, -1, 0], [1, 0, 0]]) ,
    (0.5, 'pipi', [[1, -1, 0], [-1, 0, 0]]) ,
]

B_mom_100 = [
    (-0.5, 'pipi', [[-1, 0, -1], [0, 0, 1]]) ,
    (0.5, 'pipi', [[-1, 0, 1], [0, 0, -1]]) ,
]

B_mom100 = [
    (0.5, 'pipi', [[1, 0, -1], [0, 0, 1]]) ,
    (-0.5, 'pipi', [[1, 0, 1], [0, 0, -1]]) ,
]

B_mom010 = [
    (-0.5, 'pipi', [[-1, 1, 0], [1, 0, 0]]) ,
    (0.5, 'pipi', [[1, 1, 0], [-1, 0, 0]]) ,
]

B_mom001 = [
    (-0.5, 'pipi', [[-1, 0, 1], [1, 0, 0]]) ,
    (0.5, 'pipi', [[1, 0, 1], [-1, 0, 0]]) ,
]

## p11 == A1^+ (12) \circleplus A2^ + \circleplus A2^-

A_1PLUS_mom0_1_1  = [
    (-1.0, 'S_pipi', [[0, -1, -1], [0, 0, 0]]) ,
]

A_1PLUS_mom01_1  = [
    (-1.0, 'S_pipi', [[0, 1, -1], [0, 0, 0]]) ,
]

A_1PLUS_mom_101  = [
    (-1.0, 'S_pipi', [[-1, 0, 1], [0, 0, 0]]) ,
]

A_1PLUS_mom011  = [
    (-1.0, 'S_pipi', [[0, 1, 1], [0, 0, 0]]) ,
]

A_1PLUS_mom10_1  = [
    (-1.0, 'S_pipi', [[1, 0, -1], [0, 0, 0]]) ,
]

A_1PLUS_mom101  = [
    (-1.0, 'S_pipi', [[1, 0, 1], [0, 0, 0]]) ,
]

A_1PLUS_mom1_10  = [
    (-1.0, 'S_pipi', [[1, -1, 0], [0, 0, 0]]) ,
]

A_1PLUS_mom_10_1  = [
    (-1.0, 'S_pipi', [[-1, 0, -1], [0, 0, 0]]) ,
]

A_1PLUS_mom110  = [
    (-1.0, 'S_pipi', [[1, 1, 0], [0, 0, 0]]) ,
]

A_1PLUS_mom0_11  = [
    (-1.0, 'S_pipi', [[0, -1, 1], [0, 0, 0]]) ,
]

A_1PLUS_mom_1_10  = [
    (-1.0, 'S_pipi', [[-1, -1, 0], [0, 0, 0]]) ,
]

A_1PLUS_mom_110  = [
    (-1.0, 'S_pipi', [[-1, 1, 0], [0, 0, 0]]) ,
]


A_1PLUS_mom_1_10  += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [-1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, -1, 0]]) ,
]

A_1PLUS_mom0_1_1  += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, -1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, -1]]) ,
]

A_1PLUS_mom1_10  += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, -1, 0]]) ,
]

A_1PLUS_mom10_1  += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, -1]]) ,
]

A_1PLUS_mom0_11  += [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, -1, 0]]) ,
]

A_1PLUS_mom_110  += [
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [-1, 0, 0]]) ,
]

A_1PLUS_mom01_1  += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, -1]]) ,
]

A_1PLUS_mom011  += [
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]) ,
]

A_1PLUS_mom_101  += [
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [-1, 0, 0]]) ,
]

A_1PLUS_mom110  += [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [1, 0, 0]]) ,
]

A_1PLUS_mom_10_1  += [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [-1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, -1]]) ,
]

A_1PLUS_mom101  += [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [1, 0, 0]]) ,
]

A_2PLUS_mom_1_10  = [
    (1/sqrt(2), 'pipi', [[0, -1, 0], [-1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[-1, 0, 0], [0, -1, 0]]) ,
]

A_2PLUS_mom10_1  = [
    (1/sqrt(2), 'pipi', [[0, 0, -1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, -1]]) ,
]

A_2PLUS_mom_101  = [
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [-1, 0, 0]]) ,
]

A_2PLUS_mom110  = [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 1, 0]]) ,
    (1/sqrt(2), 'pipi', [[0, 1, 0], [1, 0, 0]]) ,
]

A_2PLUS_mom0_11  = [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[0, 0, 1], [0, -1, 0]]) ,
]

A_2PLUS_mom01_1  = [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [0, 1, 0]]) ,
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, -1]]) ,
]

A_2PLUS_mom0_1_1  = [
    (1/sqrt(2), 'pipi', [[0, 0, -1], [0, -1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [0, 0, -1]]) ,
]

A_2PLUS_mom_110  = [
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[0, 1, 0], [-1, 0, 0]]) ,
]

A_2PLUS_mom101  = [
    (-1/sqrt(2), 'pipi', [[1, 0, 0], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[0, 0, 1], [1, 0, 0]]) ,
]

A_2PLUS_mom1_10  = [
    (-1/sqrt(2), 'pipi', [[0, -1, 0], [1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, 0, 0], [0, -1, 0]]) ,
]

A_2PLUS_mom_10_1  = [
    (-1/sqrt(2), 'pipi', [[0, 0, -1], [-1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[-1, 0, 0], [0, 0, -1]]) ,
]

A_2PLUS_mom011  = [
    (1/sqrt(2), 'pipi', [[0, 1, 0], [0, 0, 1]]) ,
    (-1/sqrt(2), 'pipi', [[0, 0, 1], [0, 1, 0]]) ,
]

A_1PLUS_mom0_1_1  += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [-1, 0, 0]]) ,
]

A_1PLUS_mom01_1  += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [-1, 0, 0]]) ,
]

A_1PLUS_mom_101  += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [0, -1, 0]]) ,
]

A_1PLUS_mom011  += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [-1, 0, 0]]) ,
]

A_1PLUS_mom10_1  += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [0, -1, 0]]) ,
]

A_1PLUS_mom101  += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [0, -1, 0]]) ,
]

A_1PLUS_mom1_10  += [
    (-1/sqrt(2), 'UUpipi', [[1, -1, -1], [0, 0, 1]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [0, 0, -1]]) ,
]

A_1PLUS_mom_10_1  += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [0, -1, 0]]) ,
]

A_1PLUS_mom110  += [
    (-1/sqrt(2), 'UUpipi', [[1, 1, -1], [0, 0, 1]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, 1, 1], [0, 0, -1]]) ,
]

A_1PLUS_mom0_11  += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'UUpipi', [[1, -1, 1], [-1, 0, 0]]) ,
]

A_1PLUS_mom_1_10  += [
    (-1/sqrt(2), 'UUpipi', [[-1, -1, -1], [0, 0, 1]]) ,
    (-1/sqrt(2), 'UUpipi', [[-1, -1, 1], [0, 0, -1]]) ,
]

A_1PLUS_mom_110  += [
    (-1/sqrt(2), 'UUpipi', [[-1, 1, -1], [0, 0, 1]]) ,
    (-1/sqrt(2), 'UUpipi', [[-1, 1, 1], [0, 0, -1]]) ,
]

A_2MINUS_mom_110  = [
    (-1/sqrt(2), 'pipi', [[-1, 1, -1], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[-1, 1, 1], [0, 0, -1]]) ,
]

A_2MINUS_mom0_1_1  = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, -1, -1], [-1, 0, 0]]) ,
]

A_2MINUS_mom_1_10  = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [0, 0, -1]]) ,
]

A_2MINUS_mom110  = [
    (-1/sqrt(2), 'pipi', [[1, 1, -1], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[1, 1, 1], [0, 0, -1]]) ,
]

A_2MINUS_mom10_1  = [
    (-1/sqrt(2), 'pipi', [[1, -1, -1], [0, 1, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, 1, -1], [0, -1, 0]]) ,
]

A_2MINUS_mom1_10  = [
    (-1/sqrt(2), 'pipi', [[1, -1, -1], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[1, -1, 1], [0, 0, -1]]) ,
]

A_2MINUS_mom0_11  = [
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, -1, 1], [-1, 0, 0]]) ,
]

A_2MINUS_mom101  = [
    (1/sqrt(2), 'pipi', [[1, -1, 1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, 1, 1], [0, -1, 0]]) ,
]

A_2MINUS_mom01_1  = [
    (-1/sqrt(2), 'pipi', [[-1, 1, -1], [1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, 1, -1], [-1, 0, 0]]) ,
]

A_2MINUS_mom_10_1  = [
    (-1/sqrt(2), 'pipi', [[-1, -1, -1], [0, 1, 0]]) ,
    (1/sqrt(2), 'pipi', [[-1, 1, -1], [0, -1, 0]]) ,
]

A_2MINUS_mom_101  = [
    (1/sqrt(2), 'pipi', [[-1, -1, 1], [0, 1, 0]]) ,
    (-1/sqrt(2), 'pipi', [[-1, 1, 1], [0, -1, 0]]) ,
]

A_2MINUS_mom011  = [
    (1/sqrt(2), 'pipi', [[-1, 1, 1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, 1, 1], [-1, 0, 0]]) ,
]




A_1PLUS_mom_101  += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [0, 1, 1]]) ,
    (-0.5, 'U2pipi', [[-1, 1, 0], [0, -1, 1]]) ,
    (-0.5, 'U2pipi', [[0, -1, 1], [-1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[0, 1, 1], [-1, -1, 0]]) ,
]

A_1PLUS_mom1_10  += [
    (-0.5, 'U2pipi', [[0, -1, -1], [1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[0, -1, 1], [1, 0, -1]]) ,
    (-0.5, 'U2pipi', [[1, 0, -1], [0, -1, 1]]) ,
    (-0.5, 'U2pipi', [[1, 0, 1], [0, -1, -1]]) ,
]

A_1PLUS_mom_10_1  += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [0, 1, -1]]) ,
    (-0.5, 'U2pipi', [[-1, 1, 0], [0, -1, -1]]) ,
    (-0.5, 'U2pipi', [[0, -1, -1], [-1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[0, 1, -1], [-1, -1, 0]]) ,
]

A_1PLUS_mom_1_10  += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [0, -1, 1]]) ,
    (-0.5, 'U2pipi', [[-1, 0, 1], [0, -1, -1]]) ,
    (-0.5, 'U2pipi', [[0, -1, -1], [-1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[0, -1, 1], [-1, 0, -1]]) ,
]

A_1PLUS_mom01_1  += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[-1, 1, 0], [1, 0, -1]]) ,
    (-0.5, 'U2pipi', [[1, 0, -1], [-1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[1, 1, 0], [-1, 0, -1]]) ,
]

A_1PLUS_mom110  += [
    (-0.5, 'U2pipi', [[0, 1, -1], [1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[0, 1, 1], [1, 0, -1]]) ,
    (-0.5, 'U2pipi', [[1, 0, -1], [0, 1, 1]]) ,
    (-0.5, 'U2pipi', [[1, 0, 1], [0, 1, -1]]) ,
]

A_1PLUS_mom101  += [
    (-0.5, 'U2pipi', [[0, -1, 1], [1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[0, 1, 1], [1, -1, 0]]) ,
    (-0.5, 'U2pipi', [[1, -1, 0], [0, 1, 1]]) ,
    (-0.5, 'U2pipi', [[1, 1, 0], [0, -1, 1]]) ,
]

A_1PLUS_mom_110  += [
    (-0.5, 'U2pipi', [[-1, 0, -1], [0, 1, 1]]) ,
    (-0.5, 'U2pipi', [[-1, 0, 1], [0, 1, -1]]) ,
    (-0.5, 'U2pipi', [[0, 1, -1], [-1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[0, 1, 1], [-1, 0, -1]]) ,
]

A_1PLUS_mom0_11  += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[-1, 0, 1], [1, -1, 0]]) ,
    (-0.5, 'U2pipi', [[1, -1, 0], [-1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[1, 0, 1], [-1, -1, 0]]) ,
]

A_1PLUS_mom011  += [
    (-0.5, 'U2pipi', [[-1, 0, 1], [1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[-1, 1, 0], [1, 0, 1]]) ,
    (-0.5, 'U2pipi', [[1, 0, 1], [-1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[1, 1, 0], [-1, 0, 1]]) ,
]

A_1PLUS_mom10_1  += [
    (-0.5, 'U2pipi', [[0, -1, -1], [1, 1, 0]]) ,
    (-0.5, 'U2pipi', [[0, 1, -1], [1, -1, 0]]) ,
    (-0.5, 'U2pipi', [[1, -1, 0], [0, 1, -1]]) ,
    (-0.5, 'U2pipi', [[1, 1, 0], [0, -1, -1]]) ,
]

A_1PLUS_mom0_1_1  += [
    (-0.5, 'U2pipi', [[-1, -1, 0], [1, 0, -1]]) ,
    (-0.5, 'U2pipi', [[-1, 0, -1], [1, -1, 0]]) ,
    (-0.5, 'U2pipi', [[1, -1, 0], [-1, 0, -1]]) ,
    (-0.5, 'U2pipi', [[1, 0, -1], [-1, -1, 0]]) ,
]

A_2MINUS_mom_10_1  += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, -1]]) ,
    (0.5, 'UUpipi', [[-1, 1, 0], [0, -1, -1]]) ,
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 1, 0]]) ,
    (0.5, 'UUpipi', [[0, 1, -1], [-1, -1, 0]]) ,
]

A_2MINUS_mom01_1  += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, -1]]) ,
    (0.5, 'UUpipi', [[1, 0, -1], [-1, 1, 0]]) ,
    (0.5, 'UUpipi', [[1, 1, 0], [-1, 0, -1]]) ,
]

A_2MINUS_mom110  += [
    (-0.5, 'UUpipi', [[0, 1, -1], [1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, 1, 1], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, 0, -1], [0, 1, 1]]) ,
    (0.5, 'UUpipi', [[1, 0, 1], [0, 1, -1]]) ,
]

A_2MINUS_mom0_1_1  += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, -1]]) ,
    (0.5, 'UUpipi', [[1, 0, -1], [-1, -1, 0]]) ,
]

A_2MINUS_mom011  += [
    (0.5, 'UUpipi', [[-1, 0, 1], [1, 1, 0]]) ,
    (0.5, 'UUpipi', [[-1, 1, 0], [1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, 1]]) ,
]

A_2MINUS_mom_1_10  += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, -1, 1]]) ,
    (0.5, 'UUpipi', [[-1, 0, 1], [0, -1, -1]]) ,
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 0, -1]]) ,
]

A_2MINUS_mom101  += [
    (0.5, 'UUpipi', [[0, -1, 1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[0, 1, 1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[1, 1, 0], [0, -1, 1]]) ,
]

A_2MINUS_mom0_11  += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, 1]]) ,
    (0.5, 'UUpipi', [[-1, 0, 1], [1, -1, 0]]) ,
    (-0.5, 'UUpipi', [[1, -1, 0], [-1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, -1, 0]]) ,
]

A_2MINUS_mom1_10  += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, -1, 1], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, 0, -1], [0, -1, 1]]) ,
    (0.5, 'UUpipi', [[1, 0, 1], [0, -1, -1]]) ,
]

A_2MINUS_mom_110  += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, 1, 1]]) ,
    (0.5, 'UUpipi', [[-1, 0, 1], [0, 1, -1]]) ,
    (-0.5, 'UUpipi', [[0, 1, -1], [-1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, 1, 1], [-1, 0, -1]]) ,
]

A_2MINUS_mom_101  += [
    (0.5, 'UUpipi', [[-1, -1, 0], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, 1]]) ,
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[0, 1, 1], [-1, -1, 0]]) ,
]

A_2MINUS_mom10_1  += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 1, 0]]) ,
    (0.5, 'UUpipi', [[0, 1, -1], [1, -1, 0]]) ,
    (-0.5, 'UUpipi', [[1, -1, 0], [0, 1, -1]]) ,
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, -1]]) ,
]


A_2PLUS_mom0_11  += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 0, 1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[1, 0, 1], [-1, -1, 0]]) ,
]

A_2PLUS_mom1_10  += [
    (0.5, 'UUpipi', [[0, -1, -1], [1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, -1, 1], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, 0, -1], [0, -1, 1]]) ,
    (-0.5, 'UUpipi', [[1, 0, 1], [0, -1, -1]]) ,
]

A_2PLUS_mom011  += [
    (0.5, 'UUpipi', [[-1, 0, 1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, 1]]) ,
    (0.5, 'UUpipi', [[1, 0, 1], [-1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, 1]]) ,
]

A_2PLUS_mom101  += [
    (-0.5, 'UUpipi', [[0, -1, 1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[0, 1, 1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, 1]]) ,
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, 1]]) ,
]

A_2PLUS_mom_1_10  += [
    (0.5, 'UUpipi', [[-1, 0, -1], [0, -1, 1]]) ,
    (0.5, 'UUpipi', [[-1, 0, 1], [0, -1, -1]]) ,
    (-0.5, 'UUpipi', [[0, -1, -1], [-1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[0, -1, 1], [-1, 0, -1]]) ,
]

A_2PLUS_mom110  += [
    (-0.5, 'UUpipi', [[0, 1, -1], [1, 0, 1]]) ,
    (-0.5, 'UUpipi', [[0, 1, 1], [1, 0, -1]]) ,
    (0.5, 'UUpipi', [[1, 0, -1], [0, 1, 1]]) ,
    (0.5, 'UUpipi', [[1, 0, 1], [0, 1, -1]]) ,
]

A_2PLUS_mom0_1_1  += [
    (0.5, 'UUpipi', [[-1, -1, 0], [1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[-1, 0, -1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [-1, 0, -1]]) ,
    (-0.5, 'UUpipi', [[1, 0, -1], [-1, -1, 0]]) ,
]

A_2PLUS_mom10_1  += [
    (-0.5, 'UUpipi', [[0, -1, -1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[0, 1, -1], [1, -1, 0]]) ,
    (0.5, 'UUpipi', [[1, -1, 0], [0, 1, -1]]) ,
    (0.5, 'UUpipi', [[1, 1, 0], [0, -1, -1]]) ,
]

A_2PLUS_mom_101  += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, 1]]) ,
    (0.5, 'UUpipi', [[0, -1, 1], [-1, 1, 0]]) ,
    (0.5, 'UUpipi', [[0, 1, 1], [-1, -1, 0]]) ,
]

A_2PLUS_mom01_1  += [
    (0.5, 'UUpipi', [[-1, 0, -1], [1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [1, 0, -1]]) ,
    (0.5, 'UUpipi', [[1, 0, -1], [-1, 1, 0]]) ,
    (-0.5, 'UUpipi', [[1, 1, 0], [-1, 0, -1]]) ,
]

A_2PLUS_mom_10_1  += [
    (-0.5, 'UUpipi', [[-1, -1, 0], [0, 1, -1]]) ,
    (-0.5, 'UUpipi', [[-1, 1, 0], [0, -1, -1]]) ,
    (0.5, 'UUpipi', [[0, -1, -1], [-1, 1, 0]]) ,
    (0.5, 'UUpipi', [[0, 1, -1], [-1, -1, 0]]) ,
]

A_2PLUS_mom_110  += [
    (-0.5, 'UUpipi', [[-1, 0, -1], [0, 1, 1]]) ,
    (-0.5, 'UUpipi', [[-1, 0, 1], [0, 1, -1]]) ,
    (0.5, 'UUpipi', [[0, 1, -1], [-1, 0, 1]]) ,
    (0.5, 'UUpipi', [[0, 1, 1], [-1, 0, -1]]) ,
]


## p111 == A1^+ \circleplus B

A_1PLUS_mom1_1_1  = [
    (-1/sqrt(3), 'pipi', [[0, -1, -1], [1, 0, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, -1, 0], [0, 0, -1]]) ,
    (-1/sqrt(3), 'pipi', [[1, 0, -1], [0, -1, 0]]) ,
]

A_1PLUS_mom_1_11  = [
    (-1/sqrt(3), 'pipi', [[-1, -1, 0], [0, 0, 1]]) ,
    (-1/sqrt(3), 'pipi', [[-1, 0, 1], [0, -1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[0, -1, 1], [-1, 0, 0]]) ,
]

A_1PLUS_mom11_1  = [
    (-1/sqrt(3), 'pipi', [[0, 1, -1], [1, 0, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, 0, -1], [0, 1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, 1, 0], [0, 0, -1]]) ,
]

A_1PLUS_mom_111  = [
    (-1/sqrt(3), 'pipi', [[-1, 0, 1], [0, 1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[-1, 1, 0], [0, 0, 1]]) ,
    (-1/sqrt(3), 'pipi', [[0, 1, 1], [-1, 0, 0]]) ,
]

A_1PLUS_mom_1_1_1  = [
    (-1/sqrt(3), 'pipi', [[-1, -1, 0], [0, 0, -1]]) ,
    (-1/sqrt(3), 'pipi', [[-1, 0, -1], [0, -1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[0, -1, -1], [-1, 0, 0]]) ,
]

A_1PLUS_mom_11_1  = [
    (-1/sqrt(3), 'pipi', [[-1, 0, -1], [0, 1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[-1, 1, 0], [0, 0, -1]]) ,
    (-1/sqrt(3), 'pipi', [[0, 1, -1], [-1, 0, 0]]) ,
]

A_1PLUS_mom111  = [
    (-1/sqrt(3), 'pipi', [[0, 1, 1], [1, 0, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, 0, 1], [0, 1, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, 1, 0], [0, 0, 1]]) ,
]

A_1PLUS_mom1_11  = [
    (-1/sqrt(3), 'pipi', [[0, -1, 1], [1, 0, 0]]) ,
    (-1/sqrt(3), 'pipi', [[1, -1, 0], [0, 0, 1]]) ,
    (-1/sqrt(3), 'pipi', [[1, 0, 1], [0, -1, 0]]) ,
]



# add an operator

A_1PLUS_mom111  += [
    (-1.0, 'pipi', [[1, 1, 1]]) ,
]

A_1PLUS_mom_1_11  += [
    (-1.0, 'pipi', [[-1, -1, 1]]) ,
]

A_1PLUS_mom11_1  += [
    (-1.0, 'pipi', [[1, 1, -1]]) ,
]

A_1PLUS_mom_111  += [
    (-1.0, 'pipi', [[-1, 1, 1]]) ,
]

A_1PLUS_mom_11_1  += [
    (-1.0, 'pipi', [[-1, 1, -1]]) ,
]

A_1PLUS_mom_1_1_1  += [
    (-1.0, 'pipi', [[-1, -1, -1]]) ,
]

A_1PLUS_mom1_11  += [
    (-1.0, 'pipi', [[1, -1, 1]]) ,
]

A_1PLUS_mom1_1_1  += [
    (-1.0, 'pipi', [[1, -1, -1]]) ,
]



# next p111 irrep, B

B_mom11_1  = [
     (1/sqrt(2), 'pipi', [[1, 0, -1], [0, 1, 0]]) ,
     (-1/sqrt(2), 'pipi', [[1, 1, 0], [0, 0, -1]]) ,
 ]

B_mom1_11  = [
    (-1/sqrt(6), 'pipi', [[0, -1, 1], [1, 0, 0]]) ,
    (-1/sqrt(6), 'pipi', [[1, -1, 0], [0, 0, 1]]) ,
    (sqrt(2/3), 'pipi', [[1, 0, 1], [0, -1, 0]]) ,
]

B_mom_1_1_1  = [
    (sqrt(2/3), 'pipi', [[-1, -1, 0], [0, 0, -1]]) ,
    (-1/sqrt(6), 'pipi', [[-1, 0, -1], [0, -1, 0]]) ,
    (-1/sqrt(6), 'pipi', [[0, -1, -1], [-1, 0, 0]]) ,
]

B_mom_1_1_1  = [
    (-1/sqrt(2), 'pipi', [[-1, 0, -1], [0, -1, 0]]) ,
    (1/sqrt(2), 'pipi', [[0, -1, -1], [-1, 0, 0]]) ,
]

B_mom_1_11  = [
    (-1/sqrt(6), 'pipi', [[-1, -1, 0], [0, 0, 1]]) ,
    (sqrt(2/3), 'pipi', [[-1, 0, 1], [0, -1, 0]]) ,
    (-1/sqrt(6), 'pipi', [[0, -1, 1], [-1, 0, 0]]) ,
]

B_mom111  = [
    (-1/sqrt(2), 'pipi', [[0, 1, 1], [1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, 0, 1], [0, 1, 0]]) ,
]

B_mom_1_11  = [
    (-1/sqrt(2), 'pipi', [[-1, -1, 0], [0, 0, 1]]) ,
    (1/sqrt(2), 'pipi', [[0, -1, 1], [-1, 0, 0]]) ,
]

B_mom111  = [
    (-1/sqrt(6), 'pipi', [[0, 1, 1], [1, 0, 0]]) ,
    (-1/sqrt(6), 'pipi', [[1, 0, 1], [0, 1, 0]]) ,
    (sqrt(2/3), 'pipi', [[1, 1, 0], [0, 0, 1]]) ,
]

B_mom_111  = [
    (-1/sqrt(2), 'pipi', [[-1, 0, 1], [0, 1, 0]]) ,
    (1/sqrt(2), 'pipi', [[0, 1, 1], [-1, 0, 0]]) ,
]

B_mom_11_1  = [
    (-1/sqrt(2), 'pipi', [[-1, 1, 0], [0, 0, -1]]) ,
    (1/sqrt(2), 'pipi', [[0, 1, -1], [-1, 0, 0]]) ,
]

B_mom1_1_1  = [
    (1/sqrt(2), 'pipi', [[0, -1, -1], [1, 0, 0]]) ,
    (-1/sqrt(2), 'pipi', [[1, -1, 0], [0, 0, -1]]) ,
]

B_mom1_11  = [
    (-1/sqrt(2), 'pipi', [[0, -1, 1], [1, 0, 0]]) ,
    (1/sqrt(2), 'pipi', [[1, -1, 0], [0, 0, 1]]) ,
]

B_mom_11_1  = [
    (sqrt(2/3), 'pipi', [[-1, 0, -1], [0, 1, 0]]) ,
    (-1/sqrt(6), 'pipi', [[-1, 1, 0], [0, 0, -1]]) ,
    (-1/sqrt(6), 'pipi', [[0, 1, -1], [-1, 0, 0]]) ,
]

B_mom1_1_1  = [
    (-1/sqrt(6), 'pipi', [[0, -1, -1], [1, 0, 0]]) ,
    (-1/sqrt(6), 'pipi', [[1, -1, 0], [0, 0, -1]]) ,
    (sqrt(2/3), 'pipi', [[1, 0, -1], [0, -1, 0]]) ,
]

B_mom11_1  = [
    (sqrt(2/3), 'pipi', [[0, 1, -1], [1, 0, 0]]) ,
    (-1/sqrt(6), 'pipi', [[1, 0, -1], [0, 1, 0]]) ,
    (-1/sqrt(6), 'pipi', [[1, 1, 0], [0, 0, -1]]) ,
]

B_mom_111  = [
    (-1/sqrt(6), 'pipi', [[-1, 0, 1], [0, 1, 0]]) ,
    (sqrt(2/3), 'pipi', [[-1, 1, 0], [0, 0, 1]]) ,
    (-1/sqrt(6), 'pipi', [[0, 1, 1], [-1, 0, 0]]) ,
]

for irr in dir(cmod):
    if 'mom' not in irr:
        continue
    if 'A_1PLUS' not in irr:
        continue
    else:
        irrvar = getattr(cmod, irr)
        mom = rf.mom(irr)
        toadd = [(1, 'rho', list(mom))]
        irrvar += toadd

