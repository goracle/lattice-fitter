#!/usr/bin/python3
"""A wrapper for Shewchuk summation; to be used as an np.mean replacement"""

import sys
import accupy
import numpy as np
from numpy import swapaxes as swap

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def convert_arr(arr):
    """Convert dtype since ksum only likes np.float64 types"""
    if arr.dtype != 'float64':
        arr = np.array(arr, dtype=np.float64)
    return arr

def acstd(arr, axis=0, ddof=0, fsum=False):
    """Compute standard deviation given ddof degrees of freedom"""
    ret = acsum((arr-acmean(arr, axis=axis, fsum=fsum))**2,
                 axis=axis, fsum=fsum)
    ret /= len(arr)-ddof
    ret = np.sqrt(ret)
    return ret
    
def acsum(arr, axis=0, fsum=False):
    """Peform accurate summation"""
    assert isinstance(arr, np.ndarray), "input is not a numpy array"
    if not axis:
        ret = complexsum(arr, fsum)
    else:
        arr = swap(arr, 0, axis)
        ret = complexum(arr)
        arr = swap(arr, 0, axis)
    return ret

@PROFILE
def acmean(arr, axis=0, fsum=False):
    """Take the average of the array"""
    ret = acsum(arr, axis, fsum)
    ret /= len(arr)
    return ret

def complexsum(arr, fsum):
    """ Handles complex arrays
    """
    if 'complex' in str(arr.dtype):
        real = dosum(np.real(arr), fsum)
        imag = dosum(np.imag(arr), fsum)
        ret = real+imag*1j
    else:
        ret = dosum(arr, fsum)
    return ret


def dosum(arr, fsum):
    """Perform the average"""
    if fsum:
        ret = accupy.fsum(arr)
    else:
        arr = convert_arr(arr)
        ret = accupy.ksum(arr)
    return ret


@PROFILE
def kahan_sum(a, axis=0):
    """Standard Kahan sum"""
    a = np.asarray(a)
    s = np.zeros(a.shape[:axis] + a.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(a.shape[axis]):
        # http://stackoverflow.com/a/42817610/353337
        y = a[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()
    return s
