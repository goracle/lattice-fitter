#!/usr/bin/python3
"""A wrapper for Shewchuk summation; to be used as an np.mean replacement"""

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

@PROFILE
def convert_arr(arr):
    """Convert dtype since ksum only likes np.float64 types"""
    if arr.dtype != 'float64':
        arr = np.array(arr, dtype=np.float64)
    return arr

@PROFILE
def acstd(arr, axis=0, ddof=0, fsum=False):
    """Compute standard deviation given ddof degrees of freedom"""
    larr = (np.asarray(arr).shape)[axis]
    ret = acsum((arr-acmean(arr, axis=axis, fsum=fsum))**2,
                axis=axis, fsum=fsum)
    ret /= larr - ddof
    ret = np.sqrt(ret)
    return ret

@PROFILE
def acsum(arr, axis=0, fsum=False):
    """Peform accurate summation"""
    arr = np.asarray(arr)
    assert isinstance(arr, np.ndarray), "input is not a numpy array"
    if not axis:
        ret = complexsum(arr, fsum)
    else:
        arr = swap(arr, 0, axis)
        ret = complexsum(arr, fsum)
        arr = swap(arr, 0, axis)
    return ret

@PROFILE
def acmean(arr, axis=0, fsum=False):
    """Take the average of the array"""
    ret = np.asarray(arr)
    if hasattr(ret, '__iter__'):
        larr = (ret.shape)[axis]
        if larr:
            ret = acsum(ret, axis=axis, fsum=fsum)
            ret /= larr
    return ret

@PROFILE
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


@PROFILE
def dosum(arr, fsum):
    """Perform the average"""
    if fsum:
        ret = accupy.fsum(arr)
    else:
        arr = convert_arr(arr)
        ret = accupy.ksum(arr)
    return ret


@PROFILE
def kahan_sum(arr, axis=0):
    """Standard Kahan sum"""
    arr = np.asarray(arr)
    sarr = np.zeros(arr.shape[:axis] + arr.shape[axis+1:])
    carr = np.zeros(sarr.shape)
    for i in range(arr.shape[axis]):
        # http://stackoverflow.com/a/42817610/353337
        yarr = arr[(slice(None),) * axis + (i,)] - carr
        tarr = sarr + yarr
        carr = (tarr - sarr) - yarr
        sarr = tarr.copy()
    return sarr
