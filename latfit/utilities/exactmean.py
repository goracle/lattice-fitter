#!/usr/bin/python3
"""A wrapper for Shewchuk summation; to be used as an np.mean replacement"""

import sys
import accupy
import numpy as np
from numpy import swapaxes as swap

def acmean(arr, axis=0):
    """Take the average of the array, assuming """
    assert isinstance(arr, np.ndarray), "input is not a numpy array"
    ret = np.zeros(arr.shape, dtype=arr.dtype)
    if not axis:
        ret = domean(arr)
    else:
        arr = swap(arr, 0, axis)
        ret = domean(arr)
        arr = swap(arr, 0, axis)
    return ret

def domean(arr):
    """Perform the average"""
    ret = accupy.fsum(arr)
    ret /= len(arr)
    return ret
