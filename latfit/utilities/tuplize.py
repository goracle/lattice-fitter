"""Make mutable containers immutable"""
import numpy as np

def tupl_mat(mat):
    """Make matrix immutable"""
    ret = tuple(tuple(i) for i in mat)
    return ret

def list_mat(mat):
    """Make immutable matrix mutable"""
    ret = list(list(i) for i in mat)
    return ret

