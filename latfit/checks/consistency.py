import numpy as np
from gvar import gvar

def fit_range_consistency_check(min_arr, name):
    """Check consistency of energies and phase shifts so far"""
    lparam = []
    for i in min_arr:
        lparam.append(getattr(i, name))
    consis = consistent_list_params(lparam)
    err_handle(consis, lparam, name)

def err_handle(consis, lparam, name):
    """Handle the case where the fit ranges give inconsistent results"""
    try:
        assert consis
    except AssertionError:
        print("fit ranges are inconsistent with respect to:", name)
        for i in sort_by_val(laram):
            print(gvar(i.val, i.err))
        raise
    
def sort_by_val(lparam):
    """Sort a list of Param objects by val"""
    assert isinstance(lparam, list), str(lparam)
    if hasattr(lparam[0], '__iter__'):
        ret = sorted(lparam, key=lambda param: param.val[0])
    else:
        ret = sorted(lparam, key=lambda param: param.val)
    return ret

def consistent_list_params(lparam):
    """Check the consistency across a list of Param objects"""
    ret = True
    for i in lparam:
        for j in lparam:
            ret = consistent_params(i, j)
    return ret

def consistent_params(item1, item2):
    """Check the consistency of two Param objects
    if discrepant by > 1.5 sigma, return False (inconsistent)
    """
    err = max(item.err, item2.err)
    diff = item1.val-item2.val
    err = np.asarray(err)
    diff = np.asarray(diff)
    ret = False if np.any(diff/err > 1.5) else True
    return ret
