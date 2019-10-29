import numpy as np
from gvar import gvar
from latfit.config import GEVP

def fit_range_consistency_check(meta, min_arr, name):
    """Check consistency of energies and phase shifts so far"""
    lparam = []
    for i in min_arr:
        lparam.append(getattr(i[0], name))
    consis = consistent_list_params(lparam)
    err_handle(meta, consis, lparam, name)

def err_handle(meta, consis, lparam, name):
    """Handle the case where the fit ranges give inconsistent results"""
    try:
        assert consis
    except AssertionError:
        print("fit ranges are inconsistent with respect to:", name)
        print("fit window:", meta.fitwindow)
        for i in sort_by_val(lparam):
            print(gvar(i.val, i.err))
        raise
    
def sort_by_val(lparam):
    """Sort a list of Param objects by val"""
    assert isinstance(lparam, list), str(lparam)
    if hasattr(lparam[0].val, '__iter__'):
        ret = sorted(lparam, key=lambda param: param.val[0])
    else:
        assert not GEVP, str(lparam)
        ret = sorted(lparam, key=lambda param: param.val)
    return ret

def consistent_list_params(lparam):
    """Check the consistency across a list of Param objects"""
    ret = True
    for i in lparam:
        for j in lparam:
            ret = consistent_params(i, j)
            if not ret:
                break
    return ret

def consistent_params(item1, item2):
    """Check the consistency of two Param objects
    if discrepant by > 1.5 sigma, return False (inconsistent)
    """
    diff = item1.val-item2.val
    if GEVP:
        diff = np.asarray(diff)
        err = []
        for i, _ in enumerate(item1.err):
            err.append(max(item1.err[i], item2.err[i]))
        err = np.array(err)
    else:
        err = max(item1.err, item2.err)
    err = np.asarray(err)
    diff = np.asarray(diff)
    test = np.max(np.abs(diff/err))
    ret = not test > 1.5
    return ret
