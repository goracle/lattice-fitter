"""Check consistency of fit results"""
import numpy as np
from gvar import gvar
from latfit.config import GEVP
from latfit.config import MATRIX_SUBTRACTION
from latfit.analysis.errorcodes import FitRangeInconsistency
from latfit.analysis.filename_windows import filename_plus_config_info
import latfit.config

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

# https://stackoverflow.com/questions/1158076/implement-touch-using-python
@PROFILE
def touch(fname, mode=0o666, dir_fd=None, **kwargs):
    """unix touch"""
    flags = os.O_CREAT | os.O_APPEND
    with os.fdopen(os.open(fname, flags=flags,
                           mode=mode, dir_fd=dir_fd)) as fn1:
        os.utime(fn1.fileno() if os.utime in os.supports_fd else fname,
                 dir_fd=None if os.supports_fd else dir_fd, **kwargs)

def create_dummy_skip(meta):
    """If we find an inconsistent fit,
    create a dummy file so that we skip that in any repeat run"""
    fname = filename_plus_config_info(meta, 'pvalue')+'.p'
    print("creating skip file:", fname)
    touch(fname+'.p')

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
        meta.pr_fit_window()
        print("t-t0:", latfit.config.T0)
        if MATRIX_SUBTRACTION:
            print("dt matsub:", latfit.config.DELTA_T_MATRIX_SUBTRACTION)
        for i in sort_by_val(lparam):
            print(gvar(i.val, i.err))
        create_dummy_skip(meta)
        raise FitRangeInconsistency

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
        if not ret:
            break
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
    tlist = list(np.abs(diff/err))
    test = np.max(tlist)
    try:
        idx = tlist.index(test)
    except ValueError:
        print(tlist)
        raise
    ret = not test > 1.5
    if not ret:
        print("problematic diff:", gvar(item1.val[idx], item1.err[idx]),
              gvar(item2.val[idx], item2.err[idx]))
    return ret
