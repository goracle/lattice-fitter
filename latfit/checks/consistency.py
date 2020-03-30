"""Check consistency of fit results"""
import os
import copy
# import itertools
import numpy as np
from gvar import gvar
from latfit.config import GEVP, VERBOSE
from latfit.config import MATRIX_SUBTRACTION, NOLOOP, DIMSELECT
from latfit.include import VALUE_STR, PARAM_OF_INTEREST
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

def mod180(deg):
    """Flip negative phase shift to pos"""
    deg = np.real(deg)
    if hasattr(deg, '__iter__'):
        ret = []
        for idx, item in enumerate(deg):
            if item >= 0:
                ret.append(item)
            else:
                ret.append(180+item)
        ret = np.asarray(ret)
    else:
        if deg >= 0:
            ret = deg
        else:
            ret = 180+deg
    return ret

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
    fname = filename_plus_config_info(meta, 'badfit')+'.p'
    print("creating skip file:", fname)
    touch(fname)

def fit_range_consistency_check(meta, min_arr, name, mod_180=False):
    """Check consistency of energies and phase shifts so far"""
    lparam = []
    for i in min_arr:
        lparam.append(getattr(i[0], name))
    consis = consistent_list_params(lparam, mod_180=mod_180,
                                    collapse_check='energy' in name)
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

def consistent_list_params(lparam, mod_180=False, collapse_check=False):
    """Check the consistency across a list of Param objects"""
    ret = True
    for i in lparam:
        if not ret:
            break
        for j in lparam:
            ret, _ = consistent_params(i, j, mod_180=mod_180, verb=True)
            if not ret:
                break
            if collapse_check:
                ret, idx, jdx = state_collapse_check(i, j, mod_180=mod_180)
                if not ret:
                    print("States", idx, "and", jdx,
                          "have collapsed into each other.")
                    print(gvar(i.val, i.err))
                    print(gvar(j.val, j.err))
                    break
    return ret

def state_collapse_check(iitem, jitem, mod_180=False):
    """Check all permutations of j's item against i's item for
    duplicate states"""
    ret = True
    idx, jdx = None, None
    if hasattr(iitem.val, '__iter__'):
        llen = len(iitem.val)
        for i in range(llen):
            for j in range(llen):
                #if i >= j or j == 2 or i == 2:
                if i >= j:
                    continue
                kitem = copy.deepcopy(jitem)
                litem = copy.deepcopy(iitem)
                kitem.swapidx(i, 0)
                kitem.swapidx(j, 1)
                litem.swapidx(i, 0)
                litem.swapidx(j, 1)
                litem.swapidx(0, 1)
                _, ret = consistent_params(kitem, litem, mod_180=mod_180)
                if not ret:
                    idx, jdx = i, j
                    break
    #for perm in list(itertools.permutations(range(len(iitem.val)))):
    #print("DEBUG:")
    #print('kitem', gvar(kitem.val, kitem.err))
    #print('iitem', gvar(i.val, i.err))
    #print("END DEBUG:")
    return ret, idx, jdx

def consistent_params(item1, item2, mod_180=False, verb=VERBOSE):
    """Check the consistency of two Param objects
    if discrepant by > 1.5 sigma, return False (inconsistent)
    """
    diff = np.abs(item1.val-item2.val)
    if mod_180:
        item1a = mod180(item1.val)
        item2a = mod180(item2.val)
        diff2 = np.abs(item1a-item2a)
        diff3 = np.abs(item1.val-item2a)
        diff4 = np.abs(item1a-item2.val)
    else:
        diff2 = None
    if diff2 is not None:
        if hasattr(diff, '__iter__'):
                diff = [min(i, j, k, l) for i, j, k, l in zip(
                    diff, diff2, diff3, diff4)]
        else:
            diff = min(diff1, diff2, diff3, diff4)
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
    test_inv = np.min(tlist[:2])
    try:
        idx = tlist.index(test)
    except ValueError:
        print(tlist)
        raise
    ret = not test > 1.5 # True => all consistent
    ret2 = test_inv > 1.5 # True => first two are both inconsistent
    str1 = str(gvar(item1.val[idx], item1.err[idx]))
    str2 = str(gvar(item2.val[idx], item2.err[idx]))
    cond = str1 == str2 and str1 == '0(0)'
    if not ret and not cond and verb:
        print("problematic diff:", str1,
              str2)
    return ret, ret2

def check_include(result_min):
    """Check that we've obtained the result we're after"""
    ret = not NOLOOP or not VALUE_STR
    if not ret:
        if 'phase shift' == PARAM_OF_INTEREST:
            param = result_min.phase_shift
        elif 'energy' == PARAM_OF_INTEREST:
            param = result_min.energy
        dim = DIMSELECT
        chk = gvar(param.val[dim], param.err[dim])
        chk = str(chk)
        if VERBOSE:
            print("chk, VALUE_STR:", chk, VALUE_STR)
        if chk == VALUE_STR:
            ret = True
    return ret
