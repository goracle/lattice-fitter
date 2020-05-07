"""Cuts to apply to fit results"""
import sys
import numpy as np
import gvar
from latfit.config import ISOSPIN, STRONG_CUTS
from latfit.utilities.postfit.fitwin import LENMIN
from latfit.utilities.postfit.fitwin import lenfitw, inside_win
from latfit.utilities.postfit.fitwin import wintoosmall, get_fitwindow
from latfit.utilities.postfit.strproc import round_wrt
from latfit.checks.consistency import mod180

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def fitrange_cuts(median_err, fit_range_arr, twin, dim):
    """Get fit window, apply cuts"""
    fitwindow = get_fitwindow(fit_range_arr, twin, prin=True)
    skip_list = fitrange_skip_list(fit_range_arr, fitwindow, dim)
    median_err = cut_arr(median_err, skip_list)
    fit_range_arr = cut_arr(fit_range_arr, skip_list)
    return median_err, fit_range_arr

@PROFILE
def statlvl(diff):
    """Calculate the statistical significance of a gvar diff"""
    if diff.val:
        if diff.sdev:
            sig = diff.val/diff.sdev
        else:
            sig = np.inf
    else:
        sig = 0
    if np.isnan(diff.val) or np.isnan(diff.sdev):
        sig = np.nan
    return sig



@PROFILE
def fitrange_skip_list(fit_range_arr, fitwindow, dim):
    """List of indices to skip"""
    # fit window cut
    ret = set()
    lcut = 0
    fcut = 0
    acut = 0
    for idx, item in enumerate(fit_range_arr):
        if lencut(item, dim):
            lcut += 1
            ret.add(idx)
        elif fitwincuts(item, fitwindow):
            fcut += 1
            ret.add(idx)
        if (ISOSPIN or STRONG_CUTS) and idx not in ret:
            if not arithseq(item):
                acut += 1
                ret.add(idx)
    ret = sorted(list(ret))
    print("length cut amt:", lcut)
    print("fit window cut amt:", fcut)
    print("arith. seq. cut amt:", acut)
    return ret

@PROFILE
def cut_arr(arr, skip_list):
    """Prune the results array"""
    return np.delete(arr, skip_list, axis=0)

@PROFILE
def lencut(fit_range, dim):
    """Length cut; we require each fit range to have
    a minimum number of time slices
    length cut for safety (better systematic error control
    to include more data in a given fit range;
    trade stat for system.)
    only apply if the data is plentiful (I=2)
    """
    ret = False
    iterf = hasattr(fit_range[0], '__iter__')
    effmasspt = not iterf and len(fit_range) == 1
    if effmasspt:
        ret = True
    if not ret:
        if iterf:
            if ISOSPIN or STRONG_CUTS: # I = 1, 2
                ret = any([len(i) < LENMIN for i in fit_range])
            else: # I = 0
                ret = [len(i) < LENMIN for i in fit_range][dim]
        else:
            ret = len(fit_range) < LENMIN
    if ret:
        assert effmasspt or not ISOSPIN, fit_range
    return ret

@PROFILE
def arithseq(fitrange):
    """Check if arithmetic sequence"""
    ret = True
    assert hasattr(fitrange, '__iter__'), fitrange
    assert hasattr(fitrange[0], '__iter__'), fitrange
    for fitr in fitrange:
        minp = fitr[0]
        nextp = fitr[1]
        step = nextp-minp
        maxp = fitr[-1]
        rchk = np.arange(minp, maxp+step, step)
        if list(rchk) != list(fitr):
            ret = False
    return ret

@PROFILE
def allow_cut(res, dim, best, cutstat=True, chk_consis=True):
    """If we already have a minimized error result,
    cut all that are statistically incompatible"""
    ret = False
    if best:
        battr = allow_cut.best_attr
        if battr is None:
            battr = hasattr(gvar.gvar(best[0]), '__iter__')
            allow_cut.best_attr = battr
        if battr:
            for i in best:
                ret = ret or res_best_comp(
                    res, i, dim, cutstat=cutstat, chk_consis=chk_consis)
        else:
            ret = res_best_comp(
                res, best, dim, cutstat=cutstat, chk_consis=chk_consis)
    return ret
allow_cut.best_attr = None

@PROFILE
def res_best_comp(res, best, dim, chk_consis=True, cutstat=True):
    """Compare result with best known for consistency"""
    if hasattr(res, '__iter__'):
        sdev = res[0].sdev
        res = np.mean([i.val for i in res], axis=0)
        try:
            res = gvar.gvar(res, sdev)
        except TypeError:
            print(res)
            print(sdev)
            raise
    best = best[dim]
    # best = gvar.gvar(best)
    ret = False
    if chk_consis:
        ret = not consistency(best, res, prin=False)
    if cutstat:
        if best is None:
            devc = np.inf
        else:
            devc = best.sdev if not np.isnan(best.sdev) else np.inf
        try:
            # make cut not aggressive since it's a speed trick
            devd = min(round_wrt(devc, res.sdev), res.sdev)
            ret = ret or devc < devd
        except ValueError:
            print('ret', ret)
            print('best', best)
            print('res', res)
            raise
    return ret

@PROFILE
def fitwincuts(fit_range, fitwindow, dim=None):
    """Cut all fit ranges outside this fit window,
    also cut out fit windows that are too small
    """
    ret = False

    # skip fit windows of a small length
    if wintoosmall(fitwindow):
        ret = True
    if not ret:
        # tmin, tmax cut
        dim = None # appears to help somewhat with errors
        iterf = hasattr(fit_range[0], '__iter__')
        if dim is not None:
            assert iterf, fit_range
            fit_range = fit_range[dim]
    for i, fitr in enumerate(fit_range):
        if ret:
            break
        if not hasattr(fitr, '__iter__'):
            ret = True
            break
        if i == dim:
            ret = not inside_win(fitr, fitwindow) or ret
        else:
            #dimwin = DIMWIN[i] if iterf else fitwindow
            dimwin = fitwindow
            ret = not inside_win(fitr, dimwin) or ret
    if not ret and lenfitw(fitwindow) == 1:
        print(fitwindow)
        print(fit_range)
        sys.exit(1)
    return ret

@PROFILE
def consistency(item1, item2, prin=False):
    """Check two gvar items for consistency"""
    if item1 is None or item2 is None:
        ret = True
    else:
        item1a = gvar.gvar(mod180(item1.val), item1.sdev)
        item2a = gvar.gvar(mod180(item2.val), item2.sdev)
        diff = np.abs(item1.val-item2.val)
        diff2 = np.abs(item1a.val-item2a.val)
        diff3 = np.abs(item1.val-item2a.val)
        diff4 = np.abs(item1a.val-item2.val)
        diff = min(diff, diff2, diff3, diff4)
        dev = max(item1.sdev, item2.sdev)
        sig = statlvl(gvar.gvar(diff, dev))
        ret = np.allclose(0, max(0, sig-1.5), rtol=1e-12)
        if not ret:
            if prin:
                print("sig inconsis. =", sig)
            # sanity check; relax '15' to a higher number as necessary
            assert sig < 15 or not ISOSPIN,\
                (sig, "check the best known list for",
                 "compatibility with current set",
                 "of results being analyzed", item1, item2)
    return ret
