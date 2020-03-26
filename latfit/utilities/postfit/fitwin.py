"""Fit window module"""
import numpy as np
from latfit.config import RANGE_LENGTH_MIN, ISOSPIN

LENMIN = 3
MIN_FITWIN_LEN = LENMIN + 1
assert LENMIN == RANGE_LENGTH_MIN
# I=0 we have fewer successful fit ranges, so apply a less stringent cut
if not ISOSPIN:
    LENMIN -= 1

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile




@PROFILE
def wintoosmall(win):
    """test if fit window is too small"""
    ret = False
    if lenfitw(win) < MIN_FITWIN_LEN:
        ret = True
    return ret

@PROFILE
def global_tmin(fit_range_arr):
    """Find the global tmin for this dimension:
    the minimum t for a successful fit"""
    tmin = np.inf
    for i in fit_range_arr:
        tee = np.inf
        if len(i) > 1:
            tee = min([min(j) for j in i])
        tmin = min(tee, tmin)
    return tmin

@PROFILE
def global_tmax(fit_range_arr):
    """Find the global tmax for this dimension:
    the maximum t for a successful fit"""
    tmax = 0
    for i in fit_range_arr:
        tee = 0
        if len(i) > 1:
            tee = max([max(j) for j in i])
        tmax = max(tmax, tee)
    return tmax

@PROFILE
def inside_win(fit_range, fitwin):
    """Check if fit range is inside the fit window"""
    assert hasattr(fit_range, '__iter__'), (fit_range, fitwin)
    iterf = hasattr(fit_range[0], '__iter__')
    if iterf:
        tmax = max([max(j) for j in fit_range])
        tmin = min([min(j) for j in fit_range])
    else:
        tmax = max(fit_range)
        tmin = min(fit_range)
    ret = tmax <= fitwin[1] and tmin >= fitwin[0]
    return ret


@PROFILE
def pr_best_fitwin(fitwin_votes):
    """Print the best fit window
    (most minimum results)"""
    mvotes = 0
    key = None
    for i in fitwin_votes:
        mvotes = max(fitwin_votes[i], mvotes)
        if mvotes == fitwin_votes[i]:
            key = i
    assert key is not None, key
    print("best fit window:", key)

def replace_inf_fitwin(fitw):
    """Replace inf in fit window list"""
    ret = []
    for i in fitw:
        if np.inf == i:
            ret.append((-1*np.inf, np.inf))
        else:
            ret.append(i)
    return ret

@PROFILE
def max_tmax(tot_new):
    """Find the maximum tmax for each tmin"""
    ret = {}
    for _, _, fitwin in tot_new:
        fitwin = fitwin[1]
        if fitwin[0] in ret:
            ret[fitwin[0]] = max(ret[fitwin[0]],
                                 fitwin[1])
        else:
            ret[fitwin[0]] = fitwin[1]
    return ret

def win_nan(fitwin):
    """Test if fit window is nan"""
    ret = False
    if 'nan' in str(fitwin):
        ret = True
    return ret


@PROFILE
def get_fitwindow(fit_range_arr, twin, prin=False):
    """Get fit window, print result"""
    tadd, tsub = twin
    tmin_allowed = global_tmin(fit_range_arr) + tadd
    tmax_allowed = global_tmax(fit_range_arr) + tsub
    ret = (tmin_allowed, tmax_allowed)
    assert ret[0] is not None, ret
    if prin:
        print("fit window:", ret)
    return ret

#def generate_continuous_windows(maxtmax, minsep=LENMIN-1):
@PROFILE
def generate_continuous_windows(maxtmax, minsep=LENMIN+1):
    """Generate the set of fit windows
    which is necessary for successful continuity"""
    ret = {}
    for tmin in maxtmax:
        ret[tmin] = set()
        numwin = maxtmax[tmin]-(tmin+minsep)+1
        numwin = int(numwin)
        for i in range(numwin):
            ret[tmin].add((tmin, tmin+minsep+i))
    return ret


@PROFILE
def lenfitw(fitwin):
    """Length of fit window"""
    return fitwin[1]-fitwin[0]+1
