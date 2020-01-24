"""String processing"""
import re
import gvar
import numpy as np

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def tot_to_stat(res, sys_err):
    """Get result object which has separate stat
    and sys errors"""
    err = res.sdev
    assert err > sys_err, (err, sys_err)
    err = np.sqrt(err**2-sys_err**2)
    ret = gvar.gvar(res.val, err)
    return ret


def round_wrt(err1, err2):
    """Round err2 with respect to err1"""
    if err1 != np.inf:
        err1 = round_to_n(err1, 2)
        err1 = np.float(err1)
        if err1 == int(err1):
            err1 = int(err1)
        err1 = str(err1)
        if '.' in err1:
            assert 'e' not in err1, ("not supported:", err1)
            places = len(err1.split('.')[1])
        else:
            places = -1*trailing_zeros(err1)
        ret = round(err2, places)
    else:
        ret = err2
    return ret


@PROFILE
def errstr(res, sys_err):
    """Print error string"""
    if not np.isnan(res.val):
        assert res.sdev >= sys_err, (res.sdev, sys_err)
        newr = tot_to_stat(res, sys_err)
        if newr.sdev >= sys_err:
            ret = other_err_str(newr.val, newr.sdev, sys_err)
            ret = str(newr)+ret
        else:
            ret = other_err_str(newr.val, sys_err, newr.sdev)
            ret = swap_err_str(gvar.gvar(newr.val, sys_err), ret)
    else:
        ret = res
    return ret

@PROFILE
def swap_err_str(gvar1, errstr1):
    """Swap the gvar error string with the new one"""
    val, errs = str(gvar1).split("(") # sys err
    assert np.float(errs[:-1]) >= gvar.gvar('0'+errstr1).val, (gvar1, errstr1)
    ret = str(val)+errstr1+'('+errs
    return ret

@PROFILE
def place_diff_gvar(gvar1, gvar2):
    """Find difference in places between gvar1 and gvar2"""
    one = str(gvar1)
    two = str(gvar2)
    one = one.split('(')[0]
    two = two.split('(')[0]
    one = remove_period(one)
    two = remove_period(two)
    two = len(two)
    one = len(one)
    return two-one

@PROFILE
def remove_period(astr):
    """Remove decimal point"""
    return re.sub(r'.', '', astr)

@PROFILE
def other_err_str(val, err1, err2):
    """Get string for other error from a given gvar"""
    if isinstance(val, np.float) and not np.isnan(val) and not np.isnan(err2):
        err2 = round_wrt(err1, err2)
        err = gvar.gvar(val, err2)
        if not err2:
            places = np.inf
        else:
            places = place_diff_gvar(gvar.gvar(val, err1),
                                     gvar.gvar(val, err2))
        assert places >= 0, (val, err1, err2, places)
        try:
            ret = str(err).split('(')[1][:-1]
        except IndexError:
            print('va', val)
            print('er', err2)
            raise
        if places and err2:
            assert places == 1, (val, err1, err2)
            ret = '0'+ret[0]
        ret = '(' + ret + ')'
    else:
        ret = ''
    return ret


# https://stackoverflow.com/questions/8593355/in-python-how-do-i-count-the-trailing-zeros-in-a-string-or-integer
def trailing_zeros(longint):
    """Get number of trailing zeros in a number"""
    manipulandum = str(longint)
    return len(manipulandum)-len(manipulandum.rstrip('0'))

@PROFILE
def round_to_n(val, places):
    """Round to two sigfigs"""
    # from
    # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    if np.inf == val:
        val = np.nan
        plc = 1
    else:
        plc = -int(np.floor(np.log10(val))) + (places - 1)
    ret = round(val, plc)
    return ret