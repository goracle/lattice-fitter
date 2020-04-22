"""String processing"""
import re
import gvar
import numpy as np
from latfit.jackknife_fit import jack_mean_err
from latfit.config import ISOSPIN, LATTICE_ENSEMBLE, IRREP

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
    if res is None:
        ret = None
    else:
        if isinstance(res, str):
            res = gvar.gvar(res)
        if isinstance(sys_err, str):
            sys_err = float(sys_err)
        err = res.sdev
        if not np.isnan(sys_err):
            assert err > sys_err, (err, sys_err)
        err = np.sqrt(err**2-sys_err**2)
        ret = gvar.gvar(res.val, err)
    return ret

def stat_from_blocks(res, blocks):
    """Calculate stat error again from jackknife blocks"""
    if res is None:
        ret = None
    else:
        _, err = jack_mean_err(blocks)
        if isinstance(res, str):
            res = gvar.gvar(res)
        assert not np.isnan(err)
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

def nextchars_nums(base, after):
    """Find the next character in the base string
    after the after string if they are numbers
    """
    nums = [str(i) for i in range(10)]
    base = str(base)
    after = str(after)
    if after not in base:
        ret = ""
    else:
        ret = base.split(after)[1:][0]
        ret2 = ''
        for i in str(ret):
            if i not in nums:
                break
            ret2 += i
    return ret2

def tmin_param(fname):
    """Find tmin param from filename: fname"""
    ret = nextchars_nums(fname, 'tmin')
    ret = int(ret)
    return ret

def min_fit_file(dim, rest):
    """Get file name of file to
    dump optimal fit params"""
    saven = 'fit_'+str(LATTICE_ENSEMBLE)+'_I'+str(
        ISOSPIN)+'_'+str(IRREP)+'_dim'+str(
            dim)+'_param_'+str(rest)
    saven = re.sub(' ', '_', saven)
    return saven
