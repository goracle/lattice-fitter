"""Look at files in directory to find which fit windows still need to be fit."""

import glob
from latfit.config import GEVP, MOMSTR, SYS_ENERGY_GUESS, MATRIX_SUBTRACTION
from latfit.config import ISOSPIN
import latfit.config

def filename_plus_config_info(meta, filename):
    """Create file name for fit results"""
    ret = filename_info(meta.random_fit, filename)
    ret += meta.window_str()+"_"+latfit.config.T0
    return ret

def filename_info(randfit, filename):
    """Add config info to file name"""
    if GEVP:
        filename += "_"+MOMSTR+'_I'+str(ISOSPIN)
    if randfit:
        filename += '_randfit'
    if SYS_ENERGY_GUESS:
        filename += "_sys"
    if MATRIX_SUBTRACTION:
        filename += '_dt'+str(latfit.config.DELTA_T_MATRIX_SUBTRACTION)
    return filename

def list_p_files():
    """List of all pickle files in current directory"""
    start = glob.glob('*pvalue*.p')
    ret = []
    for i in start:
        if 'err' in i:
            continue
        ret.append(i)
    return ret

def filter_pfiles(name='pvalue'):
    """Filter out filenames that do not match the current config"""
    lfn = list_p_files()
    ret = []
    test1 = filename_info(True, name)
    test2 = filename_info(False, name)
    for i in lfn:
        if test1 not in i and test2 not in i:
            continue
        ret.append(i)
    return ret

def finished_windows():
    """Find the list of windows we already have results for"""
    name = 'pvalue'
    return checked_windows(name)

def inconsistent_windows():
    """Find the inconsistent windows"""
    name = 'badfit'
    return checked_windows(name)

def checked_windows(name):
    """Find finished windows with key = name"""
    lpn = filter_pfiles(name=name)
    ret = set()
    test1 = filename_info(True, name)
    test2 = filename_info(False, name)
    for i in lpn:
        if test1 in i:
            fin = i.split(test1)[1].split('_')[1:3]
        else:
            assert test2 in i, (test2, i)
            fin = i.split(test2)[1].split('_')[1:3]
        win = tuple((float(fin[0]), float(fin[1])))
        ret.add(win)
    ret = sorted(list(ret))
    return ret
