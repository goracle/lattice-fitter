#!/usr/bin/python3
"""apply walkback procedure to user inputted lists"""
import sys
from collections import defaultdict
import os
import ast
import numpy as np
import gvar

ISOSPIN = 0

def proc_walkres_input():
    """process/sanitize input from walkres.sh script"""
    args = sys.argv[1:]
    ret = []
    temp = ""
    mdim = np.inf
    for i in args:
        if i != '@':
            temp += i
        else:
            toapp = ast.literal_eval(temp)
            toapp = filter_none(toapp)
            mdim = min(len(toapp), mdim)
            ret.append(toapp)
            temp = ""
    ret2 = []
    for app in ret:
        if len(app) > mdim:
            app = app[:mdim]
        ret2.append(app)
    return ret2

def filter_none(toapp):
    """get rid of everything beyond None in list"""
    ret = []
    for idx, item in enumerate(toapp):
        it1, it2 = item
        if idx or it1 == 'None': # p0 grd phase shift exception
            if it1 == 'None' or it2 == 'None':
                break
        ret.append(item)
    return ret


def gdoublestr(dstr):
    """convert string to gvar obj"""
    if str(dstr) == 'None':
        ret = None
    else:
        try:
            ret = gvar.gvar(dstr)
        except ValueError:
            print('dstr', dstr)
            raise
    return ret

def readin():
    """Process user input"""
    enlist = defaultdict(list)
    phlist = defaultdict(list)
    for nlist in proc_walkres_input():
        for i, (en1, ph1) in enumerate(nlist):
            en1 = gdoublestr(en1)
            ph1 = gdoublestr(ph1)
            enlist[i].append(en1)
            phlist[i].append(ph1)
    return enlist, phlist


def consis(it1, it2):
    """Check for consistency"""
    assert it1.sdev > 0, it1
    assert it2.sdev > 0, it2
    merr = max(it1.sdev, it2.sdev)
    diff = np.abs(it1.val-it2.val)
    ret = True if diff <= merr * 1.5 else False
    return ret

def consis_list(it1, clist):
    """Check it1 for consistency with list of items"""
    return np.all([consis(it1, item) for item in clist])

def walkback(eplist, reverse=False):
    """perform the (directed) walkback procedure on the phases"""
    ret = {}
    for lvl in eplist:
        # sort such that 'None' phase shifts appear first
        slist = sorted(eplist[lvl], reverse=reverse,
                       key=lambda x: (x is None, x))
        # we start walk-back with first list item, then walk
        # towards higher indices
        # direction of walk determined by
        # list sort param 'reverse'
        mmin = slist[0]
        clist = [mmin]
        for it1 in slist:
            if mmin is None:
                continue
            if mmin.sdev > it1.sdev and consis_list(it1, clist):
                mmin = it1
                clist.append(mmin)
        ret[lvl] = mmin
    return ret

def rezip(enlist, phlist):
    """Recombine energies and phase shifts"""
    ret = []
    for key in enlist:
        ret.append([str(enlist[key]), str(phlist[key])])
    return ret


def main():
    """apply walkback procedure to user inputted lists"""
    enlist, phlist = readin()
    enlist = walkback(enlist, reverse=False)
    reverse = True if ISOSPIN == 2 or not ISOSPIN else False
    phlist = walkback(phlist, reverse=reverse)
    ret = rezip(enlist, phlist)
    print(equals_str(), ret)

def equals_str():
    """Get string like 'p0_lat =' or 'p0_32c_lat =' from cwd"""
    arg = os.getcwd()
    mom = arg.split('p')[-1].split('/')[0]
    ret = 'p'+mom+'_'
    if '32c' in arg:
        ret += '32c_'
    ret += 'lat ='
    return ret

if __name__ == '__main__':
    main()
