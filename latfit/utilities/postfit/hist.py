#!/usr/bin/python3
"""Make histograms from fit results over fit ranges"""
import sys
import subprocess
import pickle
import numpy as np
import gvar
from latfit.config import ISOSPIN, LATTICE_ENSEMBLE, IRREP
from latfit.utilities.postprod.h5jack import ENSEMBLE_DICT, check_ids
from latfit.utilities.postfit.gather_data import enph_filenames
from latfit.utilities.postfit.gather_data import make_hist
from latfit.utilities.postfit.fitwin import LENMIN
from latfit.utilities.postfit.compare_print import print_sep_errors
from latfit.utilities.postfit.compare_print import print_compiled_res
from latfit.utilities.postfit.bin_analysis import print_tot, fill_best
from latfit.utilities.postfit.bin_analysis import consis_tot, BinInconsistency
from latfit.utilities.postfit.strproc import tmin_param

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def geterr(allow):
    """Process SYS_ALLOWANCE"""
    ret = allow
    if allow is not None:
        ret = []
        for _, item in enumerate(allow):
            print(item)
            ret.append([gvar.gvar(item[j]).sdev for j,
                        _ in enumerate(item)])
        ret = np.array(ret)
    return ret

@PROFILE
def check_ids_hist():
    """Check the ensemble id file to be sure
    not to run processing parameters from a different ensemble"""
    ids_check = [ISOSPIN, LENMIN, IRREP]
    ids_check = np.asarray(ids_check)
    ids = pickle.load(open('ids_hist.p', "rb"))
    assert np.all(ids == ids_check),\
        "wrong ensemble. [ISOSPIN, LENMIN]"+\
        " should be:"+str(ids)+" instead of:"+str(ids_check)
    return ids

def get_mins(bests, files, twin, nosave):
    """Get the minimum error results for this bin"""
    ret = []
    sbreak = False
    for idx, (best, fil) in enumerate(zip(bests, files)):
        toapp = make_hist(fil, best, twin, nosave=nosave, allowidx=idx)
        if not toapp:
            toapp = []
            sbreak = True
        ret.append(toapp)
    assert len(ret) == 2
    if sbreak:
        assert not ret[0] or not ret[1]
    return ret

@PROFILE
def tloop(cbest, ignorable_windows, fnames, nosave=True):
    """Make the histograms."""
    if len(fnames) == 1 and (
            'phase_shift' in fnames[0] or\
            'energy' in fnames[1]) and nosave:
        # init variables
        allow_energy, allow_phase = fill_best(cbest)
        tot = []
        tot_pr = []
        success_tadd_tsub = []
        tsub = 0
        breakadd = False
        fname = fnames[0]
        print("CBEST =", cbest)

        # get file names
        energyfn, phasefn = enph_filenames(fname)

        # loop over fit windows
        tdis_max = ENSEMBLE_DICT[LATTICE_ENSEMBLE]['tdis_max']
        while np.abs(tsub) < tdis_max:
            tadd = 0
            while tadd < tdis_max:
                min_en, min_ph = get_mins(
                    (allow_energy, allow_phase),
                    (energyfn, phasefn),
                    (tadd, tsub), nosave)
                if min_en and min_ph: # check this
                    #binl.process_res_to_best(min_en, min_ph)
                    toapp, test, toapp_pr = print_compiled_res(
                        min_en, min_ph)
                    if test:
                        success_tadd_tsub.append((tadd, tsub))
                    else:
                        breakadd = not tadd
                        break
                    tot.append(toapp)
                    consis_tot(tot)
                    tot_pr.append(toapp_pr)
                else:
                    breakadd = not tadd
                    if breakadd:
                        if not min_en:
                            print("missing energies")
                        if not min_ph:
                            print("missing phases")
                    break
                tadd += 1
            if breakadd:
                print("stopping loop on tadd, tsub:",
                      tadd, tsub)
                break
            tsub -= 1
        print("Successful (tadd, tsub):")
        for i in success_tadd_tsub:
            print(i)
        subprocess.check_output(['notify-send', '-u',
                                 'critical', '-t', '30',
                                 'hist: tloop complete'])
        print_sep_errors(tot_pr)
        newcbest = print_tot(fname, tot, cbest, ignorable_windows)
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, '', (np.nan, np.nan), nosave=nosave)
        print("minimized error results:", min_res)
        newcbest = []
    return newcbest


# don't count these fit windows for continuity check
# they are likely overfit,
# (in the case of an overfit cut : a demand that chi^2/dof >= 1)
# or failing for another acceptable/known reason
# so the data will necessarily be missing

def augment_overfit(wins):
    """Augment likely overfit fit window
    list with all subset windows"""
    wins = set(wins)
    for win in sorted(list(wins)):
        tmin = win[0]
        tmax = win[1]
        for i in range(tmax-tmin-LENMIN):
            wins.add((tmin+i+1, tmax))
    for win in sorted(list(wins)):
        tmin = win[0]
        tmax = win[1]
        for i in range(tmax-tmin-LENMIN):
            wins.add((tmin, tmax-i-1))
    return sorted(list(wins))


def next_filename(fnames, success=False, curr=None):
    """Find the next file name, binary search style"""
    direc = 'forward' if not success else 'backward'
    if curr is not None:
        cidx = fnames.index(curr)
        if direc == 'forward':
            fnames = fnames[cidx+1:]
        else:
            #print('backward', success)
            #print('curr', curr, 'fnames', fnames, 'cidx', cidx)
            fnames = [fnames[:cidx][-1]]
    lfnam = len(fnames)
    if lfnam == 1:
        ret = fnames[0]
    else:
        idx = np.ceil(lfnam/2)
        if idx == lfnam/2:
            idx +=1
        idx -= 1
        assert idx < lfnam
        idx = int(idx)
        ret = fnames[idx]
    return ret

def sort_filenames(fnames):
    """Sort file names by tmin_param"""
    sdict = {}
    ret = []
    for i in fnames:
        tmin_p = tmin_param(i)
        sdict[tmin_p] = i
        keys = sorted(list(sdict))
    for i in keys:
        ret.append(sdict[i])
    return ret

def wallback():
    """At late times, walk the plateau backwards to find optimal tmin"""
    # p11 32c, hard coded
    if LATTICE_ENSEMBLE == '32c' and IRREP == 'A1_mom11':
        ignorable_windows = [(9, 13), (10, 14)]
    else:
        ignorable_windows = []

    fnames = sys.argv[1:]
    fnames = sort_filenames(fnames)
    assert len(fnames) == len(sys.argv[1:]), (fnames, sys.argv[1:])
    print("files used:", fnames)

    # the assumption here is that all sub-windows in these windows
    # have also been checked and are also (likely) overfit
    ignorable_windows = augment_overfit(ignorable_windows)
    print("ignorable_windows =", ignorable_windows)

    cbest = []
    flag = 1
    fname = next_filename(fnames)
    useable = ()
    while flag:
        try:
            cbest = tloop(cbest, ignorable_windows, [fname])
            success = True
            useable = (cbest, ignorable_windows, [fname])
            if flag == 1: # now start the walk back
                flag = 2
        except BinInconsistency:
            success = False
            if flag == 2: # walk back ends
                break
        curr = fname
        fname = next_filename(fnames, curr=curr, success=success)
        if curr == fname:
            print("fixed point found:", fname)
            break
    print("CBEST, final =", cbest)
    if useable: # to get final plot info, rerun
        tloop(*useable)


if __name__ == '__main__':
    try:
        open('ids_hist.p', 'rb')
    except FileNotFoundError:
        print("be sure to set ISOSPIN, LENMIN",
              "correctly")
        print("edit histogram.py to remove the hold then rerun")
        # the hold
        # sys.exit(1)
        IDS_HIST = [ISOSPIN, LENMIN, IRREP]
        IDS_HIST = np.asarray(IDS_HIST)
        pickle.dump(IDS_HIST, open('ids_hist.p', "wb"))
    check_ids_hist()
    check_ids(LATTICE_ENSEMBLE)
    wallback()
