#!/usr/bin/python3
"""Make histograms from fit results over fit ranges"""
import sys
import os
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
from latfit.utilities.postfit.dropdown import weight_sum

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
def tloop(cbest, ignorable_windows, fnames, dump_min=False, nosave=True):
    """Make the histograms."""
    cbest, cbest_blks = cbest
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
                    toapp, test, toapp_pr = print_compiled_res(
                        cbest, min_en, min_ph)
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
        if success_tadd_tsub:
            print("Successful (tadd, tsub):")
            for i in success_tadd_tsub:
                print(i)
            print_sep_errors(tot_pr)
            newcbest, newcbest_blks = print_tot(
                fname, tot, (cbest, cbest_blks), ignorable_windows, dump_min)
        else:
            print("raising inconsistency since no consistent results found")
            raise BinInconsistency
            newcbest = cbest
            newcbest_blks = cbest_blks
        print("end of tloop")
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, '', (np.nan, np.nan), nosave=nosave)
        print("minimized error results:", min_res)
        newcbest = []
    return newcbest, newcbest_blks


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
            if cidx:
                fnames = [fnames[:cidx][-1]]
            else:
                fnames = []
    lfnam = len(fnames)
    if lfnam == 1:
        ret = fnames[0]
    elif not lfnam:
        ret = ""
    else:
        idx = np.ceil(lfnam/2)
        if idx == lfnam/2:
            idx += 1
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

def walkback():
    """At late times, walk the plateau backwards to find optimal tmin"""
    # p11 32c, hard coded
    ignorable_windows = []
    if LATTICE_ENSEMBLE == '32c' and IRREP == 'A1_mom11':
        pass
        #ignorable_windows = [(9, 13), (10, 14)]

    fnames = sys.argv[1:]
    fnames = sort_filenames(fnames)
    assert len(fnames) == len(sys.argv[1:]), (fnames, sys.argv[1:])
    print("files used:", fnames)

    # the assumption here is that all sub-windows in these windows
    # have also been checked and are also (likely) overfit
    ignorable_windows = augment_overfit(ignorable_windows)
    print("ignorable_windows =", ignorable_windows)

    cbest = []
    cbest_blks = []
    flag = 1
    fname = next_filename(fnames)
    useable = ()
    route = []
    print("files used:", fnames)
    while flag:
        route.append(tmin_param(fname))
        try:
            if flag != 2:
                check_bad_bin(tmin_param(fname))
            print("starting analysis on file:", fname)
            cbest, cbest_blks = tloop(
                (cbest, cbest_blks), ignorable_windows, [fname],
                dump_min=False)
            success = True
            print("success found for file:", fname)
            useable = ((cbest, cbest_blks), ignorable_windows, [fname])
            if flag == 1: # now start the walk back
                print("starting walk-back")
                flag = 2
        except BinInconsistency:
            if flag != 2:
                create_skip_file(tmin_param(fname))
            success = False
            if flag == 2: # walk back ends
                print("bin inconsistency in wall-back found; exiting")
                break
        curr = fname
        print("route so far:", route)
        fname = next_filename(fnames, curr=curr, success=success)
        if curr == fname:
            print("fixed point found:", fname)
            break
        if not fname:
            print("no more data to examine")
            break
    if useable: # to get final plot info, rerun
        tloop(*useable, dump_min=True)
    print("CBEST, final =", cbest)
    print("time route taken:", route)
    print("starting drop down version")
    drop_down(cbest_blks)
    subprocess.check_output(['notify-send', '-u',
                             'critical', '-t', '30',
                             'hist: walk-back complete'])

def drop_down(cbest_blks):
    """Perform drop down sum"""
    diml = len(cbest_blks[0])
    cbest = []
    for dim in range(diml):
        toapp = []
        acc_en = build_weightsum_list(cbest_blks, dim, 0)
        mean, err = weight_sum(np.array(acc_en))
        toapp.append(str(gvar.gvar(mean, err)))
        acc_ph = build_weightsum_list(cbest_blks, dim, 1)
        mean, err = weight_sum(np.array(acc_ph))
        toapp.append(str(gvar.gvar(mean, err)))
        cbest.append(toapp)
    print('drop down cbest:')
    print(cbest)

def build_weightsum_list(cbest_blks, dim, itemidx):
    """Ok here we are"""
    ret = []
    for i in cbest_blks:
        if not list(i[dim][itemidx]):
            #print("item not found", dim, itemidx)
            #print(i[dim][itemidx])
            #sys.exit()
            continue
        if not np.isnan(i[dim][itemidx][0]):
            toapp = i[dim][itemidx]
            for j in ret:
                if np.allclose(toapp, j, rtol=1e-12):
                    toapp = None
                    break
            if toapp is not None:
                ret.append(toapp)
        else:
            break # do not continue; we have no result at any earlier times
    if len(ret) > 1:
        for idx, res1 in enumerate(ret):
            for jdx, res2 in enumerate(ret):
                if jdx <= idx:
                    continue
                assert not np.allclose(res1, res2, rtol=1e-12)
    return ret


def check_bad_bin(tmin):
    """Check to make sure this tmin_param
    is not already known to be inconsistent"""
    tochk = 'badbin_tmin'+str(tmin)+'.cp'
    if os.path.exists(tochk):
        raise BinInconsistency

def create_skip_file(tmin):
    """Create a dummy file so we skip bin analysis on this
    tmin_param in subsequent runs"""
    tosave = 'badbin_tmin'+str(tmin)+'.cp'
    dummy = []
    fn1 = open(tosave, 'wb')
    print("saving skip file:", tosave)
    pickle.dump(dummy, fn1)
    

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
    walkback()
