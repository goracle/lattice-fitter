#!/usr/bin/python3
"""Combine pickle files with the same structure.
Essentially list extend + i/o"""
import sys
import pickle
import glob
import numpy as np
import latfit.utilities.read_file as rf
import latfit.utilities.postfit.fitwin as hist
from latfit.include import INCLUDE
from latfit.utilities.tuplize import list_mat


def lenfit(fname):
    """Find length of fit"""
    fitw = rf.pickle_fitwin(fname)
    return fitw[1]-fitw[0]+1

def singleton_cut(add, res, newfrs, tochk):
    """Obsolete cut for single results
    (new way is fit window min length)
    """
    ret = False
    # check to see if we have expected result amount
    if len(add[2]) > 1 or lenfit(tochk) == hist.LENMIN:
        count = 0
        for j in newfrs:
            if len(j) > 1: # not an eff mass pt
                count += 1
        if count <= 1 and lenfit(tochk) != hist.LENMIN:
            ret = True
        # res.extend(add[2])
    else:
        ret = True # cut on singletons from large windows
    return ret

def dummy_glob():
    """Get list of files to use to find results
    for a selected fit range"""
    start = glob.glob('energy*.p')
    ret = []
    for i in start:
        if 'err' in i:
            continue
        ret.append(i)
    return ret

def main(fit_select=''):
    """main"""
    fit_select = str(fit_select)
    verb = False if fit_select else True
    ret = []
    shape = ()
    res_mean = None
    err_check = None
    res = []
    excl_arr = []
    rotate = False
    early = np.inf
    earlylist = []
    rescount = 0
    useset = set()
    found_count = 0
    found = []
    file_list = sys.argv[1:] if not fit_select else dummy_glob()
    if fit_select and verb:
        print("fit select:", fit_select)
    for i in file_list:
        if '.cp' in i:
            continue
        if 'energy_min' in i:
            continue
        if 'mindim' in i:
            continue
        if 'badfit' in i:
            continue
        try:
            new_early = rf.earliest_time(i)
        except ValueError:
            continue
        useset.add(i)
        early = min(early, new_early)
        if early == new_early:
            earlylist.append(i)
        assert '.p' in i, str(i)
        assert '.pdf' not in i, str(i)
        add = pickle.load(open(str(i), "rb"))
        if verb:
            print(i, "add.shape", add.shape)
        cond = add.shape == (4,) and ('pvalue' not in i or 'err' not in i)
        if rotate:
            assert cond
        if cond:
            rescount += len(add[3]) # number of fit ranges gives number of results
            rotate = True # top index is not fit ranges
            if verb:
                print(i, "shape:", add.shape)
            res_mean = add[0]
            err_check = add[1]
            try:
                assert len(add[2]) == len(add[3]),\
                    (len(add[2]), len(add[3]))
                newfrs = add[3]
            except AssertionError:
                #print(add[2])
                #print(add[3])
                newfrs = add[3][:len(add[2])]

            # check fit range length
            #assert np.all([len(j) >= hist.LENMIN for j in newfrs]),\
                #([len(j) >= hist.LENMIN for j in newfrs])


            # perform check before throwing away effective mass points
            #effmasspts = add[3][len(add[2]):]
            #assert np.all([len(j) == 1 for j in effmasspts])

            if singleton_cut(add, res, newfrs, i):
                pass
                #continue
            if fit_select in str(newfrs) and fit_select:
                found.append(i)
                found_count += 1
                if verb:
                    print("*****")
                    print('file found:', i, "count:",
                          found_count)
                    print("*****")
            res.extend(add[2])
            excl_arr.extend(newfrs)
            assert len(res) == len(excl_arr), \
                (i, len(res), len(excl_arr))
        else:
            rescount += len(add)
            if verb:
                print(i, "shape:", add.shape)
        if not shape:
            shape = np.asarray(add[0]).shape
        else:
            try:
                assert list(shape) == list(np.asarray(add[0]).shape)
            except AssertionError:
                print("shape mismatch of input pickle files")
                print("check shape:", shape)
                print("shape of", i, ":", np.asarray(add[0]).shape)
                raise
        if not rotate:
            ret.extend(add)
    outfn = start_str(sorted(list(useset)))
    for i, item in enumerate(found):
        if verb:
            print("found file", i, ":", item)
    if fit_select:
        ret = parse_found_for_dts(found)
    else:
        if rotate:
            res = np.array(res)
            print('res.shape:', res.shape)
            excl_arr = np.array(excl_arr)
            assert len(res) == len(excl_arr)
            ret = [res_mean, err_check, res, excl_arr]
        try:
            ret = np.array(ret)
            assert ret.shape[0], ret.shape
            print("final shape:", ret.shape)
            print("final rescount:", rescount)
        except ValueError:
            pass
        #print("final shape:", ret.shape)
        print("finished combining:", sorted(list(useset)))
        if outfn[-1] != '_':
            outfn = outfn + '_tmin' + str(int(early)) + '.cp'
        else:
            outfn = outfn + 'tmin' + str(int(early)) + '.cp'
        print("writing results into file:", outfn)
        earlylist = prune_earlylist(earlylist)
        print("earliest time:", early, "from:")
        for i in earlylist:
            print('prl.sh', i, 'tocut/')
        if '.p.p' not in outfn:
            pickle.dump(ret, open(outfn, "wb"))
    return ret

def parse_found_for_dts(found):
    """Find t-t0 and mat dt from found"""
    ret = ()
    if found:
        #print("found fit range")
        earliest = None
        for i, item in enumerate(found):
            # assumes tminus is 1 digit long
            if earliest is None:
                item = earliest
            if rf.earliest_time(item) < rf.earliest_time(earliest):
                earliest = item
            tminus = earliest.split('TMINUS')[1][0]
            tminus = 'TMINUS'+tminus
            dt2 = None
            if 'dt' in item:
                dt2 = earliest.split('dt')[1][0]
        #print('T0 =', tminus)
        #print('DELTA_T_MATRIX_SUBTRACTION =', dt2)
        ret = (tminus, dt2)
    return ret

def prune_earlylist(earlylist):
    """Find set of file names with the earliest time"""
    early = np.inf
    ret = []
    for i in earlylist:
        early = min(early, rf.earliest_time(i))
    for i in earlylist:
        if rf.earliest_time(i) == early:
            spl = None
            for j in range(3):
                spl = i.split('I'+str(j))
                if len(spl) > 1:
                    break
            add = '*'+spl[-1]
            ret.append(add)
    return ret


def start_str(strlist):
    """Get the common starting string from a list of strings"""
    ret = ""
    for i in strlist:
        for j in strlist:
            if not ret:
                ret = common_start(i, j)
                break
            assert ret, "blank file name in given string list:" + \
                str(strlist)
        ret = common_start(i, ret)
    return ret

# from
# https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
def common_start(stra, strb):
    """ returns the longest common substring from the
    beginning of sa and sb """
    def _iter():
        for astr, bstr in zip(stra, strb):
            if astr == bstr:
                yield astr
            else:
                return

    return ''.join(_iter())




if __name__ == '__main__':
    if INCLUDE:
        FIT_SELECT = str(list_mat(INCLUDE))
    elif FIT_SELECT:
        pass
    else:
        FIT_SELECT = ''
    main(fit_select=FIT_SELECT)
