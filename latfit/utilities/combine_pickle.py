#!/usr/bin/python3
"""Combine pickle files with the same structure.
Essentially list extend + i/o"""
import sys
import pickle
import numpy as np
import latfit.utilities.read_file as rf
import latfit.utilities.postfit.fitwin as hist

def lenfit(fname):
    """Find length of fit"""
    fitw = rf.pickle_fitwin(fname)
    return fitw[1]-fitw[0]+1
    
# p0, 32c, I2

FIT_SELECT = '[[6.0, 7.0, 8.0], [10.0, 11.0, 12.0], [6.0, 7.0, 8.0, 9.0, 10.0], [8.0, 9.0, 10.0]]' # 2

FIT_SELECT = '[[6.0, 7.0, 8.0], [6.0, 7.0, 8.0], [6.0, 8.0, 10.0], [6.0, 7.0, 8.0, 9.0, 10.0]]' # 3-e

FIT_SELECT = '[[6.0, 7.0, 8.0], [7.0, 9.0, 11.0], [8.0, 9.0, 10.0], [6.0, 7.0, 8.0, 9.0, 10.0]]' # 3-p

FIT_SELECT = '[[15.0, 16.0, 17.0], [16.0, 17.0, 18.0], [17.0, 18.0, 19.0], [14.0, 15.0, 16.0]]' # 0

FIT_SELECT = '[[6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0, 10.0, 11.0], [6.0, 8.0, 10.0], [8.0, 9.0, 10.0]]' # 1

# p1, 32c, I2
FIT_SELECT = '[[9.0, 10.0, 11.0, 12.0], [10.0, 11.0, 12.0], [9.0, 10.0, 11.0]]' # 0

FIT_SELECT = '[[5.0, 9.0, 13.0], [5.0, 6.0, 7.0, 8.0, 9.0], [4.0, 5.0, 6.0]]' # 1

FIT_SELECT = '[[5.0, 7.0, 9.0], [5.0, 6.0, 7.0], [4.0, 5.0, 6.0]]' # 2

# p11, 32c, I2
FIT_SELECT = '[[10.0, 11.0, 12.0, 13.0, 14.0], [11.0, 12.0, 13.0, 14.0, 15.0], [13.0, 14.0, 15.0], [10.0, 12.0, 14.0]]' # 0

FIT_SELECT = '[[9.0, 10.0, 11.0], [8.0, 9.0, 10.0], [8.0, 9.0, 10.0], [9.0, 10.0, 11.0]]' # 1

FIT_SELECT = '[[13.0, 14.0, 15.0], [11.0, 13.0, 15.0], [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [9.0, 10.0, 11.0]]' # 2e

FIT_SELECT = '[[12.0, 13.0, 14.0, 15.0], [13.0, 14.0, 15.0], [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], [10.0, 11.0, 12.0]]' # 2p

FIT_SELECT = '[[10.0, 11.0, 12.0], [11.0, 12.0, 13.0, 14.0, 15.0], [11.0, 13.0, 15.0], [9.0, 11.0, 13.0]]' # 3

FIT_SELECT = '[[10.0, 12.0, 14.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [9.0, 10.0, 11.0]]' # 0'

# p111, 32c, I2

FIT_SELECT = '[[8.0, 9.0, 10.0, 11.0], [9.0, 10.0, 11.0]]' # 0

FIT_SELECT = '[[8.0, 9.0, 10.0, 11.0], [6.0, 10.0, 14.0]]' # 1

# p0, 24c, I2

FIT_SELECT = '[[10.0, 12.0, 14.0], [10.0, 11.0, 12.0, 13.0], [10.0, 12.0, 14.0], [10.0, 11.0, 12.0, 13.0]]' # 0, 1, 3

FIT_SELECT = '[[10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0], [10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0]]' # 2

FIT_SELECT = ''

def main():
    """main"""
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
    for i in sys.argv[1:]:
        if '.cp' in i:
            continue
        if 'energy_min' in i:
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
        print(i, "add.shape", add.shape)
        if add.shape == (4,) and ('pvalue' not in i or 'err' not in i):
            rescount += len(add[3]) 
            rotate = True # top index is not fit ranges
            print(i, "shape:", add.shape)
            res_mean = add[0]
            err_check = add[1]
            assert len(res) == len(excl_arr)
            newfrs = add[3][:len(add[2])]
            if len(add[2]) > 1 or lenfit(i) == hist.LENMIN:
                count = 0
                for j in newfrs:
                    if len(j) > 1:
                        count += 1
                if count <= 1 and lenfit(i) != hist.LENMIN:
                    continue
                res.extend(add[2])
            else:
                continue
            if FIT_SELECT in str(newfrs) and FIT_SELECT:
                found.append(i)
                found_count += 1
                print("*****")
                print('file found:', i, "count:", found_count)
                print("*****")
            excl_arr.extend(newfrs)
            assert len(res) == len(excl_arr)
        else:
            rescount += len(add)
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
        print("found file", i, ":", item)
    if found:
        sys.exit()
    if rotate:
        res = np.array(res)
        print('res.shape:', res.shape)
        excl_arr = np.array(excl_arr)
        assert len(res) == len(excl_arr)
        ret = [res_mean, err_check, res, excl_arr]
    try:
        ret = np.array(ret)
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
    main()
