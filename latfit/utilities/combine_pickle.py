#!/usr/bin/python3
"""Combine pickle files with the same structure.
Essentially list extend + i/o"""
import sys
import pickle
import numpy as np
import latfit.utilities.read_file as rf

def main():
    """main"""
    ret = []
    outfn = start_str(sys.argv[1:])
    shape = ()
    res_mean = None
    err_check = None
    res = []
    excl_arr = []
    rotate = False
    early = np.inf
    earlylist = []
    for i in sys.argv[1:]:
        if '.cp' in i:
            continue
        try:
            new_early = rf.earliest_time(i)
        except ValueError:
            continue
        early = min(early, new_early)
        if early == new_early:
            earlylist.append(i)
        assert '.p' in i, str(i)
        assert '.pdf' not in i, str(i)
        add = pickle.load(open(str(i), "rb"))
        if add.shape == (4,) and ('pvalue' not in i or 'err' not in i):
            rotate = True # top index is not fit ranges
            print(i, "shape:", add.shape)
            res_mean = add[0]
            err_check = add[1]
            assert len(res) == len(excl_arr)
            newfrs = add[3][:len(add[2])]
            if len(add[2]) > 1:
                count = 0
                for j in newfrs:
                    if len(j) > 1:
                        count += 1
                if count <= 1:
                    continue
                res.extend(add[2])
            else:
                continue
            excl_arr.extend(newfrs)
            assert len(res) == len(excl_arr)
        else:
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
    if rotate:
        res = np.array(res)
        print('res.shape:', res.shape)
        excl_arr = np.array(excl_arr)
        assert len(res) == len(excl_arr)
        ret = [res_mean, err_check, res, excl_arr]
    try:
        ret = np.array(ret)
        print("final shape:", ret.shape)
    except ValueError:
        pass
    #print("final shape:", ret.shape)
    print("finished combining:", sys.argv[1:])
    if outfn[-1] != '_':
        outfn = outfn + '_tmin' + str(int(early)) + '.cp'
    else:
        outfn = outfn + 'tmin' + str(int(early)) + '.cp'
    print("writing results into file:", outfn)
    earlylist = prune_earlylist(earlylist)
    print("earliest time:", early, "from:")
    for i in earlylist:
        print('mv', i, 'tocut/')
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
            else:
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
        for a, b in zip(stra, strb):
            if a == b:
                yield a
            else:
                return

    return ''.join(_iter())

    


if __name__ == '__main__':
    main()
