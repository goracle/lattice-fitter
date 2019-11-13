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
    outfn = outfn + '.p'
    shape = ()
    res_mean = None
    err_check = None
    res = []
    excl_arr = []
    rotate = False
    early = np.inf
    earlyfn = None
    for i in sys.argv[1:]:
        if '_.p' in i:
            continue
        new_early = rf.earliest_time(i)
        early = min(early, new_early)
        if early == new_early:
            earlyfn = i
        assert '.p' in i, str(i)
        assert '.pdf' not in i, str(i)
        add = pickle.load(open(str(i), "rb"))
        if add.shape == (4,):
            rotate = True # top index is not fit ranges
            print(i, "shape:", add.shape)
            res_mean = add[0]
            err_check = add[1]
            res.extend(add[2])
            excl_arr.extend(add[3][:len(add[2])])
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
        excl_arr = np.array(excl_arr)
        ret = [res_mean, err_check, res, excl_arr]
    ret = np.array(ret)
    #print("final shape:", ret.shape)
    print("finished combining:", sys.argv[1:])
    print("writing results into file:", outfn)
    print("earliest time:", early, "from", earlyfn)
    pickle.dump(ret, open(outfn, "wb"))
    print("done.")


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
