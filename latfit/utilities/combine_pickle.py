#!/usr/bin/python3
"""Combine pickle files with the same structure.
Essentially list extend + i/o"""
import sys
import pickle
import numpy as np

def main():
    """main"""
    ret = []
    outfn = start_str(sys.argv[1:])
    outfn = outfn + '.p'
    shape = ()
    for i in sys.argv[1:]:
        assert '.p' in i, str(i)
        assert '.pdf' not in i, str(i)
        add = pickle.load(open(str(i), "rb"))
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
        ret.extend(add)
    ret = np.array(ret)
    print("final shape:", ret.shape)
    print("finished combining:", sys.argv[1:])
    print("writing results into file:", outfn)
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
