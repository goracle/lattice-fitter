#!/usr/bin/python3
"""destructively bin sets of A2A hits"""

import sys
import os
import stat
import re
import glob
import pickle
import numpy as np
import h5py
import latfit.utilities.read_file as rf

SKIP_VEC = True
SKIP_VEC = False
SKIP_SCALAR = False
SKIP_SCALAR = True

def h5list(dd1):
    """List hdf5 files in given directory
    (dd1)
    """
    dd1 = str(dd1)
    ret = glob.glob(dd1+'/*.hdf5')
    #ret = [i for i in ret if 'exact' not in i]
    #ret = [i.split('/')[-1] for i in ret]
    return ret

def datasets(fs1):
    """Get dataset list"""
    ret = set()
    compare = set()
    for i, item in enumerate(fs1):
        fn1 = h5py.File(item, 'r')
        for name in fn1:
            if not i:
                ret.add(rf.basename(name))
            else:
                compare.add(rf.basename(name))
            ret.intersection(compare)
            compare = set()
        fn1.close()
    return sorted(list(ret))

def ishitsdir(item):
    """
    Examine to see which is the directory
    with the extra hit.  If the first item ends in 1,
    then it comes from the extra hit directory.
    """
    item = item.split('/')[-1]
    ret = None
    if '1.hdf5' in item:
        ret = True
    elif '0.hdf5' in item:
        ret = False
    return ret

def trajq(fil):
    """Get traj info"""
    ret = re.sub('_exact', '', fil)
    ret = ret.split('/')[-1]
    ret = ret.split('_')[-1]
    ret = ret.split('.')[0]
    ret = int(ret)
    return ret

def proc_hits_dirs():
    """process the directory structure"""
    dir1, dir2 = sys.argv[1], sys.argv[2]
    fs1 = h5list(dir1)
    fs2 = h5list(dir2)
    if ishitsdir(fs1[0]):
        print("assuming hits dir is", dir1)
        print("assuming base dir is", dir2)
        # hdir = dir1
        hlist = fs1
        flist = fs2
        fdir = dir2
    elif ishitsdir(fs2[0]):
        print("assuming hits dir is", dir2)
        print("assuming base dir is", dir1)
        # hdir = dir2
        hlist = fs2
        flist = fs1
        fdir = dir1
    else:
        print("no hits directory found")
        sys.exit(1)
    return hlist, flist, fdir

def main():
    """main"""
    hlist, flist, fdir = proc_hits_dirs()
     # get the dataset names from the hits directory
    dsets = datasets(hlist)
    # get the trajectories from the non-hits directory
    for fil in sorted(list(flist)):
        # we skip hits directories which have no base trajectory
        # (base meaning trajectory number matches noise seed)
        if fil.endswith('1.hdf5'):
            continue
        # get the corresponding hit file
        filh = h5hitget(fil, hlist)
        try:
            traj = trajq(fil)
        except TypeError:
            print("type error:", fil.split('/')[-1], rf.traj(fil))
            print("trajectory information not found")
            sys.exit(1)
        print("currently on traj =", traj)

        # skip if we've already
        # (even partially) overwritten the original file
        block = mark(traj, fdir, exact=('exact' in filh))
        if block:
            continue

        # bin
        for sname in dsets:
            if SKIP_VEC:
                if rf.vecp(sname) or 'vecCheck' in sname:
                    continue
            if SKIP_SCALAR:
                if not (rf.vecp(sname) or 'vecCheck' in sname):
                    continue
            print("processing dset:", sname)
            nam1 = 'traj_'+str(traj)+'_'+sname
            nam2 = 'traj_'+str(traj+1)+'_'+sname
            try:
                blk = getblock(nam1, fil)
                blkh = getblock(nam2, filh)
            except KeyError:
                print("block not found for:", fil, filh)
                print("continuing")
                continue
            assert blk.shape == blkh.shape, "shape mismatch"

            # actual bin of the two runs
            nblk = (blk+blkh)/2.0

            # overwrite step, do not turn on unless checked!
            overwrite(nam1, fil, nblk)
            # started = True

def mark(traj, fdir, exact=False):
    """Mark the base trajectory data as modified"""
    ids = np.array(1)
    if not exact:
        name = str(fdir)+'/'+'traj_'+str(traj)+'_ismodified.p'
    else:
        name = str(fdir)+'/'+'traj_'+str(traj)+'_exact_ismodified.p'
    if os.path.isfile(name):
        ret = True
    else:
        ret = False
        pickle.dump(ids, open(name, "wb"))
        remove_write_permissions(name)
    return ret

# taken from
# https://stackoverflow.com/questions/16249440/
# changing-file-permission-in-python/38511116
def remove_write_permissions(path):
    """Remove write permissions from this path,
    while keeping all other permissions intact.

    Params:
        path:  The path whose permissions to alter.
    """
    no_user_writing = ~stat.S_IWUSR
    no_group_writing = ~stat.S_IWGRP
    no_other_writing = ~stat.S_IWOTH
    no_writing = no_user_writing & no_group_writing & no_other_writing

    current_permissions = stat.S_IMODE(os.lstat(path).st_mode)
    os.chmod(path, current_permissions & no_writing)

def overwrite(nam1, fil, nblk):
    """Overwrite the original trajectory with the new, binned block"""
    fn1 = h5py.File(fil, 'r+')
    assert nam1 in fn1, "dataset:"+str(nam1)+"does not exist in:"+str(fil)
    backup = np.copy(np.array(fn1[nam1]))
    assert fn1[nam1].shape == nblk.shape
    fn1[nam1][:] = nblk
    assert np.allclose(fn1[nam1], nblk, rtol=1e-16), "write check failed"
    assert not np.allclose(fn1[nam1], backup, rtol=1e-16),\
        "write check 2 failed"
    # fn1[nam1][:] = backup
    # assert np.allclose(
    # fn1[nam1], backup, rtol=1e-16), "write check 3 failed"
    fn1.close()


def getblock(dname, fname):
    """Get hdf5 block given dataset name and file name"""
    fn1 = h5py.File(fname, 'r')
    blk = np.asarray(fn1[dname])
    fn1.close()
    return blk

def h5hitget(fil, hlist):
    """Get the corresponding hit hdf5 file"""
    if 'exact' in fil:
        key = re.sub('0_exact.hdf5', '1_exact.hdf5', fil)
    else:
        key = re.sub('0.hdf5', '', fil)
    key = key.split('/')[-1]
    ret = []
    found = False
    for item in hlist:
        if 'exact' not in key and 'exact' in item:
            continue
        if key in item:
            if trajq(fil)+1 != trajq(item):
                continue
            found = True
            ret.append(item)
    assert found,\
        "corresponding hits file not found for:"+str(fil)+" "+str(key)
    assert len(ret) == 1,\
        "Wrong number of (hits) files found for key:"+str(key)+""+str(ret)
    if 'exact' in fil:
        print("exact match:", fil, ret[0])
    return ret[0]





if __name__ == '__main__':
    main()
