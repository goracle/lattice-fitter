"""Connected diagram part of h5jack"""
from mpi4py import MPI
import numpy as np
import h5py
from latfit.utilities import exactmean as em
from latfit.utilities.postprod.checkblks import convert_traj, printblk
from latfit.utilities.postprod.checkblks import check_key
from latfit.utilities.postprod.checkblks import TESTKEY2, TEST44

#### DO NOT MODIFY THE BELOW

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

# set by h5jack
TSEP = np.nan
LT = np.nan
TSTEP = np.nan
ROWS = None
COLS = None
ROWST = np.nan
PREFIX = None
EXTENSION = None
WRITE_INDIVIDUAL = None
TDIS_MAX = np.nan
AVGTSRC = np.nan
KK_OP = None
KCORR = None

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


def fold_time(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

def dojackknife(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

# broadly useful utility functions
def get_file_name(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

def getwork(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

def getindices(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

def tdismax():
    """Return tdis max"""
    if WRITE_INDIVIDUAL:
        ret = LT-1
    else:
        ret = TDIS_MAX
    return ret



@PROFILE
def getgenconblk(base, trajl, avgtsrc=False, rowcols=None, openlist=None):
    """Get generic connected diagram of base = base
    and indices tsrc, tdis
    """
    rows = None
    cols = None
    if rowcols is not None:
        rows = rowcols[0]
        cols = rowcols[1]
    assert not np.isnan(LT), "LT has not been set"
    if avgtsrc:
        blk = np.zeros((len(trajl), LT), dtype=np.complex)
    else:
        blk = np.zeros((len(trajl), LT, LT), dtype=np.complex)
    skip = []
    outarr = np.zeros((LT, LT), dtype=np.complex)
    for i, traj in enumerate(trajl):
        filekey = get_file_name(traj)
        if openlist is None:
            fn1 = h5py.File(filekey, 'r')
        else:
            fn1 = openlist[filekey]
        traj = convert_traj(traj)
        filekey = get_file_name(traj)
        #print("key:",  'traj_'+str(traj)+'_'+base)
        try:
            if TEST44:
                outarr = np.array(fn1['traj_'+str(traj)+'_'+base][:, :TDIS_MAX+1])
            else:
                fn1['traj_'+str(traj)+'_'+base].read_direct(
                    outarr, np.s_[:, :tdismax()+1], np.s_[:, :tdismax()+1])
        except:
            namec = 'traj_'+str(traj)+'_'+base
            print(namec in fn1)
            print(fn1[namec])
            raise
        #outtemp = np.zeros((LT, LT), dtype=np.complex)
        #outtemp[:, :TDIS_MAX+1] = outarr
        #outarr = outtemp
    #    except KeyError:
    #        skip.append(i)
    #        continue
        if rows is not None and cols is not None:
            outarr = outarr[rows, cols]
        outarr *= TSTEP if 'pioncorrChk' not in base and\
            'kaoncorrChk' not in base else 1
        if avgtsrc:
            blk[i] = em.acmean(outarr)
        else:
            blk[i] = outarr
    return np.delete(blk, skip, axis=0)


@PROFILE
def getmostblks(basl, trajl, openlist):
    """Get most of the jackknife blocks,
    except for disconnected diagrams"""
    mostblks = {}
    for base in basl:
        if 'type' in base:
            continue
        if not check_key(base):
            continue
        avgtsrc = True if not WRITE_INDIVIDUAL else AVGTSRC
        if not KK_OP and ('KK2KK' in base or 'KK2sigma' in base or 'sigma2KK' in base or 'KK2pipi' in base or 'pipi2KK' in base):
            assert KK_OP is not None, "set KK_OP in h5jack.py"
            continue
        if not KCORR and ('kaoncorr' in base or 'Hbub_kaon' in base):
            assert KCORR is not None, "set KCORR in h5jack.py"
            continue
        blk = getgenconblk(base, trajl, avgtsrc=avgtsrc, openlist=openlist)
        if TESTKEY2:
            print("Printing non-averaged-over-tsrc data")
            printblk(TESTKEY2, blk)
            print("beginning of traj list = ", trajl[0], trajl[1], trajl[2])
        mostblks[base] = dojackknife(blk)
    if MPIRANK == 0:
        print("Done getting most of the jackknife blocks.")
    return mostblks
