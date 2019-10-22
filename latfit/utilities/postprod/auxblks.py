"""The auxiliary block part of h5jack"""
import sys
import numpy as np
from mpi4py import MPI
import latfit.utilities.aux_write as aux
import latfit.utilities.read_file as rf
from latfit.utilities import exactmean as em
from latfit.utilities.postprod.checkblks import TESTKEY, TESTKEY2
import latfit.utilities.postprod.mostblks as mostb

# exclude all diagrams derived from aux symmetry
NOAUX = False
NOAUX = True
# aux testing, overwrite the production set with aux diagrams
# also, don't exit on finding aux pairs in the base dataset
AUX_TESTING = True
AUX_TESTING = False

#### DO NOT MODIFY THE BELOW

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

# set by h5jack
TSEP = np.nan
LT = np.nan
TSTEP = np.nan

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

# utility functions

def getindices(*xargs, **kwargs):
    """dummy function"""
    assert None
    if xargs or kwargs:
        pass
    return xargs

@PROFILE
def aux_jack(basl, trajl, numt, openlist):
    """Get the aux diagram blocks
    There is a known redundancy with getmostblks:
    a given base may be read, then read again to apply aux.
    This should be eliminated and can save time,
    and could be considered for future improvements
    if a factor of two is needed at the analysis stage.
    """
    auxblks = {}
    for base in basl:
        # get aux diagram name
        outfn = aux.aux_filen(base, stype='hdf5')
        if base in auxblks:
            if AUX_TESTING:
                continue
            else:
                print("aux pair found in data set (redundancy)")
                print("pair =", base, outfn)
                sys.exit(1)
        if not outfn:
            continue
        if TESTKEY and TESTKEY != outfn:
            continue
        if TESTKEY2 and TESTKEY2 != outfn:
            continue
        tsep = rf.sep(base)
        if tsep is not None:
            assert tsep == TSEP, "tsep of base = "+str(tsep)+" base = "+base
        nmomaux = rf.nmom(base)
        # get modified tsrc and tdis
        rows, cols = getindices(tsep, nmomaux)
        # get block from which to construct the auxdiagram
        # mean is avg over tsrc
        blk = np.zeros((numt, LT), dtype=np.complex)
        blk = TSTEP*em.acmean(mostb.getgenconblk(base,
                                                 trajl,
                                                 openlist=openlist,
                                                 avgtsrc=False,
                                                 rowcols=[rows, cols]),
                              axis=1)
        # now do the jackknife.
        # apply -1 coefficient if doing aux to a vector T
        # (ccw/cw switch in mirror)
        auxblks[outfn] = dojackknife(blk) *(
            -1.0 if rf.vecp(base) and 'FigureT' in base else 1.0)
    if MPIRANK == 0:
        print("Done getting the auxiliary jackknife blocks.")
    return auxblks if not NOAUX else {}


@PROFILE
def check_aux_consistency(auxblks, mostblks):
    """Check consistency of blocks derived via
    aux symmetry with those from production run."""
    count = 0
    exitthis = False
    if MPIRANK == 0:
        for blk in auxblks:
            if blk in mostblks:
                count += 1
                print("unused auxiliary symmetry on diagram:",
                      blk, "total unused=", count)
                try:
                    assert np.allclose(
                        fold_time(auxblks[blk]),
                        fold_time(mostblks[blk]), rtol=1e-08)
                except AssertionError:
                    print("Auxiliary symmetry failure of diagram:"+str(blk))
                    print(fold_time(auxblks[blk]))
                    print('break')
                    print(fold_time(mostblks[blk]))
                    exitthis = True
    exitthis = MPI.COMM_WORLD.bcast(exitthis, 0)
    if exitthis:
        sys.exit(1)
