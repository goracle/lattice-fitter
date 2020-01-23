"""Main function's fit window functions"""
import sys
import mpi4py
from mpi4py import MPI

from latfit.config import RANGE_LENGTH_MIN, VERBOSE, FIT
from latfit.analysis.errorcodes import XmaxError, FinishedSkip
from latfit.analysis.errorcodes import FitRangeInconsistency
from latfit.analysis.filename_windows import finished_windows
from latfit.analysis.filename_windows import inconsistent_windows
from latfit.mainfunc.cache import partial_reset

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

def update_fitwin(meta, tadd, tsub, problemx=None):
    """Update fit window"""
    # tadd tsub cut
    if tadd or tsub:
        #print("tadd =", tadd, "tsub =", tsub)
        for _ in range(tadd):
            meta.incr_xmin(problemx=problemx)
        for _ in range(tsub):
            meta.decr_xmax(problemx=problemx)
        partial_reset()
    if finished_win_check(meta, tsub=tsub):
        raise FinishedSkip
    if inconsistent_win_check(meta, tsub=tsub):
        raise FitRangeInconsistency

def xmin_err(meta, err):
    """Handle xmax error"""
    if VERBOSE:
        print("Test fit failed; bad xmin.")
    # if we're past the halfway point, then this error is likely a late time
    # error, not an early time error (usually from pion ratio)
    if err.problemx > (meta.options.xmin + meta.options.xmax)/2:
        raise XmaxError(problemx=err.problemx)
    update_fitwin(meta, 1, 0, problemx=err.problemx)
    if VERBOSE:
        print("new xmin, xmax =", meta.options.xmin, meta.options.xmax)
    if meta.fitwindow[0] > meta.options.xmin and FIT:
        print("***ERROR***")
        print("fit window beyond xmin:", meta.fitwindow)
        sys.exit(1)
    #meta.fitwindow = fitrange_err(meta.options, meta.options.xmin,
    #                              meta.options.xmax)
    #print("new fit window = ", meta.fitwindow)
    return meta

def xmax_err(meta, err):
    """Handle xmax error"""
    if VERBOSE:
        print("Test fit failed; bad xmax. problemx:", err.problemx)
    update_fitwin(meta, 0, 1, problemx=err.problemx)
    if VERBOSE:
        print("xmin, new xmax =", meta.options.xmin, meta.options.xmax)
    if meta.fitwindow[1] < meta.options.xmax and FIT:
        print("***ERROR***")
        print("fit window beyond xmax:", meta.fitwindow)
        sys.exit(1)
    #meta.fitwindow = fitrange_err(meta.options, meta.options.xmin,
    #                              meta.options.xmax)
    #print("new fit window = ", meta.fitwindow)
    return meta

def finished_win_check(meta, tsub=None):
    """Skip if we've already got results for this window"""
    fwin = finished_windows()
    ret = False
    for i in fwin:
        if i[0] == meta.fitwindow[0] and i[1] == meta.fitwindow[1]:
            if VERBOSE:
                prs = "fit window "+str(
                    i)+" already finished.  Skipping.  rank: "+str(MPIRANK)
                if tsub is not None and int(tsub) != 1:
                    prs += " tsub: "+str(tsub)
                print(prs)
            ret = True
    return ret

def inconsistent_win_check(meta, tsub=None):
    """Check if the window has already given an inconsistent result"""
    fwin = inconsistent_windows()
    ret = False
    for i in fwin:
        if i[0] == meta.fitwindow[0] and i[1] == meta.fitwindow[1]:
            if VERBOSE:
                prs = "fit window "+str(
                    i)+" already found to be inconsistent."+\
                    "  Skipping.  rank: "+str(MPIRANK)
                if tsub is not None and int(tsub) != 1:
                    prs += " tsub: "+str(tsub)
                print(prs)
            ret = True
    return ret


def winsize_check(meta, tadd, tsub):
    """Check proposed new fit window size to be sure
    there are enough time slices"""
    new_fitwin_len = meta.fitwindow[1] - meta.fitwindow[0] + 1 - tadd - tsub
    ret = new_fitwin_len > 0 and RANGE_LENGTH_MIN <= new_fitwin_len
    return ret
