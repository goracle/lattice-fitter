"""The disconnected diagram part of h5jack"""
import sys
import re

from mpi4py import MPI
import numpy as np
import h5py

from latfit.utilities import write_discon as wd
from latfit.utilities import read_file as rf
from latfit.utilities.postprod.checkblks import convert_traj, printblk
from latfit.utilities.postprod.checkblks import check_key
from latfit.utilities.postprod.checkblks import TESTKEY, TESTKEY2
from latfit.utilities.postprod.checkblks import TEST44, FREEFIELD

# options concerning how bubble subtraction is done
TAKEREAL = False  # take real of bubble if momtotal=0
STILLSUB = False  # don't do subtraction on bubbles with net momentum
TIMEAVGD = False  # do a time translation average (bubble is scalar now)
NOSUB = False  # don't do any subtraction if true; set false if doing GEVP

# DO NOT CHANGE IF NOT DEBUGGING
# do subtraction in the old way
OUTERSUB = False  # (True): <AB>-<A><B>.  New way (False): <A-<A>><B-<B>>
JACKBUB = True  # keep true for correctness; false to debug incorrect results
assert(not(OUTERSUB and JACKBUB)), "Not supported!  new:JACKBUB = True, " + \
    "OUTERSUB = False, " + " debug:JACKBUB = False, OUTERSUB = False"

CONJBUB = False

# only save this bubble (speeds up checks involving single bubbles)
BUBKEY = ''
# BUBKEY = 'Figure_Vdis_sep4_mom1000_mom2000'

# debug rows/columns slicing
DEBUG_ROWS_COLS = False

#### DO NOT MODIFY THE BELOW

assert JACKBUB, "not correct.  we need to jackknife the bubbles.  if debugging, comment out"

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

# set by h5jack
TSEP = np.nan
LT = np.nan
TSTEP = np.nan
FNDEF = ''
GNDEF = ''
HNDEF = ''
ROWS = None
COLS = None
FREEFIELD = None

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

def getindices(*xargs, **kwargs):
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


@PROFILE
def jackknife_err(blk):
    """Get jackknife error from block with shape=(L_traj, L_time)"""
    blk = np.real(blk)
    len_t = len(blk)
    avg = np.mean(blk, axis=0)
    prefactor = (len_t-1)/len_t
    err = np.sqrt(prefactor*np.sum((blk-avg)**2, axis=0))
    return avg, err

@PROFILE
def testkey2(outkey, outcome, flag, excl=-1):
    """Print non-averaged over tsrc disconnected diagram"""
    if outkey != TESTKEY2:
        pass
    elif flag == 0 and excl == 0:
        print("Printing non-averaged over" +
              " tsrc disconnected diagram (" +
              " lowest traj number in list):", TESTKEY2)
        print(outcome.shape)
        print(outcome)
    elif flag == 1:
        print("Printing averaged over tsrc disconnected diagram:", TESTKEY2)
        printblk(TESTKEY2, outcome)
    elif flag == 2:
        print("Printing jackknife block for disconnected diagram:",
              TESTKEY2)
        printblk(TESTKEY2, outcome)
        sys.exit(0)

@PROFILE
def bubjack(bubl, trajl, openlist, bubbles=None, sub=None):
    """Do jackknife of disconnected (bubble) diagrams"""
    if bubbles is None:
        bubbles = getbubbles(bubl, trajl, openlist=openlist)
    if sub is None:
        sub = bubsub(bubbles)
    return dobubjack(bubbles, sub)

if JACKBUB:
    def bubjackout(out, bubtuple, keys, cols, numt):
        """get the disconnected diagram"""
        srckey, snkkey, outkey = keys
        bubbles, sub = bubtuple
        for excl in range(numt):
            if OUTERSUB:
                src = np.delete(bubbles[srckey], excl, axis=0)
                snk = conjbub(np.delete(bubbles[snkkey],
                                        excl, axis=0))
                outcome = np.tensordot(
                    src, snk, axes=(0, 0))[ROWS, cols]/(
                        len(src)*1.0)-np.outer(
                            sub[srckey][excl], sub[snkkey][excl])
            else:
                src = np.delete(bubbles[srckey],
                                excl, axis=0)-sub[srckey][excl]
                snk = conjbub(np.delete(
                    bubbles[snkkey], excl, axis=0)-sub[
                        snkkey][excl])
                # test 44 is over a single config,
                # so this won't be correct,
                # but it avoids an error message.
                outcome = np.tensordot(src, snk, axes=(0, 0))[
                    ROWS, cols]/(len(src)*1.0 if not FREEFIELD else 1.0)
                # mean is over tsrc
                # len(src) division is average over configs
                # (except for excluded one)
            out[outkey][excl] = np.mean(outcome, axis=0)
            testkey2(outkey, outcome, 0, excl)
        return out
else:
    def bubjackout(out, bubtuple, keys, cols, numt):
        """get the disconnected diagram"""
        srckey, snkkey, outkey = keys
        bubbles, sub = bubtuple
        if OUTERSUB:
            src = bubbles[srckey]
            snk = conjbub(bubbles[snkkey])
            outcome = -1*np.outer(sub[srckey], sub[snkkey])[
                ROWS, cols]
            for excl in range(numt):
                outcome = outcome + np.outer(
                    src[excl], snk[excl])[ROWS, cols]
                testkey2(outkey, outcome, 0, excl)
                # em.acmean is avg over tsrc
                out[outkey][excl] = np.mean(outcome, axis=0)
        else:
            src = bubbles[srckey]-sub[srckey]
            snk = conjbub(bubbles[snkkey]-sub[snkkey])
            for excl in range(numt):
                outcome = np.outer(src[excl], snk[excl])[ROWS, cols]
                testkey2(outkey, outcome, 0, excl)
                # em.acmean is avg over tsrc
                out[outkey][excl] = np.mean(outcome, axis=0)
        testkey2(outkey, out[outkey], 1)
        out[outkey] = dojackknife(out[outkey])
        testkey2(outkey, out[outkey], 2)
        return out

@PROFILE
def dobubjack(bubbles, sub, skip_v_bub2=False):
    """Now that we have the bubbles,
    compose the diagrams, jackknife
    skip_v_bub2 if FigureV and FigureBub2 are not needed
    """
    out = {}
    for srckey in bubbles:
        numt = len(bubbles[srckey])
        dsrc_split = srckey.split("@")
        for snkkey in bubbles:
            outkey, sepval = getdiscon_name(dsrc_split, snkkey.split("@"))
            skip1 = skip_v_bub2 and outkey and (
                'Bub2' in outkey or 'FigureV' in outkey)
            skip2 = sepval < 0 or not check_key(outkey) or outkey is None
            if skip1 or skip2:
                continue
            cols = np.roll(COLS, -sepval, axis=1)
            debug_rows(cols, outkey)
            out[outkey] = np.zeros((numt, LT), dtype=np.complex)
            out = bubjackout(out, (bubbles, sub),
                             (srckey, snkkey, outkey), cols, numt)
    if MPIRANK == 0:
        print("Done composing disconnected diagrams.")
    return out

@PROFILE
def getdiscon_name(dsrc_split, dsnk_split):
    """Get output disconnected diagram figure name
    (mimics dataset names of fully connected diagrams)
    """
    ptot = rf.procmom(dsrc_split[1])
    ptot2 = rf.procmom(dsnk_split[1])
    dsrc = dsrc_split[0]
    dsrc = re.sub('type4', 'type4_mom'+rf.ptostr(ptot), dsrc)
    dsnk = dsnk_split[0]
    if not (ptot[0] == -1*ptot2[0] and
            ptot[1] == -1*ptot2[1] and
            ptot[2] == -1*ptot2[2]):  # cc at sink, conserve momentum
        # dummy values to tell the function to stop processing this diagram
        sepval = -1
        discname = None
    else:
        # print(dsrc, dsnk)
        if 'type4' in dsnk:
            sepval = -1
            sepstr = ''
            outfig = None
        else:
            outfig = wd.comb_fig(dsrc, dsnk)
        try:
            sepstr, sepval = wd.get_sep(dsrc, dsnk, outfig)
        except TypeError:
            sepval = -1
            sepstr = ''
        discname = None
        if outfig is not None:
            discname = "Figure"+outfig+sepstr+wd.dismom(rf.mom(dsrc),
                                                        rf.mom(dsnk))
    if discname == TESTKEY:
        print(dsrc, dsnk, sepval)
    return discname, sepval

def debug_rows(cols, outkey):
    """debug function"""
    if TESTKEY and outkey != TESTKEY:
        return
    if DEBUG_ROWS_COLS:
        print(ROWS)
        print("Now cols")
        print(cols)
        print("Now COLS")
        print(COLS)
        sys.exit(0)

@PROFILE
def getbubbles(bubl, trajl, openlist=None):
    """Get all of the bubbles."""
    bubbles = {}
    for dsrc in bubl:
        if 'type' in dsrc:
            continue
        if BUBKEY and dsrc != BUBKEY:
            continue
        for traj in trajl:
            filekey = get_file_name(traj)
            errfn = str(filekey)
            if openlist is None:
                fn1 = h5py.File(filekey, 'r')
            else:
                fn1 = openlist[filekey]
            traj = convert_traj(traj)
            filekey = get_file_name(traj)
            keysrc = 'traj_' + str(traj) + '_' + dsrc
            assert(keysrc in fn1), "key = " + keysrc + \
                " not found in fn1:"+errfn
            #try:
            #    pdiag = fn1[keysrc].attrs['mom']
            #except KeyError:
            pdiag = rf.mom(keysrc)
            try:
                ptot = rf.ptostr(wd.momtotal(pdiag))
            except TypeError:
                print(pdiag, keysrc, filekey)
                sys.exit(1)
            savekey = dsrc+"@"+ptot
            if TAKEREAL and ptot == '000':
                toapp = np.array(fn1[keysrc]).real
            else:
                toapp = np.array(fn1[keysrc])
            bubbles.setdefault(savekey, []).append(toapp)
    for key in bubbles:
        bubbles[key] = np.asarray(bubbles[key])
    #print("Done getting bubbles.")
    return bubbles


@PROFILE
def bubsub(bubbles):
    """Do the bubble subtraction"""
    sub = {}
    for i, bubkey in enumerate(bubbles):
        if i:
            pass
        if NOSUB:
            if JACKBUB:
                sub[bubkey] = np.zeros((len(bubbles[bubkey])))
            else:
                sub[bubkey] = np.zeros((len(bubbles[bubkey][0])))
            continue
        if STILLSUB:
            if bubkey.split('@')[1] != '000':
                sub[bubkey] = np.zeros((len(bubbles[bubkey])))
                continue
        if JACKBUB:
            sub[bubkey] = dojackknife(bubbles[bubkey])
            if TIMEAVGD:
                sub[bubkey] = np.mean(sub[bubkey], axis=1)
        else:
            if TIMEAVGD:
                sub[bubkey] = np.mean(bubbles[bubkey])
            else:
                sub[bubkey] = np.mean(bubbles[bubkey], axis=0)
    #print("Done getting averaged bubbles.")
    return sub


if CONJBUB:
    @PROFILE
    def conjbub(bub):
        """Complex conjugate bubble depending on global variable
        (conjugate)"""
        return np.conjugate(bub)
else:
    @PROFILE
    def conjbub(bub):
        """Complex conjugate bubble depending on global variable
        (do not conjugate)"""
        return bub

@PROFILE
def getdisconwork(bubl):
    """Get bubble combinations to compose for this rank"""
    bublcomb = set()
    bubl = sorted(list(bubl))
    for src in bubl:
        for snk in bubl:
            bublcomb.add((src, snk))
    nodebublcomb = getwork(sorted(list(bublcomb)))
    nodebubl = set()
    for src, snk in nodebublcomb:
        nodebubl.add(src)
        nodebubl.add(snk)
    nodebubl = sorted(list(nodebubl))
    return nodebubl

@PROFILE
def getbubl(fn1):
    """Get bubble list from selected file"""
    bubl = set()
    for dat in fn1:
        #try:
        #    basen = fn1[dat].attrs['basename']
        #except KeyError:
        basen = rf.basename(dat)
        if 'Vdis' in basen or 'bubble' in basen or 'type' in basen:
            #if len(fn1[dat].shape) == 1 and basen:
            bubl.add(basen)
    return bubl

@PROFILE
def buberr(bubblks):
    """Show the result of different options for bubble subtraction"""
    for key in bubblks:
        bubblks[key] = fold_time(bubblks[key])
        if key == TESTKEY:
            avg, err = jackknife_err(bubblks[key])
            print("Printing first three jackknife samples from t=0:")
            print(bubblks[key][0][0])
            print(bubblks[key][1][0])
            print(bubblks[key][2][0])
            print(key, ":")
            for i, ntup in enumerate(zip(avg, err)):
                avgval, errval = ntup
                print('t='+str(i)+' avg:', formnum(avgval),
                      'err:', formnum(errval))

@PROFILE
def formnum(num):
    """Format complex number in scientific notation"""
    real = '%.8e' % num.real
    if num.imag == 0:
        ret = real
    else:
        if num.imag < 0:
            plm = ''
        else:
            plm = '+'
        imag = '%.8e' % num.imag
        ret = real+plm+imag+'j'
    return ret




@PROFILE
def bublist(fn1=None):
    """Get list of disconnected bubbles."""
    if not fn1:
        fn1 = h5py.File(FNDEF, 'r')
        gn1 = h5py.File(GNDEF, 'r')
        hn1 = h5py.File(HNDEF, 'r')
    bubl = getbubl(fn1).intersection(getbubl(gn1)).intersection(getbubl(hn1))
    fn1.close()
    gn1.close()
    hn1.close()
    if MPIRANK == 0:
        print("Done getting bubble list")
    bubl = sorted(list(bubl))
    return bubl
