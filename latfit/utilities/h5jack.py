#!/usr/bin/python3
"""Write jackknife blocks from h5py files"""
import sys
import os
import re
import glob
import numpy as np
import h5py
import read_file as rf
from sum_blks import isoproj
import op_compose as opc
import combine as cb
import write_discon as wd
import aux_write as aux

#representative hdf5 file, to get info about lattice
FNDEF = '1000.dat'
#size of lattice in time, lattice units
LT = 64
TSEP = 4
#format for files; don't change
STYPE='hdf5'
#precomputed indexing matrices; DON'T CHANGE
ROWST = np.tile(np.arange(LT), (LT,1))
ROWS = np.tile(np.arange(LT), (LT,1)).T
COLS = np.array([np.roll(np.arange(LT), -i, axis=0) for i in range(LT)])

##options concerning how bubble subtraction is done
TAKEREAL = False #take real of bubble if momtotal=0
STILLSUB = False #don't do subtraction on bubbles with net momentum
TIMEAVGD = False #do a time translation average (bubble is scalar now)
NOSUB = False #don't do any subtraction if true; set false if doing GEVP

##other config options
THERMNUM = 300 #eliminate configs below this number to thermalize 
TSTEP = 8 #we only measure every TSTEP time slices to save on time

###DO NOT CHANGE IF NOT DEBUGGING
JACKBUB = True #keep true for correctness; false for checking incorrect results
FOLD = True #average about the mirror point in time (True)

#diagram to look at for bubble subtraction test
TESTKEY = ''
#TESTKEY = 'FigureV_sep4_mom1src001_mom2src010_mom1snk010'
#TESTKEY = 'FigureV_sep4_mom1src000_mom2src000_mom1snk000'
#TESTKEY = 'FigureV_sep4_mom1src000_mom2src001_mom1snk000'
#TESTKEY = 'FigureV_sep4_mom1src001_mom2src000_mom1snk001'
#TESTKEY = 'FigureR_sep4_mom1src000_mom2src000_mom1snk000'

#only save this bubble (speeds up checks involving single bubbles)
BUBKEY = ''
#BUBKEY = 'Figure_Vdis_sep4_mom1000_mom2000'

try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it's not defined simply ignore the decorator.

def getindices(tsep, nmomaux):
    """Get aux indices"""
    if nmomaux == 1:
        retrows = COLS
        retcols = (-ROWST)%LT
    elif nmomaux == 2:
        retrows = COLS
        retcols = (-ROWST-tsep)%LT
    elif nmomaux == 3:
        retrows = np.roll(COLS, -tsep, axis=1)
        retcols = (-ROWST - 2*tsep)%LT
    return retrows, retcols

def trajlist():
    """Get trajectory list from files of form 
    <traj>.dat"""
    trajl = set()
    for fn in glob.glob('*.dat'):
        toadd = int(re.sub('.dat','',fn))
        if toadd >= THERMNUM: #filter out unthermalized
            trajl.add(toadd)
    trajl = sorted(list(trajl))
    print("Done getting trajectory list")
    return trajl

def baselist(fn=None):
    """Get base names of diagrams 
    (exclude trajectory info)"""
    if not fn:
        try:
            fn = h5py.File(FNDEF,'r')
        except OSError:
            print("Error: unable to locate", FNDEF)
            print("Make sure the working directory is correct.")
            sys.exit(1)
    basl = set()
    for dat in fn:
        if len(fn[dat].shape) == 2 and fn[dat].attrs['basename']:
            basl.add(fn[dat].attrs['basename'])
    fn.close()
    print("Done getting baselist")
    return basl

def bublist(fn=None):
    """Get list of disconnected bubbles."""
    if not fn:
        fn = h5py.File(FNDEF, 'r')
    bubl = set()
    for dat in fn:
        if len(fn[dat].shape) == 1 and fn[dat].attrs['basename']:
            bubl.add(fn[dat].attrs['basename'])
    fn.close()
    print("Done getting bubble list")
    return bubl

from math import fsum
@profile
def dojackknife(blk):
    """Apply jackknife to block with shape=(L_traj, L_time)"""
    out = np.zeros(blk.shape, dtype=np.complex)
    for i, _ in enumerate(blk):
        np.mean(np.delete(blk, i, axis=0), axis=0, out=out[i])
    return out

@profile
def h5write_blk(blk, outfile, extension='.jkdat'):
    """h5write block.
    """
    outh5 = outfile+extension
    if os.path.isfile(outh5):
        print("File", outh5, "exists. Skipping.")
        return
    print("Writing", outh5, "with", len(blk), "trajectories.")
    filen = h5py.File(outh5, 'w')
    filen[outfile]=blk
    filen.close()
    print("done writing jackknife blocks: ", outh5)


def overall_coeffs(iso, irr):
    """Get overall projection coefficients from iso (isopsin coefficients)
    irr (irrep projection)
    """
    ocs = {}
    for iso_dir in iso:
        for operator in irr:
            mat = re.search(r'I(\d+)/', iso_dir)
            if not mat:
                print("Error: No isopsin info found")
                sys.exit(1)
            isospin_str = mat.group(0)
            opstr = re.sub(isospin_str, '', re.sub(r'sep(\d)+/', '', iso_dir))
            for opstr_chk, outer_coeff in irr[operator]:
                if opstr_chk != opstr:
                    continue
                for original_block, inner_coeff in iso[iso_dir]:
                    ocs.setdefault(isospin_str+operator,
                                   []).append((original_block, outer_coeff*inner_coeff))
    print("Done getting projection coefficients")
    return ocs 

def jackknife_err(blk):
    """Get jackknife error from block with shape=(L_traj, L_time)"""
    len_t = len(blk)
    avg = np.mean(blk, axis=0)
    prefactor = (len_t-1)/len_t
    err = np.sqrt(prefactor*np.sum((blk-avg).real**2, axis=0))
    return avg, err

def formnum(num):
    """Format complex number in scientific notation"""
    real = '%.8e' % num.real
    if num.imag == 0:
        return real
    else:
        if num.imag < 0:
            plm = ''
        else:
            plm = '+'
        imag = '%.8e' % num.imag
        return real+plm+imag+'j'

def buberr(bubblks):
    """Show the result of different options for bubble subtraction"""
    for key in bubblks:
        if key == TESTKEY:
            avg, err = jackknife_err(bubblks[key])
            print(key, ":")
            for i, ntup in enumerate(zip(avg, err)):
                avgval, errval = ntup
                print('t='+str(i)+' avg:', formnum(avgval),
                      'err:', formnum(errval))

@profile
def h5sum_blks(allblks, ocs, outblk_shape):
    """Do projection sums on isospin blocks"""
    for opa in ocs:
        mat = re.search(r'(.*)/', opa)
        if mat:
            outdir = mat.group(0)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        flag = 0
        ntchk = None
        for base, coeff in ocs[opa]:
            if ntchk is None:
                ntchk = allblks[base].shape[0]
                outblk = np.zeros((ntchk, outblk_shape[1]), dtype=np.complex)
                if ntchk != outblk.shape[0]:
                    print("Warning:", opa, "has different number of trajectories")
                    print("new traj=", ntchk)
                    print("old traj=", outblk_shape[0])
                    print("Be careful in using these blocks in the GEVP!")
                basechk = base
            else:
                if allblks[base].shape[0] != ntchk:
                    #we have a different number of trajectories.  not intra-operator consistent
                    print("Statistics mismatch.  Not enough trajectories.")
                    print("Problematic operator:", opa)
                    print("base:", base, "base check:", basechk)
                    print("number of trajectories in base", allblks[base].shape[0])
                    print("does not match:", ntchk)
                    flag = 1
                    break
            try:
                outblk += coeff*allblks[base]
            except ValueError:
                #mismatch between shapes of outblk and base.  This is a programming error!
                print("Error: trajectory number mismatch")
                print("Problematic operator:", opa)
                print("Problematic base:", base)
                print("allblks[base].shape=", allblks[base].shape)
                print("outblk.shape=", outblk.shape)
                print("coefficient =", coeff)
                print("This is a programming error!  Please fix!")
                flag = 1
                break
        if flag == 0:
            h5write_blk(fold_time(outblk), opa)
    print("Done writing summed blocks.")
    return

@profile
def fold_time(outblk):
    if FOLD:
        retblk = [1/2 *(outblk[:,t]+outblk[:,LT-t-2*TSEP-1]) for t in range(LT)]
        return np.array(retblk).T
    else:
        return outblk
@profile
def getgenconblk(base, trajl, numt, avgtsrc=False, rowcols=None):
    """Get generic connected diagram of base=base
    and indices tsrc, tdis
    """
    rows = None
    cols = None
    if rowcols is not None:
        rows = rowcols[0]
        cols = rowcols[1]
    base2 = '_'+base
    if avgtsrc:
        blk = np.zeros((numt, LT), dtype=np.complex)
    else:
        blk = np.zeros((numt, LT, LT), dtype=np.complex)
    skip = []
    for i, traj in enumerate(trajl):
        fn = h5py.File(str(traj)+'.dat', 'r')
        try:
            outarr = np.array(fn['traj_'+str(traj)+base2])
        except KeyError:
            skip.append(i)
            continue
        if not rows is None and not cols is None:
            outarr = outarr[rows, cols]
        if avgtsrc:
            blk[i] = (LT/TSTEP)*np.mean(outarr, axis=0)
        else:
            blk[i] = outarr
    return np.delete(blk, skip, axis=0)
           
@profile
def getmostblks(basl, trajl, numt):
    """Get most of the jackknife blocks,
    except for disconnected diagrams"""
    mostblks = {}
    for base in basl:
        if TESTKEY and TESTKEY != base:
            continue
        blk = getgenconblk(base, trajl, numt, avgtsrc=True)
        mostblks[base] = dojackknife(blk)
    print("Done getting most of the jackknife blocks.")
    return mostblks

@profile
def getbubbles(bubl, trajl, numt):
    """Get all of the bubbles."""
    bubbles = {}
    for dsrc in bubl:
        if BUBKEY and dsrc != BUBKEY:
            continue
        skip = []
        for traj in trajl:
            fn = h5py.File(str(traj)+'.dat', 'r')
            keysrc = 'traj_'+str(traj)+'_'+dsrc
            try:
                ptot = rf.ptostr(wd.momtotal(fn[keysrc].attrs['mom']))
                savekey = dsrc+"@"+ptot
            except KeyError:
                continue
            if TAKEREAL and ptot == '000':
                toapp = np.array(fn[keysrc]).real
            else:
                toapp = np.array(fn[keysrc])
            bubbles.setdefault(savekey, []).append(toapp)
    for key in bubbles:
        bubbles[key] = np.asarray(bubbles[key])
    print("Done getting bubbles.")
    return bubbles 

@profile
def bubsub(bubbles):
    """Do the bubble subtraction"""
    sub = {}
    for i, bubkey in enumerate(bubbles):
        if NOSUB:
            sub[bubkey] = np.zeros((len(bubbles[bubkey])))
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
    print("Done getting averaged bubbles.")
    return sub


@profile
def bubjack(bubl, trajl, numt, bubbles=None, sub=None):
    if bubbles is None:
        bubbles = getbubbles(bubl, trajl, numt)
    if sub is None:
        sub = bubsub(bubbles)
    out = {}
    for srckey in bubbles:
        numt = len(bubbles[srckey])
        dsrc, ptot = srckey.split("@")
        for snkkey in bubbles:
            dsnk, ptot2 = snkkey.split("@")
            if ptot2 != ptot:
                continue
            outfig = wd.comb_fig(dsrc, dsnk)
            try:
                sepstr, sepval = wd.get_sep(dsrc, dsnk, outfig)
            except TypeError:
                continue
            cols = np.roll(COLS, -sepval, axis=1)
            outkey = "Figure"+outfig+sepstr+wd.dismom(rf.mom(dsrc), rf.mom(dsnk))
            if TESTKEY and outkey != TESTKEY:
                continue
            out[outkey] = np.zeros((numt, LT), dtype=np.complex)
            if JACKBUB:
                for excl in range(numt):
                    src = np.delete(bubbles[srckey], excl, axis=0)-sub[srckey][excl]
                    snk = np.delete(bubbles[snkkey], excl, axis=0)-sub[snkkey][excl]
                    np.mean(np.tensordot(src, np.conjugate(snk), axes=(0, 0))[ROWS, cols]/(len(src)*1.0), axis=0, out=out[outkey][excl])
            else:
                src = bubbles[srckey]-sub[srckey]
                snk = np.conjugate(bubbles[snkkey]-sub[snkkey])
                for excl in range(numt):
                    out[outkey][excl] = np.mean(np.outer(src[excl],snk[excl])[ROWS, cols], axis=0)
                out[outkey] = dojackknife(out[outkey])
                #np.mean is avg over tsrc
                #out[outkey][excl] = np.mean(np.array([
                #    np.mean(cb.comb_dis(src[trajnum], snk[trajnum], sepval), axis=0)
                #    for trajnum in range(numt-1)]), axis=0)
    print("Done getting the disconnected diagram jackknife blocks.")
    return out

@profile
def aux_jack(basl, trajl, numt):
    """Get the aux diagram blocks"""
    auxblks = {}
    for base in basl:
        #get aux diagram name
        outfn = aux.aux_filen(base, stype='hdf5')
        if not outfn:
            continue
        if TESTKEY and TESTKEY != outfn:
            continue
        tsep = rf.sep(base)
        assert tsep == TSEP
        nmomaux = rf.nmom(base)
        #get modified tsrc and tdis
        rows, cols = getindices(tsep, nmomaux)
        #get block from which to construct the auxdiagram
        #mean is avg over tsrc
        blk = (LT/TSTEP)*np.mean(getgenconblk(base, trajl, numt, avgtsrc=False, rowcols=[rows,cols]), axis=1)
        auxblks[outfn] = dojackknife(blk)
    print("Done getting the auxiliary jackknife blocks.")
    return auxblks

@profile
def main(FIXN=True):
    bubl = bublist()
    trajl = trajlist()
    basl = baselist()
    numt = len(trajl)
    bubblks = bubjack(bubl, trajl, numt)
    mostblks = getmostblks(basl, trajl, numt)
    auxblks = aux_jack(basl, trajl, numt)
    #do things in this order to
    #overwrite already composed disconnected diagrams (next line)
    allblks = {**mostblks, **auxblks, **bubblks}
    ocs = overall_coeffs(isoproj(FIXN, 0, dlist=list(allblks.keys()), stype=STYPE), opc.op_list(stype=STYPE))
    if TESTKEY:
        buberr(allblks)
        sys.exit(0)
    h5sum_blks(allblks, ocs, (numt, LT))

if __name__ == '__main__':
    #FIXN = input("Need fix norms before summing? True/False?")
    FIXN='False'
    if FIXN == 'True':
        FIXN = True
    elif FIXN == 'False':
        FIXN = False
    else:
        sys.exit(1)
    main(FIXN)