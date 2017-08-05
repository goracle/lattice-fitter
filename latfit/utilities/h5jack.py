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

FNDEF = '9995.dat'
LT = 32
STYPE='hdf5'
ROWS = np.tile(np.arange(LT), (LT,1))
COLS = np.array([np.roll(np.arange(LT), -i, axis=0) for i in range(LT)])

@profile
def trajlist():
    """Get trajectory list from files of form 
    <traj>.dat"""
    trajl = set()
    for fn in glob.glob('*.dat'):
        trajl.add(int(re.sub('.dat','',fn)))
    trajl = sorted(list(trajl))
    print("Done getting trajectory list")
    return trajl

@profile
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

@profile
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

@profile
def dojackknife(blk):
    """Apply jackknife to block with shape=(L_traj, L_time)"""
    for i, _ in enumerate(blk):
        blk[i] = np.mean(np.delete(blk, i, axis=0), axis=0)
    return blk

@profile
def h5write_blk(blk, outfile, extension='.jkdat'):
    """h5write block.
    """
    outfile = outfile+extension
    if os.path.isfile(outfile):
        print("File", outfile, "exists. Skipping.")
        return
    print("Writing", outfile, "with", len(blk), "trajectories.")
    filen = h5py.File(outfile, 'w')
    filen[outfile]=blk
    filen.close()
    print("done writing jackknife blocks: ", outfile)


@profile
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

@profile
def jackknife_err(blk):
    """Get jackknife error from block with shape=(L_traj, L_time)"""
    len_t = len(blk)
    avg = np.mean(blk, axis=0)
    prefactor = (len_t-1)/len_t
    err = np.sqrt(prefactor*np.sum((blk-avg)**2, axis=0))
    return avg, err

@profile
def h5sum_blks(allblks, ocs, outblk_shape):
    """Do projection sums on isospin blocks"""
    for opa in ocs:
        mat = re.search(r'(.*)/', opa)
        if mat:
            outdir = mat.group(0)
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
        outblk = np.zeros(outblk_shape, dtype=np.complex)
        flag = 0
        for base, coeff in ocs[opa]:
            try:
                outblk += coeff*allblks[base]
            except ValueError:
                print("Error: trajectory number mismatch")
                print("Problematic operator:", opa)
                flag = 1
                break
        if flag == 0:
            h5write_blk(outblk, opa)
    print("Done writing summed blocks.")
    return
            
@profile
def get_mostblks(basl, trajl, numt):
    """Get most of the jackknife blocks,
    except for disconnected diagrams"""
    mostblks = {}
    for base in basl:
        blk = np.zeros((numt, LT), dtype=np.complex)
        skip = []
        for i, traj in enumerate(trajl):
            fn = h5py.File(str(traj)+'.dat', 'r')
            try:
                blk[i] = np.mean(fn['traj_'+str(traj)+'_'+base], axis=0)
            except KeyError:
                skip.append(i)
        blk = np.delete(blk, skip, axis=0)
        mostblks[base] = dojackknife(blk)
    print("Done getting most of the jackknife blocks.")
    return mostblks

@profile
def getbubbles(bubl, trajl, numt):
    """Get all of the bubbles."""
    bubbles = {}
    print("length of bubble list:", len(bubl))
    for dsrc in bubl:
        skip = []
        for traj in trajl:
            fn = h5py.File(str(traj)+'.dat', 'r')
            keysrc = 'traj_'+str(traj)+'_'+dsrc
            try:
                savekey = dsrc+"@"+rf.ptostr(wd.momtotal(fn[keysrc].attrs['mom']))
            except KeyError:
                continue
            bubbles.setdefault(savekey, []).append(fn[keysrc])
    for key in bubbles:
        print("stored key:", key)
        bubbles[key] = np.asarray(bubbles[key])
    print("Done getting bubbles.")
    return bubbles 

@profile
def bubsub(bubbles):
    """Do the bubble subtraction"""
    sub = {}
    for i, bubkey in enumerate(bubbles):
        print("Getting jackknife of bubble:",i)
        sub[bubkey] = dojackknife(bubbles[bubkey])
    print("Done getting averaged bubbles.")
    return sub


@profile
def bubbles_jack(bubl, trajl, numt, bubbles=None, sub=None):
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
            out[outkey] = np.zeros((numt, LT), dtype=np.complex)
            for excl in range(numt):
                src = np.delete(bubbles[srckey], excl, axis=0)-sub[srckey][excl]
                snk = np.delete(bubbles[snkkey], excl, axis=0)-sub[snkkey][excl]
                np.mean(np.tensordot(src, np.conjugate(snk, out=snk), axes=(0, 0))[ROWS, cols], axis=0, out=out[outkey][excl])
                #out[outkey][excl] = np.mean(np.array([
                #    np.mean(cb.comb_dis(src[trajnum], snk[trajnum], sepval), axis=0)
                #    for trajnum in range(numt-1)]), axis=0)
            print("Done with sink:", dsnk)
    print("Done getting the disconnected diagram jackknife blocks.")
    return out

@profile
def main(FIXN=True):
    bubl = bublist()
    trajl = trajlist()
    basl = baselist()
    numt = len(trajl)
    z = bubbles_jack(bubl, trajl, numt)
    allblks = {**get_mostblks(basl, trajl, numt), **z}
    ocs = overall_coeffs(isoproj(FIXN, 0, dlist=basl, stype=STYPE), opc.op_list(stype=STYPE))
    h5sum_blks(allblks, ocs, (numt, LT))
1

if __name__ == '__main__':
    #FIXN = input("Need fix norms before summing? True/False?")
    FIXN='True'
    if FIXN == 'True':
        FIXN = True
    elif FIXN == 'False':
        FIXN = False
    else:
        sys.exit(1)
    main(FIXN)
