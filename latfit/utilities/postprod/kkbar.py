#!/usr/bin/python3
"""Convert KKbar contractions from Masaaki's format to Dan's
and do the isospin projection"""

import sys
import re
import math
import glob
import h5py
import numpy as np
import mpi4py
from mpi4py import MPI

from latfit.utilities import read_file as rf

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

LT = 64
TDIS_MAX = 16
TSEP_PIPI = 4 # 24c
TSEP_PIPI = 3 # 24c

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def main():
    """main"""
    flist = [int(i.split('.')[0]) for i in glob.glob('*.h5')]
    flist = sorted(flist)
    flist = [str(i)+'.h5' for i in flist]
    count = 0
    for i, fil in enumerate(flist):
        if i % MPISIZE != MPIRANK and MPISIZE > 1:
            continue
        traj = re.sub('.h5', '', fil)
        fn1 = h5py.File('traj_'+traj+'_5555.hdf5', 'w') # 5555 is chosen just to not generate collisions with previously named files
        print("processing:", fil)
        fn1[kk2kkstr(fil)] = get_kk_to_kk(fil)
        fn1[kk2sigmastr(fil)] = get_kk_to_sigma(fil)
        fn1[kk_bubblestr(fil)] = get_kk_bubble(fil)
        # fn1[sigma2kkstr(fil)] = get_sigma_to_kk(fil)
        for mom in generate_pion_moms():
            fn1[kk2pipistr(fil, mom)] = get_kk_to_pipi(fil, mom)
            # fn1[pipi2kkstr(fil, mom)] = get_pipi_to_kk(fil, mom)
        fn1.close()

@PROFILE
def names(fil=None):
    """Stores dataset names"""
    if fil is None:
        assert names.static is not None, names.static
        ret = names.static
    elif names.static is not None:
        ret = names.static
    else:
        with h5py.File(fil,'r') as fn1:
            ret = list(fn1.keys())
        names.static = ret
    return ret
names.static = None

@PROFILE
def getsep(fil):
    """Get tsep (time distance between K and K in KK operator
    then validate.  Assume during the whole program run
    we are operating on only one ensemble.
    """
    if getsep.tsep_ens is None:
        dataset_names = names(fil)
        seps = set()
        for name in dataset_names:
            spl = name.split('_')
            for i in spl:
                if 'y' in i:
                    sept = int(i[1:])
                    if sept != TSEP_PIPI: # separate pipi tsep, hard coded above
                        seps.add(sept)
                    assert len(i) == 2, (i, name)
        assert len(seps) == 2, seps
        assert 0 in seps, seps
        ret = sorted(list(seps))[1]
        assert ret, ret
        getsep.tsep_ens = ret
    return getsep.tsep_ens
getsep.tsep_ens = None


@PROFILE
def kk_bubblestr(fil):
    """Get dataset name for KK bubble"""
    traj = re.sub('.h5', '', fil)
    ret = 'traj_'+traj+'_Figure_kk-bubble'
    sep = getsep(fil)
    ret += '_sep'+str(sep)
    ret += '_mom1000_mom2000'
    return ret


@PROFILE
def kk2kkstr(fil):
    """get dataset name for KK->KK"""
    traj = re.sub('.h5', '', fil)
    ret = 'traj_'+traj+'_FigureKK2KK'
    sep = getsep(fil)
    ret += '_sep'+str(sep)
    ret += '_mom1src000_mom2src000_mom1snk000'
    return ret

@PROFILE
def kk2sigmastr(fil):
    """get dataset name for KK->sigma"""
    traj = re.sub('.h5', '', fil)
    ret = 'traj_'+traj+'_FigureKK2sigma'
    sep = getsep(fil)
    ret += '_sep'+str(sep)
    ret += '_momsrc000_momsnk000'
    return ret

@PROFILE
def generate_pion_moms():
    """Generate single particle 3-momenta up to (1,1,1) 
    (with +/- inserted in all combinations)"""
    ret = []
    mvals = (-1, 0, 1)
    for i in mvals:
        for j in mvals:
            for k in mvals:
                ret.append((i, j, k))
    assert len(ret) == 27, ret
    return ret

@PROFILE
def kk2pipistr(fil, mom):
    """get dataset name for KK->KK"""
    traj = re.sub('.h5', '', fil)
    ret = 'traj_'+traj+'_FigureKK2pipi'
    sep = getsep(fil)
    ret += '_sep'+str(sep)
    ret += '_mom1src000_mom2src000_mom1snk'+rf.ptostr(mom)
    return ret

@PROFILE
def foldt_kk_to_kk(out, sep):
    """Fold time for better statistics: kk to kk"""
    ret = np.zeros((LT,LT), dtype=np.complex128)
    foldlen = LT-2*sep
    for tdis in range(LT):
        tdis_2 = addt(foldlen,-1*tdis)
        ret[:,tdis] += out[:, tdis]
        ret[:,tdis] += out[:, tdis_2]
    ret *= 0.5
    return ret

@PROFILE
def foldt_kk_to_sigma(out, sep):
    """Fold time for better statistics: kk to kk"""
    ret = np.zeros((LT,LT), dtype=np.complex128)
    foldlen = LT-sep
    for tdis in range(LT):
        tdis_2 = addt(foldlen,-1*tdis)
        ret[:,tdis] += out[:, tdis]
        ret[:,tdis] += out[:, tdis_2]
    ret *= 0.5
    return ret

@PROFILE
def foldt_kk_to_pipi(out, sep):
    """Fold time for better statistics: kk to kk"""
    ret = np.zeros((LT,LT), dtype=np.complex128)
    foldlen = LT-sep-TSEP_PIPI
    for tdis in range(LT):
        tdis_2 = addt(foldlen,-1*tdis)
        ret[:,tdis] += out[:, tdis]
        ret[:,tdis] += out[:, tdis_2]
    ret *= 0.5
    return ret





@PROFILE
def addt(*times):
    ret = sum(times)
    ret = ret % LT
    return ret

@PROFILE
def get_kk_bubble(fil):
    """Get KK bubble
    """
    sep = getsep(fil)

    if sep not in get_kk_bubble.knames:
        get_kk_bubble.knames[sep] = [
            'kaon000wlvs_'+str(sep)+'_kaon000wsvl',
            'kaon000wsvl_'+str(sep)+'_kaon000wlvs']
        get_kk_bubble.idxl[sep] = list(np.roll(range(LT), sep))

    # bubble dataset names
    dname1 = get_kk_bubble.knames[sep][0]
    dname2 = get_kk_bubble.knames[sep][1]

    # setup container
    ret = np.zeros(LT, dtype=np.complex128)

    print("getting kk bubble")

    fname = h5py.File(fil,'r')
    for dtee, idx in zip(get_kk_bubble.dts, get_kk_bubble.idxl[sep]):
        # idx = addt(dtee,-1*sep)
        # assert knames[0] in fn1, knames[0]
        # assert knames[1] in fn1, knames[1]
        ret[dtee] += mcomplex(fname[dname1][idx])
        ret[dtee] += mcomplex(fname[dname2][idx])
    fname.close()
    return ret
get_kk_bubble.knames = {}
get_kk_bubble.idxl = {}
get_kk_bubble.dts = list(range(LT))

@PROFILE
def get_kk_to_sigma(fil):
    """Get contractions for KK->sigma+isospin project
    """
    dataset_names = names(fil)
    fname = h5py.File(fil,'r')
    knames = [i for i in dataset_names if i.count('sigma') == 1]
    tsrcs = sorted(list(set([i.split('_')[-1] for i in knames])))
    tsrcs = np.array(tsrcs, dtype=np.int)
    sep = getsep(fil)
    tsrcs += sep # hack to deal with tsep offset in the first dataset
    ret = np.zeros((LT,LT), dtype=np.complex128)
    fn1 = h5py.File(fil, 'r')
    kstr = ['kaon000wlvs_y', 'kaon000wsvl_y']
    #sigmastr = 'sigma000'

    # T diagrams
    print("getting kk->sigma")
    for tsrc in tsrcs:
        for tdis in range(LT):
            toadd = 0+0j
            v1 = addt(tsrc,tdis) # inner sink
            #v2 = addt(tsrc,tdis,sep)
            v3 = addt(tsrc,-sep)
            v4 = tsrc # inner src
            dt = v3 # definition of dt: earliest time slice (in vertex sequence)

            # T diagram sets
            coeff = math.sqrt(2)/2
            sets = yyx_T_sigma_diagrams_sets(v1,v3,v4, sep)


            idx = addt(sep, tdis) # definition of Masaaki's index for T
            for i, seq in enumerate(sets):
                if not i:
                    seq = list(seq)
                    # fix the ordering, since sll and lls switch are handled by time switch
                    # (coeff happens to be the same, so this is the only change we need to make)
                    seq[0], seq[1] = seq[1], seq[0]
                    seq = tuple(seq)
                y1, y2, x = modseq3(seq, dt, idx, sep)
                dname = kstr[0]+str(y1)+'_'+kstr[1]+str(y2)+'_sigma000_x'+str(
                    x)+'_dt_'+str(dt)
                # assert dname in fname, (dname, fname)
                toadd = mcomplex(fname[dname][idx])
                if tsrc == tsrcs[0] and not tdis:
                    print(coeff, dname, idx)
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff

    fname.close()
    return foldt_kk_to_sigma(ret, sep)

@PROFILE
def mcomplex(toadd):
    """Convert tuple to complex number"""
    ret = complex(toadd[0], toadd[1])
    return ret

@PROFILE
def yyx_T_sigma_diagrams_sets(v1,v3,v4, sep):
    """ T diagrams (for K->sigma)
    +sqrt(2)/2 Tr[ gP Sl(x-z) gS Sl(z-y) gP Ss(y-x) ] xzy lls
    +sqrt(2)/2 Tr[ gP Ss(x-y) gP Sl(y-z) gS Sl(z-x) ] xyz sll

    discon, ignore
    -sqrt(2) Tr[ gP Sl(x-y) gP Ss(y-x) ] Tr[ gS Sl(z-z) ]
    -sqrt(2) Tr[ gP Ss(x-y) gP Sl(y-x) ] Tr[ gS Sl(z-z) ]
    """
    x = v3
    # w = v4
    z = v1
    y = v4 # 2->4
    dt = v3 # definition
    seqs = [(x, z, y), (x, y, z)]
    ret = []
    for seq in seqs:
        seq = cycle4(seq, sep, dt)
        ret.append(seq)
    return ret



@PROFILE
def get_kk_to_pipi(fil, mom):
    """Get contractions for KK->pipi+isospin project
    """
    negmom = tuple(np.array(mom)*-1)
    negmom = rf.ptostr(negmom)
    mom = rf.ptostr(mom)
    fname = h5py.File(fil, 'r')
    sep = getsep(fil)
    if not get_kk_to_pipi.knames:
        dataset_names = names(fil)
        get_kk_to_pipi.knames = [
            i for i in dataset_names if i.count('pion') == 2]
        get_kk_to_pipi.tsrcs = sorted(list(set(
            [i.split('_')[-1] for i in get_kk_to_pipi.knames])))
        get_kk_to_pipi.tsrcs = np.array(get_kk_to_pipi.tsrcs, dtype=np.int)
        get_kk_to_pipi.tsrcs += sep # hack to deal with tsep offset in the first dataset
    ret = np.zeros((LT,LT), dtype=np.complex128)
    kstr = ['kaon000wlvs', 'kaon000wsvl']
    pistr = ['pion'+mom, 'pion'+mom]
    pistr_neg = ['pion'+negmom, 'pion'+negmom]
    if mom == '000':
        print("getting kk->pipi")

    # R diagrams
    for tsrc in get_kk_to_pipi.tsrcs:
        for tdis in range(LT):
            toadd = 0+0j
            v1 = addt(tsrc,tdis) # inner sink
            v2 = addt(tsrc,tdis,TSEP_PIPI)
            v3 = addt(tsrc,-1*sep)
            v4 = tsrc # inner src
            dt = v3 # definition of dt: earliest time slice (in vertex sequence)

            # R diagram sets
            coeff = -1*math.sqrt(3)/2
            sets = yyxx_R_pipi_diagrams_sets(v1,v2,v3,v4, sep)
            set1 = sets[:4]
            set2 = sets[4:]

            # kstr[0], kstr[1]
            idx = addt(sep, tdis) # definition of Masaaki's index for R
            for seq in set1:
                y1, y2, x1, x2 = modseq4(seq, dt, idx, sep)
                assert x1 != x2, seq
                if x1 < x2:
                    pi1str = pistr[0]
                    pi2str = pistr_neg[1]
                else:
                    pi1str = pistr_neg[0]
                    pi2str = pistr[1]
                dname = kstr[0]+'_y'+str(y1)+'_'+kstr[1]+'_y'+str(
                    y2)+'_'+pi1str+'_x'+str(x1)+'_'+pi2str+'_x'+str(
                        x2)+'_dt_'+str(dt)
                # assert dname in fname, (dname, fname)
                toadd = mcomplex(fname[dname][idx])
                if tsrc == get_kk_to_pipi.tsrcs[0] and not tdis and mom == '000':
                    print(coeff, dname, idx)
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff

            # same as above, but kstr[1], kstr[0]
            for seq in set2:
                y1, y2, x1, x2 = modseq4(seq, dt, idx, sep)
                assert x1 != x2, seq
                if x1 < x2:
                    pi1str = pistr[1]
                    pi2str = pistr_neg[0]
                else:
                    pi1str = pistr_neg[1]
                    pi2str = pistr[0]
                dname = kstr[1]+'_y'+str(y1)+'_'+kstr[0]+'_y'+str(y2)+'_'+pi1str+'_x'+str(
                    x1)+'_'+pi2str+'_x'+str(x2)+'_dt_'+str(dt)
                # assert dname in fname, (dname, fname)
                toadd = mcomplex(fname[dname][idx])
                if tsrc == get_kk_to_pipi.tsrcs[0] and not tdis and mom == '000':
                    print(coeff, dname, idx)
                    print("toadd", toadd*coeff)
                ret[tsrc, tdis] += toadd*coeff

    fname.close()
    return foldt_kk_to_pipi(ret, sep)
get_kk_to_pipi.knames = []
get_kk_to_pipi.tsrcs = []

@PROFILE
def yyxx_R_pipi_diagrams_sets(v1,v2,v3,v4, sep):
    """R diagrams for KK->pipi
    llls
    -sqrt(3)/2 Tr[ gP Sl(x-w) gP Sl(w-z) gP Sl(z-y) gP Ss(y-x) ] xwzy
    -sqrt(3)/2 Tr[ gP Sl(x-z) gP Sl(z-w) gP Sl(w-y) gP Ss(y-x) ] xzwy

    slll
    -sqrt(3)/2 Tr[ gP Ss(x-y) gP Sl(y-w) gP Sl(w-z) gP Sl(z-x) ] xywz
    -sqrt(3)/2 Tr[ gP Ss(x-y) gP Sl(y-z) gP Sl(z-w) gP Sl(w-x) ] xyzw
    
    disconnected (so ignore)
    +sqrt(3) Tr[ gP Sl(x-y) gP Ss(y-x) ] Tr[ gP Sl(z-w) gP Sl(w-z) ] 
    +sqrt(3) Tr[ gP Ss(x-y) gP Sl(y-x) ] Tr[ gP Sl(z-w) gP Sl(w-z) ] 
    """
    #x = v3
    #w = v4
    #z = v1
    #y = v2
    # copied from R diagram
    key = (v1,v2,v3,v4, sep)
    if key not in yyxx_R_pipi_diagrams_sets.cache:
        w = v4
        x = v1
        y = v2
        z = v3
        dt = v3 # definition
        seqs = [(x, w, z, y), (x, z, w, y), (x, y, w, z), (x, y, z, w)]
        ret = []
        for seq in seqs:
            seq = cycle4(seq, sep, dt)
            ret.append(seq)
        yyxx_R_pipi_diagrams_sets.cache[key] = ret
    return yyxx_R_pipi_diagrams_sets.cache[key]
yyxx_R_pipi_diagrams_sets.cache = {}




@PROFILE
def get_kk_to_kk(fil):
    """Get contractions for KK->KK+isospin project
    involves D (direct type) diagrams and R (rectangle type diagrams)
    disconnected component is skipped for now (handled elsewhere)
    """
    dataset_names = names(fil)
    fname = h5py.File(fil,'r')
    knames = [i for i in dataset_names if i.count('kaon') == 4 or (
        not i.count('pion') and not i.count('sigma') and not i.count('kaon') == 2)]
    tsrcs = sorted(list(set([i.split('_')[-1] for i in knames])))
    tsrcs = np.array(tsrcs, dtype=np.int)
    sep = getsep(fil)
    tsrcs += sep # hack to deal with tsep offset in the first dataset
    ret = np.zeros((LT,LT), dtype=np.complex128)
    fn1 = h5py.File(fil, 'r')
    kstr = ['kaon000wsvl', 'kaon000wlvs']
    print("getting kk->kk")
    for tsrc in tsrcs:
        for tdis in range(LT):
            #print("kk to kk: tsrc, tdis =", tsrc, tdis)
            toadd = 0+0j
            v1 = addt(tsrc,tdis) # inner sink
            v2 = addt(tsrc,tdis,sep)
            v3 = addt(tsrc,-sep)
            v4 = tsrc # inner src
            dt = v3 # definition of dt: earliest time slice (in vertex sequence)

            # R diagram sets
            sets = yyxx_Rdiagrams_sets(v1,v2,v3,v4, sep)
            set1 = sets[:4]
            set2 = sets[4:]
            assert len(set1) == len(set2), (len(set1), len(set2), set1, set2)

            # kstr[0], kstr[1]
            idx = addt(sep, tdis) # definition of Masaaki's index for R
            for seq, coeff in set1:
                y1, y2, x1, x2 = modseq4(seq, dt, idx, sep)
                dname = kstr[0]+'_y'+str(y1)+'_'+kstr[1]+'_y'+str(y2)+'_'+kstr[
                    0]+'_x'+str(x1)+'_'+kstr[1]+'_x'+str(x2)+'_dt_'+str(dt)
                # assert dname in fname, (dname, fname)
                toadd = mcomplex(fname[dname][idx])
                if tsrc == tsrcs[0] and not tdis:
                    print(coeff, dname, idx)
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff

            # same as above, but kstr[1], kstr[0]
            for seq, coeff in set2:
                y1, y2, x1, x2 = modseq4(seq, dt, idx, sep)
                dname = kstr[1]+'_y'+str(y1)+'_'+kstr[0]+'_y'+str(y2)+'_'+kstr[
                    1]+'_x'+str(x1)+'_'+kstr[0]+'_x'+str(x2)+'_dt_'+str(dt)
                # assert dname in fname, (dname, fname)
                toadd = mcomplex(fname[dname][idx])
                if tsrc == tsrcs[0] and not tdis:
                    print(coeff, dname, idx)
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff

            # D diagrams
            sets = xy_xy_Ddiagrams_sets(v1,v2,v3,v4)
            set1 = sets[:2]
            set2 = sets[2:]
            assert len(set1) == len(set2), (len(set1), len(set2), sets)

            # kstr[0], kstr[1]
            for seq, coeff, revl in set1:
                toadd = 1+0j
                revl = [revl[0:2], revl[2:]]
                for tr1, rev in zip(seq, revl):
                    assert len(tr1) == 2, tr1
                    dt1 = min(tr1)
                    idx = addt(max(tr1), -1*dt1)
                    assert rev == 'ls' or rev == 'sl', rev
                    if rev == 'ls':
                        idx = addt(LT,-idx) # reverse prop direction is handled by inverting the index
                    dname = kstr[1]+'_y0_'+kstr[0]+'_x0_dt_'+str(dt1)
                    # assert dname in fname, (dname, fname)
                    if tsrc == tsrcs[0] and not tdis:
                        print(coeff, dname, idx)
                    toadd *= mcomplex(fname[dname][idx])
                if tsrc == tsrcs[0] and not tdis:
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff

            # same as above, but kstr[1], kstr[0]
            for seq, coeff, revl in set2:
                toadd = 1+0j
                revl = [revl[0:2], revl[2:]]
                for tr1, rev in zip(seq, revl):
                    dt1 = min(tr1)
                    idx = addt(max(tr1), -min(tr1))
                    assert rev == 'ls' or rev == 'sl', rev
                    if rev == 'ls':
                        idx = addt(LT,-idx) # reverse prop direction is handled by inverting the index
                    dname = kstr[1]+'_y0_'+kstr[0]+'_x0_dt_'+str(dt1)
                    # assert dname in fname, (dname, fname)
                    toadd *= mcomplex(fname[dname][idx])
                    if tsrc == tsrcs[0] and not tdis:
                        print(coeff, dname, idx)
                if tsrc == tsrcs[0] and not tdis:
                    print("toadd", toadd*coeff)
                ret[tsrc,tdis] += toadd*coeff
    fname.close()
    return foldt_kk_to_kk(ret, sep)

@PROFILE
def modseq3(seq, dt, idx, sep):
    """Change from absolute to relative (time) coordinates"""
    key = (seq, dt, idx, sep)
    if key not in modseq3.cache:
        y1, y2, x = seq
        y1 = addt(y1, -dt)
        y2 = addt(y2, -dt)
        x = addt(x, -dt-idx)
        seq = (y1, y2, x)
        chkseq(seq, sep)
        modseq3.cache[key] = seq
    return modseq3.cache[key]
modseq3.cache = {}


@PROFILE
def modseq4(seq, dt, idx, sep):
    """Change from absolute to relative (time) coordinates"""
    key = (seq, dt, idx, sep)
    if key not in modseq4.cache:
        y1, y2, x1, x2 = seq
        y1 = addt(y1, -dt)
        y2 = addt(y2, -dt)
        x1 = addt(x1, -dt-idx)
        x2 = addt(x2, -dt-idx)
        seq = (y1, y2, x1, x2)
        chkseq(seq, sep)
        modseq4.cache[key] = seq
    return modseq4.cache[key]
modseq4.cache = {}

def double_l(slist):
    """ return [*slist, *slist]
    (hack for kernprof, since it doesn't like [*a, *a] syntax)
    """
    ret = []
    for i in slist:
        ret.append(i)
    for i in slist:
        ret.append(i)
    return ret

@PROFILE
def xy_xy_Ddiagrams_sets(v1, v2, v3, v4):
    """D diagrams
    ls sl
    +1/2 Tr[ gP Sl(x-w) gP Ss(w-x) ] Tr[ gP Ss(y-z) gP Sl(z-y) ] xw yz lssl 
    +1 Tr[ gP Sl(x-y) gP Ss(y-x) ] Tr[ gP Sl(z-w) gP Ss(w-z) ] xy zw lsls (xy pair call disconnected; skip)
    +1 Tr[ gP Sl(x-y) gP Ss(y-x) ] Tr[ gP Ss(z-w) gP Sl(w-z) ] xy zw lssl (disconnected; skip)
    +1/2 Tr[ gP Sl(x-z) gP Ss(z-x) ] Tr[ gP Ss(y-w) gP Sl(w-y) ] xz yw lssl
    
    +1/2 Tr[ gP Ss(x-w) gP Sl(w-x) ] Tr[ gP Sl(y-z) gP Ss(z-y) ] xw yz slls
    +1 Tr[ gP Ss(x-y) gP Sl(y-x) ] Tr[ gP Ss(z-w) gP Sl(w-z) ] xy zw slsl (disconnected; skip)
    +1 Tr[ gP Ss(x-y) gP Sl(y-x) ] Tr[ gP Sl(z-w) gP Ss(w-z) ] xy zw slls (disconnected; skip)
    +1/2 Tr[ gP Ss(x-z) gP Sl(z-x) ] Tr[ gP Sl(y-w) gP Ss(w-y) ] xz yw slls
    """
    #coeffs = [1/2, 1, 1, 1/2, 1/2, 1, 1, 1/2]
    coeffs = [1/2, 1/2, 1/2, 1/2]
    x = v3
    w = v4
    z = v1
    y = v2
    dt = v3 # definition
    #seqs = [((x, w), (y, z)), ((x, y), (z, w)), ((x, y), (z, w)), ((x, z), (y, w))]
    seqs = [((x, w), (y, z)), ((x, z), (y, w))]
    seqs = double_l(seqs) # seqs = [*seqs, *seqs]
    #revidx = ['lssl', 'lsls', 'lssl', 'lssl', 'slls', 'slsl', 'slls', 'slls']
    revidx = ['lssl', 'lssl', 'slls', 'slls']
    assert len(seqs) == 4, (seqs, len(seqs))
    ret = []
    for seq, coeff, revl in zip(seqs, coeffs, revidx):
        #if isdiscon(seq): # handled separately by bubble code (since it needs subtraction)
        #    print("not used:", seq)
        #    continue
        #print("used:", seq)
        ret.append((seq, coeff, revl))
    return ret


@PROFILE
def isdiscon(seq):
    """check that there's one late time and one early time in the 
    trace (D diagram)"""
    ret = False
    tr1, tr2 = seq
    i, j = tr1
    k, l = tr2
    if i <= min(tr2) and j <= min(tr2):
        ret = True
    else:
        if k <= min(tr1) and l <= min(tr1):
            ret = True
    return ret


@PROFILE
def chkseq(seq, sep):
    """Check that all relative times are now either 0 or tsep"""
    for i in seq:
        assert not i or i == sep or i == TSEP_PIPI, (seq, sep)

@PROFILE
def cycle4(seq, sep, dt):
    """cycle until the earliest time slices are first"""
    key = (str(sorted(list(seq))), sep, dt)
    if key not in cycle4.cache:
        seq = np.array(seq)
        done = False
        min1 = dt
        min2 = addt(dt, sep)
        assert min1 in seq, (min1, seq)
        assert min2 in seq, (min2, seq)
        count = 0
        while not done:
            seq = np.roll(seq, -1)
            chk = {seq[0], seq[1]}
            if min1 in chk and min2 in chk:
                done = True
                break
            assert count < len(seq), (seq, min1, min2, done, chk)
            count += 1
        cycle4.cache[key] = tuple(seq)
    return cycle4.cache[key]
cycle4.cache={}

@PROFILE
def yyxx_Rdiagrams_sets(v1,v2,v3,v4, sep):
    """Convert to Masaaki's vertex format (for KK->KK).

    connected:
    lsls
    -1/2 Tr[ gP Sl(x-w) gP Ss(w-z) gP Sl(z-y) gP Ss(y-x) ] xwzy
    -1/2 Tr[ gP Sl(x-z) gP Ss(z-w) gP Sl(w-y) gP Ss(y-x) ] xzwy
    -1 Tr[ gP Sl(x-y) gP Ss(y-w) gP Sl(w-z) gP Ss(z-x) ] xywz
    -1 Tr[ gP Sl(x-y) gP Ss(y-z) gP Sl(z-w) gP Ss(w-x) ] xyzw

    slsl
    -1 Tr[ gP Ss(x-w) gP Sl(w-z) gP Ss(z-y) gP Sl(y-x) ] xwzy
    -1 Tr[ gP Ss(x-z) gP Sl(z-w) gP Ss(w-y) gP Sl(y-x) ] xzwy
    -1/2 Tr[ gP Ss(x-y) gP Sl(y-w) gP Ss(w-z) gP Sl(z-x) ] xywz
    -1/2 Tr[ gP Ss(x-y) gP Sl(y-z) gP Ss(z-w) gP Sl(w-x) ] xyzw
    /kaon000wlvs_y4_kaon000wsvl_y0_kaon000wlvs_x4_kaon000wsvl_x0_dt_40
    time sequence is v3, v4, v1, v2 in Dan's format
    first sequence is definitional: x=v3, w=v4, z=v1, y=v2
    """
    coeffs = [-1/2, -1/2, -1, -1, -1, -1, -1/2, -1/2]
    w = v4
    x = v1
    y = v2
    z = v3
    dt = v3 # definition
    seqs = [(x, w, z, y), (x, z, w, y), (x, y, w, z), (x, y, z, w)]
    seqs = double_l(seqs) # seqs = [*seqs, *seqs]
    ret = []
    for seq, coeff in zip(seqs, coeffs):
        #print("cycle4:", seq)
        seq = cycle4(seq, sep, dt)
        ret.append((seq, coeff))
    return ret



if __name__ == '__main__':
    main()
