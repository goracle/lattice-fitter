"""Process kaon files"""

import sys
import re
from collections import defaultdict
import numpy as np
import h5py
from latfit.utilities.h5jack import LT as LT_CHECK
import kaonprojop
import kaonpostproc as kpp
import kaondecompose
from kaonanalysis import TSTEP12
import latfit.utilities.read_file as rf

print("Imported kaonfileproc with LT=", LT_CHECK)

def genKey(momdiag):
    """Get operator output key"""
    kpitsep = deltat(momdiag)
    mom = rf.mom(momdiag, printerr=False)
    if mom is None:
        if 'ktopi_' in momdiag or 'sigma' in momdiag:
            mom = np.array([0, 0, 0]) # K to single particle, K has p=0
        else:
            assert mom, "Bad diagram name : "+str(momdiag)
    if len(mom) == 2:
        mom = mom[1]
    keyirr = str(np.dot(mom, mom))+'@'+str(kpitsep)
    return keyirr

def deltat(momdiag):
    """find k-pi tsep"""
    mat = re.search('deltat_(\d)', momdiag)
    assert mat, "bad diagram name : "+str(momdiag)
    return int(mat.group(1))


def procSigmaType23(type23, trajl, otype):
    """Process type 23 sigma diagrams into Q_i pieces"""
    ltraj = len(trajl)
    type12 = False if '3' in otype else True
    ncontract = 4 if type12 else 8
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        for momdiag in type23:

            t1arr = np.asarray(fn1[trajstr+'_'+momdiag])

            # do some checks
            checkarr(t1arr, type12)

            # decompose into pieces,
            # optionally average over tK, type3 has tstep1
            t1arr = kaondecompose.decompose(t1arr, ncontract, True,
                                            1 if '3' in otype else TSTEP12)

            # our operators are momentum irreps,
            # so do the projection now onto A1
            keyirr = genKey(momdiag)
            #assert rf.norm2(rf.mom(momdiag)) == 0, "sigma momentum is"+\
            #    " fixed to be 0"

            ret = None
            for i in np.arange(1, 11):
                if otype == 'type2':
                    kpp.QOP_sigma[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojSigmaType2(t1arr, i)

                elif otype == 'type3':
                    kpp.QOP_sigma[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojSigmaType3(t1arr, i)

                else:
                    print("bad type given for sigma diagram:", otype)
                    raise

def proctype123(type123, trajl, otype):
    """Process type 1 diagrams into Q_i pieces"""
    type12 = False if '3' in otype else True
    ncontract = 4 if type12 else 8
    ltraj = len(trajl)
    ret = []
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        for momdiag in type123:

            print("importing:", momdiag)

            t1arr = np.asarray(fn1[trajstr+'_'+momdiag])

            # do some checks
            checkarr(t1arr, type12, mix='mix' in momdiag)

            # decompose into pieces,
            # optionally average over tK, type3 has tstep1
            if 'mix' in momdiag:
                t1arr = kaondecompose.decompose_mix(t1arr, avgTk=True)
            else:
                t1arr = kaondecompose.decompose(t1arr,
                                                ncontract, avgTk=True,
                                                tstep=1 if '3' in otype\
                                                else TSTEP12)

            # our operators are momentum irreps,
            # so do the projection now onto A1
            keyirr = genKey(momdiag)
            for i in np.arange(1, 11):
                if otype == 'type1':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType1(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType1(t1arr, i, 'I2')

                elif otype == 'type2':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType2(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType2(t1arr, i, 'I2')

                elif otype == 'type3':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType3(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.QiprojType3(t1arr, i, 'I2')
                elif 'mix' in momdiag:
                    ret.append(t1arr)
                else:
                    assert None, "bad otype = "+str(otype)
    return np.asarray(ret)

def proctype4(type4, trajl, avgTk=False):
    """Process type 4"""
    ltraj = len(trajl)
    if avgTk:
        shape = (ltraj, 8, 4, LT_CHECK)
    else:
        shape = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    type4dict = {}
    type4dict = defaultdict(lambda: np.zeros(
        shape, dtype=np.complex), type4dict)
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        for momdiag in type4:

            t1arr = np.asarray(fn1[trajstr+'_'+momdiag])

            # decompose into pieces, optionally average over tK, tstep=1
            type4dict[momdiag][num] = kaondecompose.decompose(
                t1arr, 8, avgTk, 1)

    return type4dict

def procmix4(mix, trajl, tKavg=False):
    """processes the mix4 diagrams"""
    ltraj = len(trajl)
    shape = (ltraj, 2, LT_CHECK, LT_CHECK)
    ret = np.zeros(shape, dtype=np.complex)
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        for momdiag in mix:

            t1arr = np.asarray(fn1[trajstr+'_'+momdiag])

            # do some checks
            checkarr(t1arr, False, True)
            assert shape[1:] == (2, LT_CHECK,
                                 LT_CHECK), "Bad shape for type 4"

            # decompose into pieces, optionally average over tk, tstep=1
            ret[num] = kaondecompose.decompose_mix(t1arr, tKavg)

    return ret


def checkarr(t1arr, type12=False, mix=False):
    """Some asserts on the k->x arrays"""
    assert len(t1arr.shape) == 1, "Should be 1d array"
    size = len(t1arr)
    ncontract = None
    if mix:
        assert 2*LT_CHECK**2 == size, "Bad mix array size:"+str(size)
    else:
        ncontract = 4 if type12 else 8
        assert LT_CHECK**2*ncontract*4 == size, "Bad results array size:"+\
            str(size)
    return size
