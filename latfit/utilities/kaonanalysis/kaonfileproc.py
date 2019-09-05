"""Process kaon files"""

import re
from collections import defaultdict
import numpy as np
import h5py
from accupy import kdot
import kaonprojop
import kaonpostproc as kpp
import kaondecompose
from kaonanalysis import TSTEP12
from latfit.utilities.h5jack import LT as LT_CHECK
import latfit.utilities.read_file as rf

print("Imported kaonfileproc with LT=", LT_CHECK)

def gen_key(momdiag):
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
    keyirr = str(kdot(mom, mom))+'@'+str(kpitsep)
    return keyirr

def deltat(momdiag):
    """find k-pi tsep"""
    mat = re.search(r'(.*?)deltat_(\d+)', momdiag)
    assert mat, "bad diagram name : "+str(momdiag)
    return int(mat.group(2))


def proc_sigma_type23(type23, trajl, otype):
    """Process type 23 sigma diagrams into Q_i pieces"""
    # ltraj = len(trajl)
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
            keyirr = gen_key(momdiag)
            #assert rf.norm2(rf.mom(momdiag)) == 0, "sigma momentum is"+\
            #    " fixed to be 0"

            for i in np.arange(1, 11):
                assert str(i) in kpp.QOP_SIGMA, "Missing Q:"+str(i)
                assert otype == 'type2' or otype == 'type3',\
                    "bad type name given for sigma diagram:"+str(otype)
                if otype == 'type2':
                    kpp.QOP_SIGMA[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_sigma_type2(t1arr, i)

                elif otype == 'type3':
                    kpp.QOP_SIGMA[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_sigma_type3(t1arr, i)


def proctype123(type123, trajl, otype):
    """Process type 1 diagrams into Q_i pieces"""
    type12 = False if '3' in otype else True
    ncontract = 4 if type12 else 8
    # ltraj = len(trajl)
    ret = defaultdict(lambda: np.zeros((len(trajl), 2, LT_CHECK),
                                       dtype=np.complex))
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
                t1arr = kaondecompose.decompose_mix(t1arr,
                                                    avg_tk=True)
            else:
                t1arr = kaondecompose.decompose(t1arr,
                                                ncontract, avg_tk=True,
                                                tstep=1 if '3' in otype\
                                                else TSTEP12)

            # our operators are momentum irreps,
            # so do the projection now onto A1
            keyirr = gen_key(momdiag)
            for i in np.arange(1, 11):
                assert str(i) in kpp.QOPI0, "Missing Q:"+str(i)
                assert str(i) in kpp.QOPI2, "Missing Q:"+str(i)
                if otype == 'type1':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type1(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type1(t1arr, i, 'I2')

                elif otype == 'type2':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type2(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type2(t1arr, i, 'I2')

                elif otype == 'type3':
                    kpp.QOPI0[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type3(t1arr, i, 'I0')

                    kpp.QOPI2[str(i)][keyirr][num] +=\
                        kaonprojop.qi_proj_type3(t1arr, i, 'I2')
                elif 'mix' in momdiag:
                    ret[momdiag][num] = t1arr
                else:
                    assert None, "bad otype = "+str(otype)
    return ret

def proctype4(type4, trajl, avg_tk=False):
    """Process type 4"""
    ltraj = len(trajl)
    if avg_tk:
        shape = (ltraj, 8, 4, LT_CHECK)
    else:
        shape = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    type4 = np.zeros(shape, dtype=np.complex)
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        t1arr = np.asarray(fn1[trajstr+'_type4'])

        # decompose into pieces, optionally average over tK, tstep=1
        type4[num] = kaondecompose.decompose(t1arr, 8, avg_tk, 1)

    return type4

def procmix4(mix, trajl, avg_tk=False):
    """processes the mix4 diagrams"""
    ltraj = len(trajl)
    shape = (ltraj, 2, LT_CHECK, LT_CHECK) if not avg_tk else (
        ltraj, 2, LT_CHECK)
    ret = np.zeros(shape, dtype=np.complex)
    for num, traj in enumerate(trajl):
        trajstr = 'traj_'+str(traj)
        fn1 = h5py.File(trajstr+'.hdf5', 'r')
        for momdiag in mix:
            print(momdiag)
            t1arr = np.asarray(fn1[trajstr+'_'+momdiag])

            # do some checks
            checkarr(t1arr, False, True)

            # decompose into pieces, optionally average over tk, tstep=1
            ret[num] = kaondecompose.decompose_mix(t1arr, avg_tk)

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
