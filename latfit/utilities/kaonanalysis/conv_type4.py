#!/usr/bin/python3

import sys
import os
from mpi4py import MPI
import h5py
import numpy as np
from latfit.utilities import read_file as rf

LT = 64
TSEP = 3
RANK = MPI.COMM_WORLD.Get_rank()
SIZE = MPI.COMM_WORLD.Get_size()
print('rank, size', RANK, SIZE)

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
    reto = retrieve_order()
    type3sets = type3_sets(dirname='1130')
    for item in sys.argv[1:]:
        num = config_num(item)
        conv_type3(num, reto, type3sets)
        print("done with type3")
        print("converting type4")
        conv_type4(num, reto)


@PROFILE
def get_toapp(fn1, key, tdis, rcache):
    """Process data point"""
    if key in rcache:
        arr = rcache[key]
    else:
        arr = np.array(fn1[key])
        rcache[key] = arr
    assert len(arr) == LT, len(arr)
    ctup = arr[tdis]
    # should be complex number, so tup should have 2 numbers
    assert len(ctup) == 2, ctup
    toapp = np.complex(ctup[0], ctup[1])
    return toapp, rcache

@PROFILE
def type3_sets(dirname=''):
    """Build string sets for type3"""
    # build momset, deltatset (only needs to be run once)
    tsep = None
    momset = set()
    deltatset = set()
    prev = os.getcwd()
    if dirname:
        os.chdir(dirname)
    for filename in os.listdir('.'):
        if filename.count('pion') != 2:
            continue
        # get pion dt's relative to kaon
        dname = filename.split('_')
        dt1, dt2 = None, None
        for idx, item in enumerate(dname):
            if idx and 'pion' in item:
                dt1 = int(dname[idx-1])
            elif item == 'kaon000':
                dt2 = int(dname[idx-1])
                break
        assert dt1 is not None, (dt1, filename)
        assert dt2 is not None, (dt2, filename)
        tsep = np.abs(dt1 - dt2)
        assert TSEP == tsep, (TSEP, tsep, filename)
        deltat = min(dt1, dt2)
        deltatset.add(deltat)

        # build individual pion momentum list
        mid = filename.split('n')[1].split('p')[0]
        dtee = mid.split('_')[-2]
        if mid.count(dtee) > 1:
            continue
        mom = mid.split(dtee)[0][:-1]
        assert mom, (mom, filename)
        momset.add(mom)

    momset = sorted(list(momset))
    #print(momset)
    deltatset = sorted(list(deltatset))

    momsetneg = []

    # build negmomset (only needs to be run once)
    for mom in momset:
        numeric_mom = rf.procmom(mom)
        neg = [-1 * i for i in numeric_mom]
        assert len(neg) == 3, (neg, mom)
        rstr = ''
        for dir in neg:
            if dir < 0:
                dir = -1*dir
                rstr += '_'
            rstr += str(dir)
        momsetneg.append(rstr)
    if dirname:
        os.chdir(prev)
    return momset, momsetneg, deltatset


def create_dsets(threen1, okey, lreto):
    """Create and zero out datasets for diagram and mix counter-part"""
    dset = threen1.create_dataset(okey, (lreto*LT**2,),
                                    dtype=np.complex128)
    dset[:] = np.zeros(lreto*LT**2, dtype=np.complex128)
    dsetm = threen1.create_dataset(okey+'_mix3', (2*LT**2,),
                                    dtype=np.complex128)
    dsetm[:] = np.zeros(2*LT**2, dtype=np.complex128)


def populate3(threen1, num, reto, type3sets):
    """Fill output file with blank datasets
    Return a dict of the dataset handles"""
    momset, momsetneg, deltatset = type3sets
    tsep = TSEP
    for sink in ['sigma', 'pion']:
        for deltat in deltatset:
            deltat = str(deltat)
            okey = 'traj_'+num+'_type3_'+sink+'_deltat_'+deltat
            create_dsets(threen1, okey, len(reto))
    for mom, negmom in zip(momset, momsetneg):
        for deltat in deltatset:
            deltat = str(deltat)
            okey = 'traj_'+num+'_type3_deltat_'+deltat+'_tsep'+str(
                tsep)+'_mom'+mom
            create_dsets(threen1, okey, len(reto))
    print("done populating type3 output")

@PROFILE
def conv_type3(num, reto, type3sets):
    """Convert type3 from Masaaki's format to Dan's
    files are in dir named <config num>
    example:  pion000_14_pion000_11_kaon000_0.1130.h5
    """
    momset, momsetneg, deltatset = type3sets
    tsep = TSEP
    os.chdir(num)
    threen1 = h5py.File('../traj_'+num+'_1_kpipi.hdf5', 'w',
                        driver='mpio', comm=MPI.COMM_WORLD)
    count = -1
    populate3(threen1, num, reto, type3sets)
    threen1.close()
    threen1 = h5py.File('../traj_'+num+'_1_kpipi.hdf5', 'r+',
                        driver='mpio', comm=MPI.COMM_WORLD)
    #for item in threen1:
    #    if not RANK:
    #        print(item)
    #    threen1[item][RANK] = RANK+0j
    #    break
    #sys.exit()

    # K->sigma/pi
    for sink in ['sigma', 'pion']:
        for deltat in deltatset:
            count += 1
            if RANK != (count % SIZE):
                continue
            deltat = str(deltat)
            output = 'traj_'+num+'_type3_'+sink+'_deltat_'+deltat

            dt1 = deltat

            # read/convert
            const = (num, sink, None, reto)
            ret, mret = readconv_type3(dt1, *const)

            # sum/avg
            ret *= -1 # another correction since Masaaki applied g5

            # write
            okey = 'traj_'+num+'_type3_'+sink+'_deltat_'+deltat
            threen1[okey][:] = ret
            threen1[okey+'_mix3'][:] = mret
            print("done converting:", output, 'count =', count)

    # K->pipi
    for mom, negmom in zip(momset, momsetneg):
        for deltat in deltatset:
            count += 1
            if RANK != (count % SIZE):
                continue

            deltat = str(deltat)
            output = 'traj_'+num+'_type3_deltat_'+deltat+'_sep'+str(tsep)+\
                '_mom'+mom

            dt1 = deltat
            dt2 = str(int(deltat)+tsep)

            # read/convert
            const = (num, mom, negmom, reto)
            const2 = (num, negmom, mom, reto)
            ret, mret = readconv_type3((dt1, dt2), *const)
            pair = readconv_type3((dt2, dt1), *const2)

            # sum/avg
            ret += pair[0]
            ret *= -1 # another correction since Masaaki applied g5
            mret += pair[1]
            #ret /= 2 # done in kaonanalysis
            #mret /= 2

            # write
            okey = 'traj_'+num+'_type3_deltat_'+deltat+'_tsep'+str(
                tsep)+'_mom'+mom
            threen1[okey][:] = ret
            threen1[okey+'_mix3'][:] = mret
            print("done converting:", output, 'count =', count)
    print("attempting to close type3 buffer.")
    threen1.close()
    print("type3 buffer closed.")
    os.chdir('..')
    print("done with cd, rank", RANK)

def readconv_type3(dts, num, mom_sink, negmom, reto):
    if isinstance(dts, str) or isinstance(dts, int):
        sink = mom_sink
        dt1 = dts
        inp = sink+'000_'+dt1+'_'
    else:
        mom = mom_sink
        dt1, dt2 = dts
        assert negmom, (mom, negmom, dts)
        inp = 'pion'+mom+'_'+dt1+'_pion'+negmom+'_'+dt2+'_'
    rcache = {}
    fname = inp+'kaon000_0.'+num+'.h5'
    assert fname not in readconv_type3.chk, fname
    readconv_type3.chk.add(fname)
    try:
        fn1 = h5py.File(fname, 'r')
    except OSError:
        print('cannot open:', fname)
        raise
    ret, mret = inner(reto, fn1, rcache, inp=inp)
    fn1.close()
    return ret, mret
readconv_type3.chk = set()

@PROFILE
def inner(reto, fn1, rcache, inp=''):
    """The inner conversion loop"""
    ret = []
    mret = []
    leto = len(reto)
    for tk in range(LT):
        tail = '_dt_'+str(tk)
        for tdis in range(LT):
            for ridx, con in enumerate(reto):
                key = inp+'kaon000_0_'+con+tail+'/correlator'
                toapp, rcache = get_toapp(fn1, key, tdis, rcache)
                ret.append(toapp)
            for mtype in ['mixid', 'mixg5']: # order is important here, and is flipped from what one would naively expect from Dan's format; this is due to application of g5 by Masaaki
                key = inp+'kaon000_0_'+mtype+tail+'/correlator'
                toapp, rcache = get_toapp(fn1, key, tdis, rcache)
                mret.append(toapp)
    ret = np.array(ret, dtype=np.complex)
    mret = np.array(mret, dtype=np.complex)
    return ret, mret

@PROFILE
def config_num(item):
    """Get (gauge) config number"""
    num = int(item.split('.')[0])
    num = str(num)
    assert item == num+'.h5', item
    return num

@PROFILE
def conv_type4(num, reto):
    """Convert type4 from Masaaki's format to Dan's"""

    # read/convert
    fn1 = h5py.File(num+'.h5', 'r')
    rcache = {}
    ret, mret = inner(reto, fn1, rcache)
    fn1.close()

    # write
    if not RANK:
        fn1 = h5py.File('traj_'+num+'_0_kpipi.hdf5', 'w')
        fn1['traj_'+num+'_type4'] = ret
        fn1['traj_'+num+'_mix4'] = mret
        fn1.close()

    return ret
    

@PROFILE
def retrieve_order():
    """Gives Masaaki's list of contractions in Dan's order

    answer:

    lloop_type0_GammaMUGamma5_GammaMUGamma5
    lloop_type2_GammaMUGamma5_GammaMUGamma5
    lloop_type1_GammaMU_GammaMU
    lloop_type3_GammaMU_GammaMU

    sloop_type0_GammaMUGamma5_GammaMUGamma5
    sloop_type2_GammaMUGamma5_GammaMUGamma5
    sloop_type1_GammaMU_GammaMU
    sloop_type3_GammaMU_GammaMU

    lloop_type0_GammaMU_GammaMU
    lloop_type2_GammaMU_GammaMU
    lloop_type1_GammaMUGamma5_GammaMUGamma5
    lloop_type3_GammaMUGamma5_GammaMUGamma5

    sloop_type0_GammaMU_GammaMU
    sloop_type2_GammaMU_GammaMU
    sloop_type1_GammaMUGamma5_GammaMUGamma5
    sloop_type3_GammaMUGamma5_GammaMUGamma5

    lloop_type0_GammaMUGamma5_GammaMU
    lloop_type2_GammaMUGamma5_GammaMU
    lloop_type1_GammaMUGamma5_GammaMU
    lloop_type3_GammaMUGamma5_GammaMU

    sloop_type0_GammaMUGamma5_GammaMU
    sloop_type2_GammaMUGamma5_GammaMU
    sloop_type1_GammaMUGamma5_GammaMU
    sloop_type3_GammaMUGamma5_GammaMU

    lloop_type0_GammaMU_GammaMUGamma5
    lloop_type2_GammaMU_GammaMUGamma5
    lloop_type1_GammaMU_GammaMUGamma5
    lloop_type3_GammaMU_GammaMUGamma5

    sloop_type0_GammaMU_GammaMUGamma5
    sloop_type2_GammaMU_GammaMUGamma5
    sloop_type1_GammaMU_GammaMUGamma5
    sloop_type3_GammaMU_GammaMUGamma5
    """
    ret = []
    gaml1 = ['GammaMUGamma5_GammaMUGamma5', 'GammaMU_GammaMU',
            'GammaMUGamma5_GammaMU', 'GammaMU_GammaMUGamma5']
    gaml2 = list(gaml1)
    gaml2[0], gaml2[1] = gaml2[1], gaml2[0]
    for gam1, gam2 in zip(gaml1, gaml2):
        for loop in ['lloop', 'sloop']:
            for tnum in [0, 2, 1, 3]:
                if tnum in [0, 2]:
                    gam = gam1
                else:
                    gam = gam2
                type = 'type'+str(tnum)
                toapp = loop+'_'+type+'_'+gam
                ret.append(toapp)
    assert len(ret) == len(set(ret)), ret
    #for i in ret:
    #    print(i)
    #sys.exit()
    return ret










if __name__ == '__main__':
    main()
