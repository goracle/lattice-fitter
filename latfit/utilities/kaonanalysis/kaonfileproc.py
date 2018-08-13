"""Process kaon files"""

from h5jack import LT as LT_CHECK
import projectkop
import kaonpostproc as kpp
import kaondecompose

print("Imported kaonfileproc with LT=", LT_CHECK)

def genKey(momdiag):
    """Get operator output key"""
    kpitsep = deltat(momdiag)
    mom = rf.mom(momdiag)
    # keyirr = str(np.dot(mom, mom))+'@'+str(kpitsep)
    # keyirr = str(np.dot(mom, mom))+'@'+str(kpitsep)
    return keyirr

def deltat(momdiag):
    """find k-pi tsep"""
    return int(re.search('deltat_(\d)+_').group(1))


def procSigmaType23(type23, trajl, otype):
    """Process type 23 sigma diagrams into Q_i pieces"""
    ltraj = len(trajl)
    type12 = False if '3' in otype else True
    ncontract = 4 if type12 else 8
    for num, traj in enumerate(trajl):
        for momdiag in type23:

            t1arr = np.asarray(h5py.File('traj_'+traj+'_'+momdiag, 'r'))

            # do some checks
            checkarr(t1arr, type12)

            # decompose into pieces, optionally average over tK, type3 has tstep1
            t1arr = kaondecompose.decompose(t1arr, ncontract, True, 1 if '3' in otype else TSTEP12)

            # our operators are momentum irreps, so do the projection now onto A1
            keyirr = genKey(momdiag)
            assert rf.norm2(rf.mom(momdiag)) == 0, "sigma momentum is fixed to be 0"

            ret = None
            for i in np.arange(1, 11):
                elif otype == 'type2':
                    kpp.QOP_sigma[str(i)][keyirr][num] +=  projectkop.QiprojSigmaType2(t1arr, i, 'I0')
                    kpp.QOP_sigma[str(i)][keyirr][num] +=  projectkop.QiprojSigmaType2(t1arr, i, 'I2')
                elif otype == 'type3':
                    kpp.QOP_sigma[str(i)][keyirr][num] +=  projectkop.QiprojSigmaType3(t1arr, i, 'I0')
                    kpp.QOP_sigma[str(i)][keyirr][num] +=  projectkop.QiprojSigmaType3(t1arr, i, 'I2')
    return ret

def proctype123(type123, trajl, otype):
    """Process type 1 diagrams into Q_i pieces"""
    type12 = False if '3' in otype else True
    ncontract = 4 if type12 else 8
    ltraj = len(trajl)
    for num, traj in enumerate(trajl):
        for momdiag in type123:

            t1arr = np.asarray(h5py.File('traj_'+traj+'_'+momdiag, 'r'))

            # do some checks
            checkarr(t1arr, type12)

            # decompose into pieces, optionally average over tK, type3 has tstep1
            t1arr = kaondecompose.decompose(t1arr, ncontract, True, 1 if '3' in otype else TSTEP12)

            # our operators are momentum irreps, so do the projection now onto A1
            keyirr = genKey(momdiag)
            ret = None
            for i in np.arange(1, 11):
                if otype == 'type1':
                    kpp.QOPI0[str(i)][keyirr][num] +=  projectkop.QiprojType1(t1arr, i, 'I0')
                    kpp.QOPI2[str(i)][keyirr][num] +=  projectkop.QiprojType1(t1arr, i, 'I2')
                elif otype == 'type2':
                    kpp.QOPI0[str(i)][keyirr][num] +=  projectkop.QiprojType2(t1arr, i, 'I0')
                    kpp.QOPI2[str(i)][keyirr][num] +=  projectkop.QiprojType2(t1arr, i, 'I2')
                elif otype == 'type3':
                    kpp.QOPI0[str(i)][keyirr][num] +=  projectkop.QiprojType3(t1arr, i, 'I0')
                    kpp.QOPI2[str(i)][keyirr][num] +=  projectkop.QiprojType3(t1arr, i, 'I2')
                elif otype == 'mix3':
                    ret = projectkop.QiprojType3(t1arr, i, 'I0')
    return ret

def proctype4(type4, trajl, avgtk=False):
    """Process type 4"""
    ltraj = len(trajl)
    shape = (ltraj, 8, 4, LT_CHECK, LT_CHECK)
    type4dict = defaultdict(lambda: np.zeros(shape, dtype=np.complex), type4dict)
    for num, traj in enumerate(trajl):
        for momdiag in type4:

            # do some checks
            t1arr = np.asarray(h5py.File('traj_'+traj+'_'+momdiag, 'r'))
            assert shape[1:] == (8, 4, Lt, Lt), "Bad shape for type 4"

            # decompose into pieces, optionally average over tK, tstep=1
            type4dict[momdiag][num] = kaondecompose.decompose(t1arr, 8, avgtk, 1)

    return type4dict

def procmix(mix, trajl, tkavg=False):
    """processes the mix4 diagrams"""
    ltraj = len(trajl)
    shape = (ltraj, 2, LT_CHECK, LT_CHECK)
    mixdict = defaultdict(lambda: np.zeros(shape, dtype=np.complex), mixdict)
    for num, traj in enumerate(trajl):
        for momdiag in mix:

            # do some checks
            t1arr = np.asarray(h5py.File('traj_'+traj+'_'+momdiag, 'r'))
            checkarr(t1arr, False, True)
            assert shape[1:] == (2, Lt, Lt), "Bad shape for type 4"

            # decompose into pieces, optionally average over tk, tstep=1
            mixdict[momdiag][num] = kaondecompose.decomposeMix(t1arr, tkavg, 1)

    return mixdict


def checkarr(t1arr, type12=False, mix=False):
    """Some asserts on the k->x arrays"""
    assert len(t1arr.shape) == 1, "Should be 1d array"
    size = len(t1arr)
    ncontract = None
    if mix:
        assert 2*LT_CHECK**2 == size, "Bad mix array size:"+str(size)
    else:
        ncontract = 4 if type12 else 8
        assert LT_CHECK**2*ncontract*4 == size, "Bad results array size:"+str(size)
    return size



