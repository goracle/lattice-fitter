#!/usr/bin/python3
"""Make the ratio

C_pipi(t)-C_pipi(t+1)/
(
C_pi^2(t)-C_pi^2(t+1)
)
where C_pi is the single pion correlator with the same center of mass

"""
import os
import sys
import re
import copy
import glob
import numpy as np
import h5py
import read_file as rf
from sum_blks import isoproj
from h5jack import TSEP, LT, overall_coeffs, h5sum_blks
from h5jack import avg_irreps, TSTEP, fold_time
from numpy import log, exp
import op_compose as opc
import os.path
from scipy.optimize import minimize_scalar
import h5jack
from h5jack import getwork, gatherdicts
from mpi4py import MPI

TSTEP = int(TSTEP)

DELETE_TSRC = False
DELETE_TSRC = True

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

MOM = [0, 0, 0]
STYPE = 'hdf5'

# ensemble specific hack
DELTAT = 4 if TSTEP == 10 else 3

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def main():
    """Make the ratio

    C_pipi(t)-C_pipi(t+1)/
    (
    C_pi^2(t)-C_pi^2(t+1)
    )
    where C_pi is the single pion correlator with the same center of mass
    usage:
    <pipi correlator (hdf5 file)> <center of mass momentum; magnitude,
    or triple specifying specific 3 momentum (e.g. 0 0 1)>
    """
    if len(sys.argv) == 2:
        ptot = MOM
    elif len(sys.argv) == 3:
        ptotstr = sys.argv[2]+'unit'+('s' if int(sys.argv[2]) != 1 else '')
    elif len(sys.argv) in [5, 6]:
        ptot = np.array([sys.argv[i+2] for i in range(3)])
        dosub = True if len(sys.argv) == 5 else False
    else:
        print("Number of arguments error."+\
              "  Needs input file and COM momentum")
        print("e.g.: test.dat 0 0 1")
        raise
    if len(sys.argv) != 3:
        ptotstr = rf.ptostr(ptot)
    for i in (True, False):
        do_ratio(ptotstr, i)

@PROFILE
def ccosh(en1, tsep):
    return exp(-en1*tsep)+exp(-en1*(LT-tsep))


@PROFILE
def agree(arr1, arr2):
    """Check if measurements
    in array 1 and array 2
    agree within errors
    """
    err1 = np.std(arr1, axis=0)*np.sqrt(len(arr1)-1)
    err2 = np.std(arr2, axis=0)*np.sqrt(len(arr2)-1)
    assert len(arr1) == len(arr2)
    diff = np.abs(np.real(arr1)-np.real(arr2))
    diff = np.mean(diff, axis=0)
    ret = True
    assert err1.shape == err2.shape
    assert diff.shape == err1.shape
    ret = np.all(diff<=2*err1 or diff<=2*err2)
    err = ""
    if not ret:
        err = "disagreement: diff, err1, err2:"+str(
            diff)+","+str(err1)+","+str(err2)
    return ret, err

@PROFILE
def foldt(dt1):
    """Find distance from tsrc in non-mod framework"""
    dt1 = dt1 % LT
    if dt1 > LT/2:
        dt1 = LT-dt1
    return dt1

@PROFILE
def foldpioncorr(corr):
    """Fold the pion correlator about the midpoint
    """
    ret = np.zeros(corr.shape, dtype=np.complex)
    for i in range(LT):
        ret[:, i] = 0.5*(corr[:, i]+corr[:, LT-i-1])
    return ret

@PROFILE
def effparams(corrorig, dt1, dt2=None, tsrc=None):
    """Get effective amplitude and effective mass
    from a given single particle correlator,
    from two different time separations
    """
    corr = np.array(corrorig, dtype=np.complex)
    # corr = foldpioncorr(corrorig)
    np.seterr(over='raise')
    np.seterr(invalid='raise')
    assert LT == len(corr[0])

    # find the tmin and tmax
    dt1 = dt1 % LT
    dt2 = dt2 % LT
    tmod1 = foldt(dt1)
    tmod2 = foldt(dt2)
    tmin = dt1
    tmax = dt2
    if tmod1 > tmod2:
        tmin, tmax = tmax, tmin
    # if distance to tsrc is the same, set all ATW terms to NaN
    if tmod1 == tmod2:
        if dt2 == (LT/2+DELTAT/2) % LT:
            dt2 = LT/2
        elif dt2 == (LT/2 - DELTAT/2) % LT:
            dt2 = LT/2
        elif dt2 == DELTAT/2:
            dt2 = 0
        elif dt2 == (-DELTAT/2) % LT:
            dt2 = 0
        dt2 = int(dt2)
        tmod2 = foldt(dt2)
        tmin = dt1
        tmax = dt2
        if tmod1 > tmod2:
            tmin, tmax = tmax, tmin
        assert dt1 and tmod1 and tmax

    gflag = 0 # 0 if no errors detected
    # take the ratio of the correlator at different time slices
    try:
        rconfig = log(np.real(corr[:, tmin])/np.real(corr[:, tmax]))
    except FloatingPointError:
        #print('floating point error in correlator log.')
        #print("args:")
        for ind, rat in enumerate(np.real(corr[:, tmin])/np.real(
                corr[:, tmax])):
            # if imaginary energy
            # (either numerator or denominator has decayed completely)
            # set amps to NaN
            if rat < 0:
                if tmin == 13 or tmax == 13:
                    print("ratio is less than 0")
                    print(tmin, tmax)
                    print('failing rank:', MPIRANK)
                    assert None
                gflag = 1
        if not gflag:
            rconfig = log(np.real(corr[:, tmin])/np.real(corr[:, tmax]))
        else:
            rconfig = np.zeros(len(corr), dtype=np.complex)+1
    try:
        if tmod2 < LT/4 and tmod1 < LT/4 and tmod1 != tmod2:
            assert rconfig[0] > 0
    except AssertionError:
        print('correlator ratio < 1')
        print(tmin, tmax, corr[0, tmin], corr[0, tmax])
        print(dt1, dt2)
        print(tmod1, tmod2)
        print(tmin, tmax)
        print(tsrc)
        sys.exit(1)

    # lists containing all the configs and time separations
    energies = []
    amps1 = []
    amps2 = []

    for num, ratio in enumerate(rconfig):
        assert ratio
        assert len(amps1) == num
        assert len(amps2) == num
        def func(en1):
            ratio_func = log(ccosh(en1, tmin)/ccosh(en1, tmax))
            ret = (ratio_func-ratio)**2
            return ret

        np.seterr(over='warn')
        np.seterr(invalid='warn')
        minret = minimize_scalar(func)
        np.seterr(over='raise')
        np.seterr(invalid='raise')

        eff_energy = minret.x

        # collect NaN information
        flag = 0 if not gflag else gflag

        if eff_energy < 0 and func(-eff_energy) < 1e-12:
            eff_energy *= -1
        elif eff_energy < 0 and abs(eff_energy) > 1e-8:
            print("negative energy found")
            print(eff_energy)
            print(func(eff_energy))
            print(dt1, dt2)
            flag = 1
        if flag:
            amp1 = np.nan*(1+1j)
            amp2 = np.nan*(1+1j)
        else:
            amp1 = corr[num, tmin]/ccosh(eff_energy, tmin)
            amp2 = corr[num, tmax]/ccosh(eff_energy, tmax)
        assert isinstance(amp1, np.complex), str(
            corr[num,tmin])+" "+str(ccosh(eff_energy,tmin))+" "+str(amp1)
        assert isinstance(amp2, np.complex)
        assert np.all(np.isnan([amp1, amp2])) or eff_energy > 0 or abs(
            eff_energy) < 1e-8
        amps1.append(amp1)
        amps2.append(amp2)
        energies.append(eff_energy)
    amps1 = np.array(amps1)
    amps2 = np.array(amps2)
    energies = np.asarray(energies)
    assert len(amps1) and len(amps2)
    try:
        agreement, errstr = agree(amps1, amps2)
        assert agreement
    except AssertionError:
        if np.mean(energies, axis=0) < 5 and np.mean(
                energies, axis=0) > 1e-8 and not np.isnan(np.mean(amps1))\
                and not np.isnan(np.mean(amps2)):
            print("amplitudes of cosh do not agree:")
            print(errstr)
            print("failing rank:", MPIRANK)
            print("times:", dt1, dt2)
            print(np.mean(amps1, axis=0),
                  np.mean(amps2, axis=0),
                  np.mean(energies, axis=0))
            print("setting amplitudes to NaN")
        amps1 = np.nan*np.zeros(len(amps1), dtype=np.complex)
        amps2 = np.nan*np.zeros(len(amps2), dtype=np.complex)
        energies = np.nan*energies
    amplitudes = np.asarray(amps1)
    return amplitudes, energies

@PROFILE
def atw_transform(pi1, reverseatw=False):
    """Redefine the pion correlators so they propogate in one direction.
    Thus, when they are multiplied
    they give the non-interacting around the world terms
    """
    pi1 = np.asarray(pi1)
    #pi2 = np.asarray(pi2)
    #assert pi1.shape == pi2.shape
    #assert pi1.shape[1] == pi1.shape[2]
    assert pi1.shape[1] == LT
    newpi1 = np.zeros(pi1.shape, np.complex)
    newpi2 = np.zeros(pi1.shape, np.complex)
    for tdis in range(LT):
        zeroit = False
        for tsrc in range(LT):
            if tsrc in skiplist() and DELETE_TSRC:
                continue
            dt2 = tdis+DELTAT if reverseatw else tdis-DELTAT
            #print(tsrc, tdis)
            if zeroit:
                amp1, en1 = (np.nan, np.nan)
            else:
                amp1, en1 = effparams(np.array(
                    pi1[:, tsrc]), tdis, dt2, tsrc)
                if np.any(np.isnan(amp1)):
                    zeroit = True
                    if tdis < LT/2-2*TSEP or tdis > LT/2+2*TSEP:
                        print("nan'ing tdis=", tdis, 'rank=', MPIRANK)
                        assert tdis != 17 or reverseatw, "rank:"+str(MPIRANK)
            #amp2, en2 = effparams(np.array(pi2[:, tsrc]), tdis, dt2, tsrc)
            try:
                newpi1[:, tsrc, tdis] = amp1*exp(-en1*tdis)
                newpi2[:, tsrc, tdis] = amp1*exp(-en1*(LT-tdis))
            except FloatingPointError:
                print("floating point error")
                print('energies', en1[0], en2[0])
                print("tdis:", tdis)
                print("tsrc:", tsrc)
                sys.exit(1)
    return newpi1, newpi2
    #return newpi1, newpi2

def getpi(fname, reverseatw):
    """Get file handle"""
    fn1 = h5py.File(fname, 'r')
    momf = np.asarray(rf.mom(fname))
    skip = False
    if rf.norm2(momf) != 3: # debug purposes only
        pass
        # skip = True
    for i in fn1:
        toppi = np.array(fn1[i])*(2 if 'Chk' in fname else 1)
        save1 = i
    if skip:
        toppi1, toppi2 = (toppi, toppi)
    else:
        toppi1, toppi2 = atw_transform(toppi, reverseatw=reverseatw)
    return (toppi1, toppi2)

def skiplist():
    """Which tsrc to skip, in list form"""
    if int(LT/TSTEP) == LT/TSTEP:
        ltl = [i for i in range(LT) if i % TSTEP]
    else:
        ltl = [i for i in range(LT) if i % TSTEP and LT-i > TSTEP]
    return ltl


def avgtsrc(top):
    """Average over tsrc """
    if DELETE_TSRC:
        assert np.asarray(top).shape[1] == LT
        top1 = np.delete(top, skiplist(), axis=1)
    else:
        top1 = top
    ret = np.mean(top1, axis=1)
    return ret


@PROFILE
def piondirect(atw=False, reverseatw=False):
    """Do pion ratio unsummed."""
    if reverseatw: assert atw
    base = 'pioncorrChk_*_unsummed.jkdat'
    baseglob = glob.glob(base)
    allblks = {}
    count = {}
    numt = None
    atwdict = {}
    if atw:
        splitglob = getwork(list(baseglob))
        for num1, fname in enumerate(splitglob):
            print("rank", MPIRANK, "getting", fname,
                  num1, "/", len(splitglob))
            atwdict[fname] = getpi(fname, reverseatw)
        atwdict = gatherdicts(atwdict)
        print("gather complete")

    for num1, fname in enumerate(baseglob):
        if MPIRANK:
            break
        fn1 = h5py.File(fname, 'r')
        momf = np.asarray(rf.mom(fname))
        numt = len(momf) if numt is None else numt
        assert numt == len(momf), "inconsistent config number"
        for i in fn1:
            toppi = np.array(fn1[i])*(2 if 'Chk' in fname else 1)
            save1 = i
        if atw:
            if fname not in atwdict:
                toppi1, toppi2 = atw_transform(toppi, reverseatw=reverseatw)
                atwdict[fname] = (toppi1, toppi2)
            else:
                toppi1, toppi2 = atwdict[fname]
        for num2, gname in enumerate(baseglob):
            momg = np.asarray(rf.mom(gname))
            assert numt == len(momg), "inconsistent config number"
            gn1 = h5py.File(gname, 'r')
            print(gname)
            for i in gn1:
                bottompi = np.array(gn1[i])*(2 if 'Chk' in gname else 1)
                save2 = i
            if atw:
                if gname not in atwdict:
                    bottompi1,bottompi2 = atw_transform(
                        bottompi, reverseatw=reverseatw)
                    atwdict[gname] = (bottompi1, bottompi2)
                else:
                    bottompi1, bottompi2 = atwdict[gname]

            # roll the tsrc of one of the pions back by TSEP
            if atw:
                shiftpi1 = np.roll(bottompi1, TSEP, axis=1)
                shiftpi2 = np.roll(bottompi2, TSEP, axis=1)
            else:
                shiftpi = np.roll(bottompi, TSEP, axis=1)

            # compute the first topology
            # each topology has two around the world terms, so add in pairs
            if atw:
                top1 = shiftpi1*toppi2+shiftpi2*toppi1
            else:
                top1 = shiftpi*toppi

            # make the tdis the inner pion distance
            top1 = np.roll(top1, -1*TSEP, axis=2)

            # average over tsrc
            top1 = avgtsrc(top1)

            if not atw:
                try:
                    assert top1[0][0] > 100
                except AssertionError:
                    print("top1 is (likely) too small")
                    print(toppi[0,0])
                    print(bottompi[0,0])
                    print(shiftpi[0,0])
                    print('top1[0,0]=', top1[0][0])
                    for i,row in enumerate(top1):
                        for j, col in enumerate(row):
                            print(i,j, col)
                        break
                    sys.exit(1)

            # compute the second topology
            # make one of the pions 2*tsep as long in time
            if atw:
                top2 = np.roll(shiftpi1, -2*TSEP, axis=2)*toppi2+np.roll(
                    shiftpi2, -2*TSEP, axis=2)*toppi1
            else:
                top2 = np.roll(shiftpi, -2*TSEP, axis=2)*toppi

            # average over tsrc
            top2 = avgtsrc(top2)

            # for use in I=1
            top3 = -1*top2

            try:
                assert top2[0][0] > 100, str(top2[0][0])
            except AssertionError:
                print("topology 2 has anomalous size")
                print(top2)
                sys.exit(1)
            momstr = 'sep'+str(TSEP)+'_mom1src'+rf.ptostr(momf)+\
                '_mom2src'+rf.ptostr(momg)+'_mom1snk'
            key1 = addfigd(momstr)+rf.ptostr(momf)
            key3halves = addfigdvec(momstr)+rf.ptostr(momf)
            key2 = addfigd(momstr)+rf.ptostr(momg)
            key3 = addfigdvec(momstr)+rf.ptostr(momg)
            if 'mom1src000_mom2src001_mom1snk001' in key1:
                pass
                #print(fname, gname)
            if 'mom1src000_mom2src001_mom1snk001' in key2:
                pass
                #print(fname, gname, '2')
            if 'mom1src000_mom2src001_mom1snk001' in key1 or\
               'mom1src000_mom2src001_mom1snk001' in key2:
                #print(save1)
                #print(save2)
                temp = np.mean(top2,axis=0)
                temp = np.real(temp)
                temp2 = np.mean(toppi, axis=0)
                temp2 = np.mean(temp2, axis=0)
                temp2= np.real(temp2)
                #print(np.log(temp[5]/temp[6]))
                #print(np.log(temp2[5]/temp2[6]))
                #sys.exit(0)
            assert toppi.shape == bottompi.shape, "shape mismatch"
            if key1 not in allblks:
                allblks[key1] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key1] = 1
            else:
                count[key1] += 1
            if key3halves not in allblks:
                allblks[key3halves] = np.zeros((len(toppi), LT),
                                               dtype=np.complex)
                count[key3halves] = 1
            else:
                count[key3halves] += 1
            if key2 not in allblks:
                allblks[key2] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key2] = 1
            else:
                count[key2] += 1
            if key3 not in allblks:
                allblks[key3] = np.zeros((len(toppi), LT), dtype=np.complex)
                count[key3] = 1
            else:
                count[key3] += 1
            normfactor = 2
            allblks[key1] += top1/normfactor
            allblks[key3halves] += top1/normfactor
            allblks[key2] += top2/normfactor
            allblks[key3] += top3/normfactor

    if not MPIRANK:
        print("Starting write section")
        for i in allblks:
            try:
                assert count[i] == 2
            except AssertionError:
                assert count[i] == 1
                print("topologies used:", count[i])
                print(i)
        temp = opc.op_list(stype=STYPE)
        ocs = overall_coeffs(
            isoproj(False, 0, dlist=list(
                allblks.keys()), stype=STYPE), opc.op_list(stype=STYPE))
        assert isinstance(ocs, dict)
        suffix = '_pisq' if not atw else '_pisq_atw'
        if atw and reverseatw:
            suffix = suffix + 'R'
        for i in list(ocs):
            ocs[i+suffix] = ocs[i]
            del ocs[i]
        h5sum_blks(allblks, ocs, (numt, LT))
        avg_irreps(suffix+'.jkdat')
        print("Finished write section")

@PROFILE
def coeffto1(ocs):
    """Set coefficients to 1"""
    for opa in ocs:
        for i in range(len(ocs[opa])):
            base, coeff = ocs[opa][i]
            ocs[opa][i] = (base, 1.0)
    return ocs

@PROFILE
def aroundworldsubtraction(allblks, deltat=3):
    """Around the world subtraction (by 3)"""
    for blk in allblks:
        t1blk = copy.deepcopy(np.asarray(allblks[blk]))
        t2blk = copy.deepcopy(np.roll(allblks[blk], deltat, axis=1))
        retblk = t1blk-t2blk
        assert retblk[0][8] < 0
        assert retblk[0][9] < 0
        assert retblk[0][10] < 0
        assert retblk[0][11] < 0
        assert retblk[0][11] < 0
        assert abs(retblk[0][11]) < abs(retblk[0][8])
        allblks[blk] = copy.deepcopy(retblk)
    return allblks

@PROFILE
def directratio(allblks, deltat_matrix=3):
    """Take ratio of two directo diagrams
    (1x1 GEVP)
    """
    for blk in allblks:
        temp = copy.deepcopy(np.array(allblks[blk]))
        temp2 = copy.deepcopy(np.array(allblks[blk]))
        assert abs(temp[0][11]) < abs(temp[0][8])
        temp3 = temp[0][8]
        temp = np.roll(temp, deltat_matrix, axis=1)
        assert temp[0][11] == temp3
        assert abs(temp[0][13]) > abs(allblks[blk][0][13])
        assert abs(temp[0][13]) > abs(temp2[0][13])
        allblks[blk] = temp2/temp
        assert abs(allblks[blk][0][13]) < 1
    return allblks

@PROFILE
def addfigdvec(strin):
    """Add figure D vec to string"""
    return 'FigureD_vec_'+strin
 

@PROFILE
def addfigd(strin):
    """Add figure D to string"""
    return 'FigureD_'+strin
            


@PROFILE
def do_ratio(ptotstr, dosub):
    """Make the ratio for a given ptotal string and 
    bool for whether or not to do subtraction in the numerator
    """
    filename = sys.argv[1]+'.jkdat' if len(sys.argv[1].split(
        '.')) == 1 else sys.argv[1]
    fn1 = h5py.File(filename, 'r')
    data, group = anticipate(fn1)
    pionstr = 'pioncorrChk_mom'+ptotstr
    pionstr = 'pioncorrChk_mom'+ptotstr
    print("using pion correlator:", pionstr)
    gn1 = h5py.File(pionstr+'.jkdat', 'r')
    pion = np.array(gn1[pionstr])
    multsub = 1.0 if dosub else 0.0
    C_pipiminus = np.array(
        [
            [
                data[config][time]-data[config][time+1]*multsub
                for time in range(len(data[0])-1)
            ]
            for config in range(len(data))
        ]
    )
    pionminus = np.array(
        [
            [
                (pion[config][time])**2-(pion[config][time+1])**2
                for time in range(len(data[0])-1)
            ]
            for config in range(len(data))
        ]
    )
    filesplit = filename.split('.')
    ext = '.jkdat'
    suffix = ext if dosub else '_nosub'+ext
    hn1 = h5py.File(filesplit[0]+'_pionratio'+suffix, 'w')
    hn1[filesplit[0]+'_pionratio'] = C_pipiminus/pionminus
    hn1.close()
    print("Finished with", sys.argv[1])
    return


@PROFILE
def anticipate(fn):
    """Anticipate the hdf5 structure: one dataset in one group"""
    group = ''
    for i in fn:
        group = i
    data = ''
    for i in fn[group]:
        data = i
    return np.array(fn[group+'/'+i]), group

PIONCORRS = ['pioncorrChk_mom000.jkdat',
             'pioncorrChk_p1.jkdat',
             'pioncorrChk_p11.jkdat',
             'pioncorrChk_p111.jkdat']

for i in range(len(PIONCORRS)):
    assert os.path.isfile(PIONCORRS[i])
    DATAN = re.sub('.jkdat', '', PIONCORRS[i])
    assert h5py.File(PIONCORRS[i], 'r')
    tostore = np.array(h5py.File(PIONCORRS[i], 'r')[DATAN])
    assert isinstance(PIONCORRS, list)
    PIONCORRS[i] = tostore

@PROFILE
def jkdatrm(fstr):
    """Remove .jkdat from the end of a filename
    since the datasets are traditionally just the file name
    """
    fstr = re.sub('.jkdat', '', fstr)
    return fstr

@PROFILE
def datasetname(filen):
    """Build the dataset name from the parent directory
    """
    parentdir = os.path.abspath(os.path.join('.', os.pardir))
    parentdir = parentdir.split('/')[-1]
    ret = parentdir + '/' + filen
    ret = jkdatrm(ret)
    return ret

@PROFILE
def divide_multiply(tplat=10):
    """Divide the diagonal elements by the result of piondirect()
    then multiply by the asymptotic pion correlator squared
    which is assumed plateaued at t=tplat
    """
    base = '*_pisq.jkdat'
    baseglob = glob.glob(base)
    for fn1 in baseglob:
        numerator_part1 = re.sub('_pisq', '', fn1)
        try:
            numerator_part1 = np.asarray(h5py.File(numerator_part1, 'r')[
                datasetname(numerator_part1)])
        except KeyError:
            print(datasetname(numerator_part1))
            sys.exit(1)
        assert numerator_part1

        denom = np.asarray(h5py.File(fn1, 'r')[datasetname(fn1)])
        assert denom

        # momentum magnitude, find the pion correlator
        mommag = rf.norm2(rf.mom(numerator_part2))
        numerator_part2 = PIONCORRS[i][:, tplat]
        num = numerator_part1*numerator_part2
        assert num

        # write
        writestring = re.sub('.jkdat', '_pionratio.jkdat', fn1)
        if os.path.isfile(writestring):
            continue
        gn1 = h5py.File(writestring, 'w')
        gn1[datasetname(writestring)] = num/denom
        gn1.close()


if __name__ == '__main__':
    print("start")
    h5jack.AVGTSRC = True # hack to get file names right.
    piondirect(atw=True)
    print("after atw, rank", MPIRANK)
    piondirect(atw=True, reverseatw=True)
    print("after reverse atw", MPIRANK)
    piondirect()
    print("end")
    for dirn in ['I0', 'I1', 'I2']:
        os.chdir(dirn)
        print(os.getcwd())
        sys.exit(0)
        divide_multiply()
        os.chdir('..')
