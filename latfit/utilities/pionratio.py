#!/usr/bin/python3
"""Make the ratio

C_pipi(t)-C_pipi(t+1)/
(
C_pi^2(t)-C_pi^2(t+1)
)
where C_pi is the single pion correlator with the same center of mass

"""
import os
import os.path
import sys
import re
import copy
import glob
from numpy import log, exp
import numpy as np
from scipy.optimize import minimize_scalar
import h5py
from mpi4py import MPI

from lafit.utilities.postprod.h5jack import getwork, gatherdicts, check_ids
from lafit.utilities.postprod.h5jack import TSEP, LT, overall_coeffs,\
    h5sum_blks
from lafit.utilities.postprod.h5jack import avg_irreps, TSTEP
from lafit.utilities.postprod.h5jack import tdismax
import lafit.utilities.postprod.h5jack as h5jack
import op_compose as opc
from latfit.utilities import exactmean as em
import read_file as rf
from sum_blks import isoproj

TSTEP = int(TSTEP)

# do not change, delete_tsrc should be true
DELETE_TSRC = False
DELETE_TSRC = True

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()

MOM = [0, 0, 0]
STYPE = 'hdf5'

# ensemble specific hack
# DELTAT is T-T0 where T, T0 are RHS, LHS time separations
# used in the GEVP
DELTAT = 2 if TSTEP == 10 else 3
print("Using DELTAT=", DELTAT)

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
        # dosub = True if len(sys.argv) == 5 else False
    else:
        print("Number of arguments error."+\
              "  Needs input file and COM momentum")
        print("e.g.: test.dat 0 0 1")
        assert None
    if len(sys.argv) != 3:
        ptotstr = rf.ptostr(ptot)
    for i in (True, False):
        do_ratio(ptotstr, i)

@PROFILE
def ccosh(en1, tsep):
    """Cosh, but lattice version."""
    return exp(-en1*tsep)+exp(-en1*(LT-tsep))


@PROFILE
def agree(arr1, arr2):
    """Check if measurements
    in array 1 and array 2
    agree within errors
    """
    err1 = em.acstd(arr1, axis=0)*np.sqrt(len(arr1)-1)
    err2 = em.acstd(arr2, axis=0)*np.sqrt(len(arr2)-1)
    assert len(arr1) == len(arr2)
    diff = np.abs(np.real(arr1)-np.real(arr2))
    diff = em.acmean(diff, axis=0)
    ret = True
    assert err1.shape == err2.shape
    assert diff.shape == err1.shape
    ret = np.all(diff <= 2*err1 or diff <= 2*err2)
    err = ""
    if not ret:
        err = "disagreement: diff, err1, err2:"+str(
            diff)+", "+str(err1)+", "+str(err2)
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
    assert None, "should not be used (use the h5jack version)"
    ret = np.zeros(corr.shape, dtype=np.complex)
    for i in range(LT):
        ret[:, i] = 0.5*(corr[:, i]+corr[:, LT-i-1])
    return ret

def tmintmax():
    """Find tmin and tmax"""

def proc_t(dt1, dt2):
    """Process dt1, dt2"""
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
    t_test = tmod2 < LT/6 and tmod1 < LT/6 and tmod1 != tmod2
    return tmin, tmax, t_test

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

    tmin, tmax, t_test = proc_t(dt1, dt2)

    gflag = 0 # 0 if no errors detected
    # take the ratio of the correlator at different time slices
    try:
        rconfig = log(np.real(corr[:, tmin])/np.real(corr[:, tmax]))
    except FloatingPointError:
        #print('floating point error in correlator log.')
        #print("args:")
        for _, rat in enumerate(np.real(corr[:, tmin])/np.real(
                corr[:, tmax])):
            # if imaginary energy
            # (either numerator or denominator has decayed completely)
            # set amps to NaN
            if rat < 0:
                if tmin == int(13*DELTAT/4) or tmax == int(13*DELTAT/4):
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
        if t_test:
            assert rconfig[0] > 0
    except AssertionError:
        print('correlator ratio < 1')
        print(tmin, tmax, corr[0, tmin], corr[0, tmax])
        print(dt1, dt2)
        #print(tmod1, tmod2)
        print(tmin, tmax)
        print(tsrc)
        sys.exit(1)
    return amps_energies(rconfig, tmin, tmax, corr, gflag)

def eff_energy(ratio, tmin, tmax, gflag):
    """ Find effective energy from ratio of correlation function
    at two separate time slices
    """
    def func(en1, ratio_inner=ratio):
        """Function to minimize"""
        ratio_func = log(ccosh(en1, tmin)/ccosh(en1, tmax))
        ret = (ratio_func-ratio_inner)**2
        return ret

    np.seterr(over='warn')
    np.seterr(invalid='warn')
    minret = minimize_scalar(func)
    np.seterr(over='raise')
    np.seterr(invalid='raise')
    effenergy = minret.x

    # collect NaN information
    flag = 0 if not gflag else gflag

    if effenergy < 0 and func(-effenergy) < 1e-12:
        effenergy *= -1
    elif eff_energy < 0 and abs(effenergy) > 1e-8:
        print("negative energy found")
        print(effenergy)
        print(func(effenergy))
        # print(dt1, dt2)
        flag = 1

    return effenergy, flag

def append_energy_amps(loop, ttup, corr, gflag):
    """Add to the final amplitude and energy arrays"""

    num, ratio = loop
    tmin, tmax = ttup
    energies = []
    amps1 = []
    amps2 = []

    assert ratio
    assert len(amps1) == num
    assert len(amps2) == num

    effenergy, flag = eff_energy(ratio, tmin, tmax, gflag)

    if flag:
        amp1 = np.nan*(1+1j)
        amp2 = np.nan*(1+1j)
    else:
        amp1 = corr[num, tmin]/ccosh(effenergy, tmin)
        amp2 = corr[num, tmax]/ccosh(effenergy, tmax)
    assert isinstance(amp1, np.complex), str(
        corr[num, tmin])+" "+str(ccosh(effenergy, tmin))+" "+str(amp1)
    assert isinstance(amp2, np.complex)
    assert np.all(np.isnan([amp1, amp2])) or effenergy > 0 or abs(
        effenergy) < 1e-8
    amps1.append(amp1)
    amps2.append(amp2)
    energies.append(effenergy)
    #print('effenergy', effenergy, 'amp1', amp1, 'amp2', amp2)
    return amps1, amps2, energies


def amps_energies(rconfig, tmin, tmax, corr, gflag):
    """Find effective amplitudes and energies"""
    # lists containing all the configs and time separations
    for num, ratio in enumerate(rconfig):
        amps1, amps2, energies = append_energy_amps(
            (num, ratio), (tmin, tmax), corr, gflag)
    amps1 = np.array(amps1)
    amps2 = np.array(amps2)
    energies = np.asarray(energies)
    assert amps1 and amps2, "length amps1: "+str(len(amps1))+\
        " length amps2: "+str(len(amps2))
    try:
        agreement, errstr = agree(amps1, amps2)
        assert agreement
    except AssertionError:
        if em.acmean(energies, axis=0) < 5 and em.acmean(
                energies, axis=0) > 1e-8 and not np.isnan(em.acmean(amps1))\
                and not np.isnan(em.acmean(amps2)):
            print("amplitudes of cosh do not agree:")
            print(errstr)
            print("failing rank:", MPIRANK)
            # print("times:", dt1, dt2)
            print(em.acmean(amps1, axis=0),
                  em.acmean(amps2, axis=0),
                  em.acmean(energies, axis=0))
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
    slist = skiplist()
    for tdis in range(LT):
        zeroit = False
        if tdis > tdismax()+2*TSEP:
            continue
        for tsrc in range(LT):
            if tsrc in slist and tsrc+TSEP in slist and DELETE_TSRC:
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
                        assert tdis != int(
                            17*DELTAT/4) or reverseatw, "rank:"+str(MPIRANK)
            #amp2, en2 = effparams(np.array(pi2[:, tsrc]), tdis, dt2, tsrc)
            try:
                newpi1[:, tsrc, tdis], newpi2[:, tsrc, tdis] = morecorr(
                    amp1, en1, tdis, dt2, pi1[:, tsrc, :])
            except FloatingPointError:
                print("floating point error")
                # print('energies', en1[0], en2[0])
                print("tdis:", tdis)
                print("tsrc:", tsrc)
                sys.exit(1)
    return newpi1, newpi2
    #return newpi1, newpi2

def simpledivide(arr, pow1):
    """
    We have A*exp(-Et), and A*exp(-E*dt2)
    We want A*exp(-E*(LT-t))
    We can get this by forming
    A*exp(-Et)*[

    A*exp(-Et)/(A*exp(-E*dt2))

    ]**[(LT-t)/t/(t-dt2)]

    == A*exp(-E*(LT-t))

    The hypothesis is that this is more correlated than eff mass method for
    early times.
    """
    ret = np.copy(arr)
    ret = np.real(ret)
    print(pow1)
    print(arr)
    sys.exit(0)
    for i in range(LT):
        if i:
            try:
                ret[:, i] = arr[:, i]**((LT-i)/pow1/i)
            except FloatingPointError:
                ret[:, i] = np.nan
    return ret


def morecorr(amp, expo, tdis, dt2, corr):
    """Return pions which are more correlated
    """
    pow1 = tdis-dt2
    corr = np.real(corr)
    corr_ancillary = np.copy(np.roll(corr, pow1, axis=1))
    if tdis < LT/4 and tdis > pow1 and False:
        newpi1 = np.copy(corr)[:, tdis]
        newpi2 = np.copy(corr*simpledivide(corr/corr_ancillary, pow1))
        newpi2 = newpi2[:, tdis]
        if tdis < LT/2:
            assert newpi2[0] < newpi1[0], str(newpi2)+" "+str(newpi1)
    else:
        newpi1 = amp*exp(-expo*tdis)
        newpi2 = amp*exp(-expo*(LT-tdis))
    return newpi1, newpi2

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
    #    save1 = i
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
        ltl = [i for i in range(LT) if i % TSTEP or LT-i < TSTEP]
    return ltl

def innerouter(top1pair, top2pair, mompair):
    """Check to make sure inner pions have energy >= outer pions
    (Luchang's condition)
    """
    top1, mom1snk1 = top1pair
    top2, mom1snk2 = top2pair
    mom1src, mom2src = mompair

    mom1snk1 = np.asarray(rf.mom(mom1snk1))
    mom1snk2 = np.asarray(rf.mom(mom1snk1))
    mom1src = np.asarray(rf.mom(mom1src))
    mom2src = np.asarray(rf.mom(mom2src))

    assert np.all(mom1snk1 == mom2src)
    assert np.all(mom1snk2 == mom1src)
    assert len(top1.shape) == 2
    assert len(top2.shape) == 2
    assert len(mom1snk1) == len(mom1snk2)
    assert len(mom1snk1) == 3
    if rf.norm2(mom1src) < rf.norm2(mom2src):
        top1 *= np.nan
        top2 *= np.nan
    elif rf.norm2(mom1src) > rf.norm2(mom2src):
        top1 *= np.nan
    ret = (top1, top2)
    return ret


def avgtsrc(top):
    """Average over tsrc """
    if DELETE_TSRC:
        assert np.asarray(top).shape[1] == LT
        top1 = np.delete(top, skiplist(), axis=1)
    else:
        top1 = top
    ret = em.acmean(top1, axis=1)
    return ret

def zerosimple(blk):
    """Zero out the correlator beyond
    the maximum tdis
    (simplified)
    """
    for tdis in range(LT):
        if tdis > tdismax():
            blk[:, tdis] = 0*(1+1j)
    return blk


def zerotdis(blk, atw=False):
    """Zero out the correlator beyond
    the maximum tdis
    """
    blk = np.asarray(blk)
    assert len(blk.shape) == 2
    assert len(blk[0]) == LT
    for tdis in range(LT):
        if not atw:
            break
        if tdis + TSEP >= LT or (
                tdis-TSEP <= tdismax() and tdis > tdismax()):
            # tdis time slices get rolled back in top1
            # so we must explicitly zero these terms
            # we can't skip these because we need them for top2
            blk[:, tdis] = 0*(1+1j)
        elif tdis > tdismax():
            assert not np.any(blk[:, tdis]) or np.all(np.isnan(blk))\
                or np.all(np.isnan(blk[:, tdis])), \
                "tdis="+str(tdis)+" "+str(blk[:, tdis])+" "+str(blk)
        else:
            assert np.all(blk[:, tdis]) or np.all(np.isnan(blk))\
                or np.all(np.isnan(blk[:, tdis])), \
                "tdis="+str(tdis)+" "+str(blk[:, tdis])+" "+str(blk)
    blk = zerosimple(blk)
    return blk

def top_pi(fname, atw, reverseatw, atwdict, numt):
    """Get top pion (according to some diagram representation),
    and its momentum"""
    fn1 = h5py.File(fname, 'r')
    momf = np.asarray(rf.mom(fname))
    numt = len(momf) if numt is None else numt
    assert numt == len(momf), "inconsistent config number"
    for i in fn1:
        toppi = np.array(fn1[i])*(2 if 'Chk' in fname else 1)
        # save1 = i
    if atw:
        if fname not in atwdict:
            toppi1, toppi2 = atw_transform(toppi, reverseatw=reverseatw)
            atwdict[fname] = (toppi1, toppi2)
            toppi = (toppi1, toppi2)
        else:
            toppi1, toppi2 = atwdict[fname]
    return toppi, numt

def shift_pi(gname, atw, reverseatw, atwdict, numt):
    """Get the bottom pion and shift the tsrc"""
    gn1 = h5py.File(gname, 'r')
    momg = np.asarray(rf.mom(gname))
    assert numt == len(momg), "inconsistent config number"
    #print(gname)
    for i in gn1:
        bottompi = np.array(gn1[i])*(2 if 'Chk' in gname else 1)
    #    save2 = i
    if atw:
        if gname not in atwdict:
            bottompi1, bottompi2 = atw_transform(
                bottompi, reverseatw=reverseatw)
            atwdict[gname] = (bottompi1, bottompi2)
        else:
            bottompi1, bottompi2 = atwdict[gname]

    # roll the tsrc of one of the pions back by TSEP
    if atw:
        shiftpi1 = np.roll(bottompi1, TSEP, axis=1)
        shiftpi2 = np.roll(bottompi2, TSEP, axis=1)
        shiftpi = (shiftpi1, shiftpi2)
    else:
        shiftpi = np.roll(bottompi, TSEP, axis=1)
    return shiftpi, bottompi

def top_one(atw, pi_tuple, fname, gname):
    """Compute the first topology
    each topology has two around the world terms, so add in pairs
    """

    # unpack pion correlation functions
    toppi, _, shiftpi = pi_tuple
    if atw:
        shiftpi1, shiftpi2 = shiftpi
        toppi1, toppi2 = toppi

    if atw:
        if np.all(rf.mom(gname) == rf.mom(fname)):
            top1 = shiftpi1*toppi2
        else:
            top1 = shiftpi1*toppi2+shiftpi2*toppi1
    else:
        top1 = shiftpi*toppi

    # make the tdis the inner pion distance
    top1 = np.roll(top1, -1*TSEP, axis=2)

    # average over tsrc
    top1 = avgtsrc(top1)

    if not atw:
        try:
            assert top1[0][0] > 100 or np.isnan(
                top1[0][0]), top1[0][0]
        except AssertionError:
            print("top1 is (likely) too small")
            #print(toppi[0, 0])
            #print(bottompi[0, 0])
            #print(shiftpi[0, 0])
            debug_print_top(top1)
            print('top1[0, 0]=', top1[0][0])
            sys.exit(1)
    return top1

def debug_print_top(top):
    """For debug purposes, print topology"""
    for i, row in enumerate(top):
        for j, col in enumerate(row):
            print(i, j, col)
        break


def top_two(atw, pi_tuple, fname, gname):
    """compute the second topology
    make one of the pions 2*tsep as long in time
    """
    momf = np.asarray(rf.mom(fname))
    momg = np.asarray(rf.mom(gname))

    # unpack pion correlation functions
    toppi, _, shiftpi = pi_tuple
    if atw:
        shiftpi1, shiftpi2 = shiftpi
        toppi1, toppi2 = toppi

    if atw:
        if np.all(momf == momg):
            top2 = np.roll(shiftpi1, -2*TSEP, axis=2)*toppi2
        else:
            top2 = np.roll(shiftpi1, -2*TSEP, axis=2)*toppi2+np.roll(
                shiftpi2, -2*TSEP, axis=2)*toppi1
    else:
        top2 = np.roll(shiftpi, -2*TSEP, axis=2)*toppi

    # average over tsrc
    top2 = avgtsrc(top2)

    if not atw:
        try:
            assert top2[0][0] > 100 or np.isnan(
                top2[0][0]), str(top2[0][0])
        except AssertionError:
            print("topology 2 has anomalous size (momg, momf)=",
                  gname, fname)
            debug_print_top(top2)
            print(top2)
            sys.exit(1)
    return top2

def top_keys(fname, gname):
    """Get topology keys"""

    momf = np.asarray(rf.mom(fname))
    momg = np.asarray(rf.mom(gname))

    momstr = 'sep'+str(TSEP)+'_mom1src'+rf.ptostr(momf)+\
        '_mom2src'+rf.ptostr(momg)+'_mom1snk'
    key1 = addfigd(momstr)+rf.ptostr(momg)
    key3halves = addfigdvec(momstr)+rf.ptostr(momg)
    key2 = addfigd(momstr)+rf.ptostr(momf)
    key3 = addfigdvec(momstr)+rf.ptostr(momf)
    ret = (key1, key3halves, key2, key3)
    if 'mom1src000_mom2src001_mom1snk001' in key1:
        pass
        #print(fname, gname)
    if 'mom1src000_mom2src001_mom1snk001' in key2:
        pass
        #print(fname, gname, '2')
    if 'mom1src000_mom2src001_mom1snk001' in key1 or\
        'mom1src000_mom2src001_mom1snk001' in key2:
        pass
        #print(save1)
        #print(save2)
        #print(np.log(temp[5]/temp[6]))
        #print(np.log(temp2[5]/temp2[6]))
        #sys.exit(0)
    return ret

def zero_out_and_count(allblks, count, keys, toppi):
    """Zero out the output container and count the topologies added together
    """
    key1, key3halves, key2, key3 = keys
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
    return allblks, count

def add_topologies(allblks, top1, top2, top3, keys):
    """Add topologies together and save"""
    key1, key3halves, key2, key3 = keys
    normfactor = 2
    allblks[key1] += top1/normfactor
    allblks[key3halves] += top1/normfactor
    allblks[key2] += top2/normfactor
    allblks[key3] += top3/normfactor
    return allblks

def getatwdict(atw, reverseatw, baseglob):
    """Get and save around the world pieces
    """
    atwdict = {}
    if atw:
        splitglob = getwork(list(baseglob))
        for num1, fname in enumerate(splitglob):
            print("rank", MPIRANK, "getting", fname,
                  num1, "/", len(splitglob))
            atwdict[fname] = getpi(fname, reverseatw)
        atwdict = gatherdicts(atwdict)
        print("gather complete")
    return atwdict

def get_topologies(atwdict, fgnames, atw_bools, topret):
    """Get three topologies of pion * pion"""
    toppi, numt = topret
    gname, fname = fgnames
    atw, reverseatw = atw_bools
    shiftpi, bottompi = shift_pi(gname, atw, reverseatw,
                                 atwdict, numt)
    pi_tuple = (toppi, bottompi, shiftpi)
    top1 = top_one(atw, pi_tuple, fname, gname)
    top2 = top_two(atw, pi_tuple, fname, gname)
    top1, top2 = innerouter((top1, gname), (top2, fname),
                            (fname, gname))
    # for use in I=1
    # top3 = -1*top2

    assert np.asarray(toppi).shape == bottompi.shape,\
        "shape mismatch"
    return top1, top2, -1*top2, numt

@PROFILE
def piondirect(atw=False, reverseatw=False):
    """Do pion ratio unsummed."""
    if reverseatw:
        assert atw
    baseglob = glob.glob('pioncorrChk_*_unsummed.jkdat')
    assert baseglob, "unsummed diagrams missing"
    allblks = {}
    count = {}
    numt = None
    atwdict = getatwdict(atw, reverseatw, baseglob)
    for _, fname in enumerate(baseglob):
        if MPIRANK:
            break
        toppi, numt = top_pi(fname, atw, atwdict, numt)

        for _, gname in enumerate(baseglob):

            top1, top2, top3, numt = get_topologies(atwdict,
                                                    (gname, fname),
                                                    (atw, reverseatw),
                                                    (toppi, numt))
            keys = top_keys(fname, gname)
            allblks, count = zero_out_and_count(allblks, count, keys, toppi)
            allblks = add_topologies(allblks, top1, top2, top3, keys)
    pionratiowrite(allblks, count, atw, reverseatw, numt)

def pionratiowrite(allblks, count, atw, reverseatw, numt):
    """Write section"""
    if not MPIRANK:
        print("Starting write section")
        for i in allblks:
            try:
                assert count[i] == 2
            except AssertionError:
                assert count[i] == 1
                #print("topologies used:", count[i])
                #print(i)
            allblks[i] = zerotdis(allblks[i], atw=atw)
            if 'mom1src000_mom2src000_mom1snk000' in i:
                assert count[i] == 2
                #for time, val in enumerate(allblks[i][0]):
                #    print(time, val)
                #sys.exit(0)
        # temp = opc.op_list(stype=STYPE)
        ocs = overall_coeffs(
            isoproj(False, 0, dlist=list(
                allblks.keys()), stype=STYPE), opc.op_list(stype=STYPE))
        assert isinstance(ocs, dict)
        suffix = '_pisq' if not atw else '_pisq_atw'
        if atw and reverseatw:
            suffix = suffix + 'R'
        if atw:
            suffix = suffix + '_dt'+str(DELTAT)
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
            base, _ = ocs[opa][i]
            ocs[opa][i] = (base, 1.0)
    return ocs

@PROFILE
def aroundworldsubtraction(allblks, deltat=3):
    """Around the world subtraction (by 3)"""
    assert None, "not supported"
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
    data = anticipate(fn1)
    pionstr = 'pioncorrChk_mom'+ptotstr
    pionstr = 'pioncorrChk_mom'+ptotstr
    print("using pion correlator:", pionstr)
    gn1 = h5py.File(pionstr+'.jkdat', 'r')
    pion = np.array(gn1[pionstr])
    multsub = 1.0 if dosub else 0.0
    c_pipiminus = np.array(
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
    hn1[filesplit[0]+'_pionratio'] = c_pipiminus/pionminus
    hn1.close()
    print("Finished with", sys.argv[1])
    return


@PROFILE
def anticipate(fn1):
    """Anticipate the hdf5 structure: one dataset in one group"""
    group = ''
    for i in fn1:
        group = i
    data = ''
    for i in fn1[group]:
        data = i
    return np.array(fn1[group+'/'+data])

PIONCORRS = ['pioncorrChk_mom000.jkdat',
             'pioncorrChk_p1.jkdat',
             'pioncorrChk_p11.jkdat',
             'pioncorrChk_p111.jkdat']

for INDEX, _ in enumerate(PIONCORRS):
    if not h5jack.FREEFIELD and '__name__' == '__main__':
        assert os.path.isfile(PIONCORRS[INDEX])
        DATAN = re.sub('.jkdat', '', PIONCORRS[INDEX])
        assert h5py.File(PIONCORRS[INDEX], 'r')
        tostore = np.array(h5py.File(PIONCORRS[INDEX], 'r')[DATAN])
        assert isinstance(PIONCORRS, list)
        PIONCORRS[INDEX] = tostore

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
def divide_multiply(_=10):
    """Divide the diagonal elements by the result of piondirect()
    then multiply by the asymptotic pion correlator squared
    which is assumed plateaued at t=tplat
    """
    assert None, "old version, no longer supported"
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
        # mommag = rf.norm2(rf.mom(numerator_part2))
        #numerator_part2 = PIONCORRS[0][:, tplat]
        numerator_part2 = 1
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
    check_ids()
    h5jack.AVGTSRC = True # hack to get file names right.
    h5jack.WRITE_INDIVIDUAL = False # hack to get file names right.
    piondirect()
    print("after pion ratio")
    piondirect(atw=True)
    print("after atw, rank", MPIRANK)
    piondirect(atw=True, reverseatw=True)
    print("after reverse atw", MPIRANK)
    print("end")
    for dirn in ['I0', 'I1', 'I2']:
        os.chdir(dirn)
        print(os.getcwd())
        sys.exit(0)
        divide_multiply()
        os.chdir('..')
