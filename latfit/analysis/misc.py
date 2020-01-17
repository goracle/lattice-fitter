"""Single particle energies/dispersive energies"""
import sys
from math import sqrt, pi
import pickle
import numpy as np
from latfit.utilities.op_compose import freemomenta
import latfit.utilities.read_file as rf
from latfit.utilities import exactmean as em
from latfit.analysis.errorcodes import BoolThrowErr

BOX_LENGTH = np.nan
MASS = np.nan
LATTICE = None
PIONRATIO = BoolThrowErr()
IRREP = None
CONTINUUM = BoolThrowErr()


def dummy(*x):
    """dummy function.  does nothing"""
    assert None, "safety switch"
    return x
BINOUT = dummy
HALFTOTAL = dummy
ELIM_JKCONF_LIST = [np.nan]

def elim_jkconfigs(ret, elimlist):
    """dummy version"""
    assert None
    if ret or elimlist:
        pass

def massfunc():
    """Return the pion mass"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    pionstr = '_pioncorrChk_'
    if massfunc.MASS is None:
        try:
            fn1 = open('energy_min_'+LATTICE+pionstr+'mom000.jkdat.p', 'rb')
            massfunc.MASS = pickle.load(fn1)
            print("load of jackknifed mass successful")
            massfunc.MASS = massfunc.MASS.flatten()
            massfunc.MASS = select_subset(massfunc.MASS)
        except FileNotFoundError:
            print("jackknifed mass not found")
            massfunc.MASS = MASS
            raise
    return massfunc.MASS
massfunc.MASS = None

def update_binhalf():
    """after we set these functions update (the mass)"""
    massfunc.MASS = select_subset(massfunc.MASS, elimlist=[])


def p1f():
    """E_pi(|p|=1)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    pionstr = '_pioncorrChk_'
    if LATTICE == '24c':
        ret = 0.29641
    elif LATTICE == '32c':
        ret = 0.22248
    if p1f.P1 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p1f.jkdat.p', 'rb')
            p1f.P1 = pickle.load(fn1)
            print("load of jackknifed p1 energy successful")
        except FileNotFoundError:
            #print("jackknifed p1 energy not found")
            pass
    if p1f.P1 is not None:
        ret = p1f.P1
    return ret
p1f.P1 = np.nan

def p11f():
    """E_pi(|p|=11)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    pionstr = '_pioncorrChk_'
    if LATTICE == '24c':
        ret = 0.39461
    elif LATTICE == '32c':
        ret = 0.2971
    if p11f.P11 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p11f.jkdat.p', 'rb')
            p11f.P11 = pickle.load(fn1)
            print("load of jackknifed p11 energy successful")
        except FileNotFoundError:
            #print("jackknifed p11 energy not found")
            pass
    if p11f.P11 is not None:
        ret = p11f.P11
    return ret
p11f.P11 = np.nan

def p111f():
    """E_pi(|p|=11)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    pionstr = '_pioncorrChk_'
    if LATTICE == '24c':
        ret = 0.4715
    elif LATTICE == '32c':
        ret = 0.3514
    if p111f.P111 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p111f.jkdat.p', 'rb')
            p111f.P111 = pickle.load(fn1)
            print("load of jackknifed p111 energy successful")
        except FileNotFoundError:
            #print("jackknifed p111 energy not found")
            pass
    if p111f.P111 is not None:
        ret = p111f.P111
    return ret
p111f.P111 = np.nan

def fitepi(norm):
    """Select the right E_pi"""
    ret = None
    if norm == 0:
        ret = massfunc()
    elif norm == 1:
        ret = p1f()
        ret = select_subset(ret)
    elif norm == 2:
        ret = p11f()
        ret = select_subset(ret)
    elif norm == 3:
        ret = p111f()
        ret = select_subset(ret)
    else:
        assert ret is not None, ""
    return ret

def select_subset(arr, elimlist=None):
    """Apply binning and subsetting of data as needed (duplication of code in proc_folder)"""
    ret = arr
    elimlist = ELIM_JKCONF_LIST if elimlist is None else elimlist
    if hasattr(ret, '__iter__'):
        ret = BINOUT(ret)
        ret = HALFTOTAL(ret)
        if ELIM_JKCONF_LIST:
            ret = elim_jkconfigs(ret, elimlist)
    ret = np.asarray(ret)
    for i in ret.shape:
        assert i == max(ret.shape) or i == 1
    ret = ret.flatten()
    return ret


def correct_epipi(energies, config_num=None, irr=None, uncorrect=False):
    """Correct 24c dispersion relation errors using fits
    lattice disp -> continuum
    """
    irr = IRREP if irr is None else irr
    assert irr is not None, "irrep not set."
    assert hasattr(energies, '__iter__'),\
        "corrections not supported for single correlators"
    energies = np.asarray(energies)
    correction = np.zeros(energies.shape, np.float)
    for dim in range(len(energies)):
        moms = freemomenta(irr, dim)
        if moms is None:
            break
        try:
            assert moms is not None, "momenta not found; irrep has resonance"
        except AssertionError:
            print("momenta not found")
            print(moms, irr, dim)
            sys.exit(1)
        mom1, mom2 = moms
        norma = rf.norm2(mom1)
        normb = rf.norm2(mom2)
        epi1 = fitepi(norma)
        epi2 = fitepi(normb)
        crect = ((dispersive(mom1, continuum=True)+dispersive(
            mom2, continuum=True))-(epi1+epi2))
        crect = np.asarray(crect)
        origshape = crect.shape
        if hasattr(crect, '__iter__') and\
           crect.shape:
            #assert config_num is not None, "index bug "+\
            #    str(crect)+" "+str(energies[dim])
            if config_num is None:
                crect = em.acmean(crect, axis=0)
            elif len(crect) > config_num:
                crect = crect[config_num]
        try:
            correction[dim] = crect
        except ValueError:
            print("error in lattice spacing correction function.")
            print("correction:", crect)
            print("energies:", energies)
            print("crect.shape=", crect.shape)
            print("energies.shape=", energies.shape)
            print("config num", config_num)
            print("orig shape", origshape)
            raise
        if uncorrect:
            correction *= -1
        if PIONRATIO:
            correction = np.asarray(correction)
            correction *= 0.0
        #if BOX_LENGTH == 32: # assume 32c has good dispersion relation for now.
        #    correction *= 0
    return correction

def uncorrect_epipi(epipi, irr=None):
    """Uncorrect a single energy (continuum disp->lattice disp)"""
    if epipi is None:
        ret = None
    else:
        energies = [epipi]
        ret = correct_epipi(energies, irr, uncorrect=True)[0]
        ret += energies[0]
    return ret

def norm2(mom):
    """Norm^2"""
    return mom[0]**2+mom[1]**2+mom[2]**2

def dispersive(momentum, mass=None, box_length=None, continuum=False):
    """get the dispersive analysis energy == sqrt(m^2+p^2)"""
    mass = massfunc() if mass is None else mass
    mass = np.asarray(mass)
    assert continuum == CONTINUUM, "dispersion relation mismatch."
    box_length = BOX_LENGTH if box_length is None else box_length
    if hasattr(mass, '__iter__') and mass.shape and momentum is not None:
        ret = [sqrt(i**2 + 4*np.sin(pi/box_length)**2*norm2(momentum)) for i in mass]
        ret = [sqrt(i**2+(2*pi/box_length)**2*norm2(momentum)) for i in mass]\
            if continuum else ret
    else:
        if momentum is not None:
            ret = sqrt(mass**2 + 4*np.sin(pi/box_length)**2*norm2(momentum))
            ret = sqrt(mass**2+(2*pi/box_length)**2*norm2(momentum))\
                if continuum else ret
        else:
            ret = None
    if momentum is not None:
        if not em.acsum(momentum):
            try:
                assert np.allclose(ret, mass, rtol=1e-8), "precision gain"
            except AssertionError:
                ret = massfunc()
                assert np.allclose(ret, mass, rtol=1e-16), "precision gain"
    if ret is not None:
        ret = np.asarray(ret)
        ret = ret.flatten()
    return ret
