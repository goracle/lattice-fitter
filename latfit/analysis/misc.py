"""Misc. Functions"""
import sys
from math import sqrt, pi, sin
import numpy as np
import pickle
from latfit.utilities.op_compose import freemomenta
import latfit.utilities.read_file as rf

BOX_LENGTH = 1
MASS = 0
LATTICE = None
PIONRATIO = False


P1 = None
P11 = None
P111 = None
#p1
#P1 = 0.29641(25)
#p11
#P11 = 0.39461(38)
#p111
#P111 = 0.4715(11)
#p0
#P0 = 0.13957(19)


def mass():
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    if misc.MASS == 0:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'mom000.jkdat.p', 'rb')
            misc.MASS = pickle.load(fn1)
        except FileNotFoundError:
            pass
    return misc.MASS

def p1():
    """E_pi(|p|=1)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    if LATTICE == '24c':
        ret = 0.29641
    elif LATTICE == '32c':
        ret = 0.22248
    if p1.P1 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p1.jkdat.p', 'rb')
            P1 = pickle.load(fn1)
        except FileNotFoundError:
            pass
    if p1.P1 is not None:
        ret = p1.P1
    return ret
p1.P1 = None

def p11():
    """E_pi(|p|=11)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    if LATTICE == '24c':
        ret = 0.39461
    elif LATTICE == '32c':
        ret = 0.2971
    if p11.P11 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p11.jkdat.p', 'rb')
            p11.P11 = pickle.load(fn1)
        except FileNotFoundError:
            pass
    if p11.P11 is not None:
        ret = p11.P11
    return ret
p11.P11 = None

def p111():
    """E_pi(|p|=11)"""
    pionstr = '_pioncorrChk_' if not PIONRATIO else '_pioncorr_'
    if LATTICE == '24c':
        ret = 0.4715
    elif LATTICE == '32c':
        ret = 0.3514
    if p111.P111 is None:
        try:
            fn1 = open('x_min_'+LATTICE+pionstr+'p111.jkdat.p', 'rb')
            p111.P111 = pickle.load(fn1)
        except FileNotFoundError:
            pass
    if p111.P111 is not None:
        ret = p111.P111
    return ret
p111.P111 = None

IRREP = None
CONTINUUM = True
def bintotal(ret):
    assert None
    return ret
def halftotal(ret):
    assert None
    return ret

def fitepi(norm):
    """Select the right E_pi"""
    ret = None
    if norm == 0:
        ret = MASS
    elif norm == 1:
        ret = p1()
    elif norm == 2:
        ret = p11()
    elif norm == 3:
        ret = p111()
    else:
        assert ret is not None, ""
    if hasattr(ret, '__iter__'):
        ret = binout(ret)
        ret = halftotal(ret)
    return ret

def correct_epipi(energies, irr=None, uncorrect=False):
    """Correct 24c dispersion relation errors using fits
    lattice disp -> continuum
    """
    irr = IRREP if irr is None else irr
    assert irr is not None, "irrep not set."
    assert hasattr(energies, '__iter__'),\
        "corrections not supported for single correlators"
    correction = np.zeros(np.asarray(energies).shape, np.float)
    for dim in range(len(energies)):
        moms = freemomenta(irr, dim)
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
        correction[dim] = (dispersive(mom1, continuum=True)+dispersive(
            mom2, continuum=True))-(epi1+epi2)
        if uncorrect:
            correction *= -1
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
    mass = MASS if mass is None else mass
    assert continuum == CONTINUUM, "dispersion relation mismatch."
    box_length = BOX_LENGTH if box_length is None else box_length
    ret = sqrt((mass)**2+ 4*sin(
        pi/box_length)**2*norm2(momentum)) # two pions so double the mass
    ret = sqrt(mass**2+(2*pi/box_length)**2*norm2(
        momentum)) if continuum else ret
    return ret
