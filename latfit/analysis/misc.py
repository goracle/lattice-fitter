"""Misc. Functions"""
import sys
from math import sqrt, pi, sin
import numpy as np
from latfit.utilities.op_compose import freemomenta
import latfit.utilities.read_file as rf

MASS = 0
BOX_LENGTH = 1

#p1
#P1 = 0.29641(25)
P1 = 0.29641
#p11
#P11 = 0.39461(38)
P11 = 0.39461
#p111
#P111 = 0.4715(11)
P111 = 0.4715
#p0
#P0 = 0.13957(19)
P0 = 0.13975

IRREP = None
CONTINUUM = True

def fitepi(norm):
    ret = None
    if norm == 0:
        ret = P0
    elif norm == 1:
        ret = P1
    elif norm == 2:
        ret = P11
    elif norm == 3:
        ret = P111
    else:
        assert ret is not None, ""
    return ret

def correct_epipi(energies, irr=None, uncorrect=False):
    """Correct 24c dispersion relation errors using fits
    lattice disp -> continuum
    """
    irr = IRREP if irr is None else irr
    assert irr is not None, "irrep not set."
    assert hasattr(energies, '__iter__'),\
        "corrections not supported for single correlators"
    correction = np.zeros(np.asarray(energies).shape,np.float)
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
    return correction

def uncorrect_epipi(epipi, irr=None):
    """Uncorrect a single energy (continuum disp->lattice disp)"""
    if epipi is None:
        ret = None
    else:
        energies = [epipi]
        ret = correct_epipi(energies, irr, uncorrect=True)[0]
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
