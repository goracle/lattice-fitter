"""Misc. Functions"""
from math import sqrt, pi, sin

MASS = 0
BOX_LENGTH = 1

def norm2(mom):
    """Norm^2"""
    return mom[0]**2+mom[1]**2+mom[2]**2

def dispersive(momentum, mass=None, box_length=None):
    """get the dispersive analysis energy == sqrt(m^2+p^2)"""
    mass = MASS if mass is None else mass
    box_length = BOX_LENGTH if box_length is None else box_length
    return sqrt((mass)**2+ 4*sin(pi/box_length)**2*norm2(momentum)) # two pions so double the mass
