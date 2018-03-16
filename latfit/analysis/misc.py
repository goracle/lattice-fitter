"""Misc. Functions"""
from math import sqrt, pi

MASS = 0
BOX_LENGTH = 1

def norm2(mom):
    """Norm^2"""
    return mom[0]**2+mom[1]**2+mom[2]**2

def dispersive(momentum, mass=None, box_length=None):
    """get the dispersive analysis energy == sqrt(m^2+p^2)"""
    if mass is None:
        mass = MASS
    if box_length is None:
        box_length = BOX_LENGTH
    return sqrt((mass)**2+(2*pi/box_length)**2*norm2(momentum))
