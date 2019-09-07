#!/usr/bin/python3
"""Get the phase shift for I1 moving frames"""
import sys
import numpy as np
from czeta import czeta
import latfit.utilities.read_file as rf
from latfit.utilities import exactmean as em

COMP = [None, None, None]
L_BOX = 0.0
IRREP = None
MPION = 0.0

def zetalm(l_arg, m_arg, qtwo, gamma):
    """Compute generalized Luscher zeta function"""
    zeta = czeta()
    dx1, dy1, dz1 = COMP
    zeta.set_dgam(dx1, dy1, dz1, gamma)
    zeta.set_lm(l_arg, m_arg)
    ret = zeta.evaluate(qtwo)
    return ret

def main():
    """Get the phase shift for I1 moving frames
    arglist = [binpath, str(epipi), str(PION_MASS), str(lbox),
            str(comp[0]), str(comp[1]), str(comp[2]), str(gamma),
            str(int(not FIT_SPACING_CORRECTION))]
    """
    this = sys.modules[__name__]
    assert len(sys.argv) == 6, "needs: epipi, mpion, lbox, com units, irrep"
    epipi = sys.argv[1]
    setattr(this, 'MPION', np.float(sys.argv[2]))
    setattr(this, 'L_BOX', np.float(sys.argv[3]))
    setattr(this, 'COMP', rf.procmom(sys.argv[4]))
    setattr(this, 'IRREP', str(sys.argv[5]))
    return phase(epipi)


class Wfun:
    """Compute wlm"""
    def __init__(self):
        """init"""
        self.lbox = L_BOX
        self._kmom = None
        self._gamma = 1.0
        self.qarg = None
        self.cache = {}

    @property
    def kmom(self):
        """Remember the energy"""
        return self._kmom

    @kmom.setter
    def kmom(self, kmom):
        """set the k"""
        self._kmom = kmom
        self.qarg = (self.kmom*self.lbox/np.pi/2)

    def wfunfun(self, l_arg, m_arg):
        """computes w_lm, l_arg==l, m_arg==m"""
        key = (l_arg, m_arg)
        if key in self.cache:
            ret = self.cache[key]
        else:
            zlm = zetalm(l_arg, m_arg, self.qarg^2, self.gamma)
            denom = np.pi**(3/2)*np.sqrt(2*l_arg+1)
            denom *= self.gamma*self.qarg**(l_arg+1)
            denom = np.complex(denom)
            ret = zlm/denom
            if not ret:
                print("alert! w_"+str(l_arg)+", "+str(m_arg)+" = 0")
            self.cache[key] = ret
        return ret

    @property
    def gamma(self):
        """rel gamma"""
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        """gamma setter"""
        self._gamma = gamma

def computek(epipi):
    """Get the k (relative pipi momentum for zeta)"""
    return np.sqrt(epipi*epipi/4 - MPION**2)

def computegamma(epipi, comp):
    """Compute relativistic gamma given energy"""
    dotprod = np.dot(comp, comp)
    arg = epipi**2-(2*np.pi/L_BOX)**2*dotprod
    ret = epipi/np.sqrt(arg)
    return ret

def phase(epipi):
    """Get the phase shift in degrees for I=1 moving frame"""
    assert COMP, "center of mass momentum not set"
    assert L_BOX, "box length not set"
    assert MPION, "mass of pion not set"
    # sort this, per quantization conditions given in https://arxiv.org/pdf/1704.05439.pdf
    comp = np.asarray(sorted(list(COMP)))
    comp = np.abs(comp)

    # set up wlm
    wlm = Wfun()
    wlm.kmom = computek(epipi)
    wlm.gamma = computegamma(epipi, comp)

    cot = getcot(wlm)
    tan = 1/cot
    ret = np.arctan(tan)*180/np.pi
    return ret

def getcot(wlm):
    """Get cotangent(phase shift)"""
    assert IRREP is not None, "irrep not set"

    # initial starting value to cotangent
    cot = wlm(0, 0)
    units = em.acsum(np.abs(COMP))

    foundirr = True
    if units == 1:
        foundirr = False
        if IRREP == 'A_1PLUS_mom1':
            cot += 2*wlm(2, 0)
        elif IRREP == 'B_mom1':
            cot += -wlm(2, 0)
    elif units == 2:
        foundirr = False
        if IRREP == 'A_1PLUS_mom11':
            cot += wlm(2, 0)/2
            cot += 1j*np.sqrt(6)*wlm(2, 1)-np.sqrt(3/2)*wlm(2, 2)
        elif IRREP == 'A_2PLUS_mom11':
            cot += wlm(2, 0)/2
            cot += -1j*np.sqrt(6)*wlm(2, 1)-np.sqrt(3/2)*wlm(2, 2)
        elif IRREP == 'A_2MINUS_mom11':
            cot += wlm(2, 0)+np.sqrt(6)*wlm(2, 2)
    elif units == 3:
        foundirr = False
        if IRREP == 'A_1PLUS_avg_mom111':
            cot += -1j*wlm(2, 2)*np.sqrt(8/3)
            cot += np.real(wlm(2, 1))*np.sqrt(8/3)
            cot += np.imag(wlm(2, 1))*np.sqrt(8/3)
        elif IRREP == 'B_mom111':
            cot += 1j*np.sqrt(6)*wlm(2, 2)
    assert foundirr, "bad irrep specified:"+str(IRREP)
    return cot

if __name__ == '__main__':
    main()
