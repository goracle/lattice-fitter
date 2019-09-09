"""All error classes for custom error handling"""
from latfit.config import UNCORR
import latfit.config

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

class TooManyBadFitsError(Exception):
    """Error if too many jackknifed fits have a large chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, pvalue=None, message=''):
        print("***ERROR***")
        if UNCORR:
            print("Too many fits have bad chi^2")
            print("chi^2 average up to this point:", chisq)
        else:
            print("Too many fits have bad t^2")
            print("t^2 average up to this point:", chisq)
        print("pvalue up to this point:", pvalue)
        super(TooManyBadFitsError, self).__init__(message)
        self.message = message

class EnergySortError(Exception):
    """If the energies are not sorted in ascending order
    (if the systematic errors are large)
    """
    @PROFILE
    def __init__(self, message=''):
        print("***ERROR***")
        print("Energies are not sorted in ascending order")
        super(EnergySortError, self).__init__(message)
        self.message = message

class BadJackknifeDist(Exception):
    """Exception for bad jackknife distribution"""
    @PROFILE
    def __init__(self, message=''):
        print("***ERROR***")
        if UNCORR:
            print("Bad jackknife distribution, variance in chi^2 too large")
        else:
            print("Bad jackknife distribution, variance in t^2 too large")
        super(BadJackknifeDist, self).__init__(message)
        self.message = message

class NoConvergence(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, message=''):
        print("***ERROR***")
        print("Minimizer failed to converge")
        super(NoConvergence, self).__init__(message)
        self.message = message

class DOFNonPos(Exception):
    """Exception for dof < 0"""
    @PROFILE
    def __init__(self, dof=None, message=''):
        print("***ERROR***")
        print("dof < 1: dof=", dof)
        print("FIT_EXCL=", latfit.config.FIT_EXCL)
        super(DOFNonPos, self).__init__(message)
        self.dof = dof
        self.message = message

class BadChisq(Exception):
    """Exception for bad chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, message='', dof=None):
        print("***ERROR***")
        if UNCORR:
            print("chisq/dof >> 1 or p-value >> 0.5 chi^2/dof =",
                  chisq, "dof =", dof)
        else:
            print("t^2/dof >> 1 or p-value >> 0.5 t^2/dof =",
                  chisq, "dof =", dof)
        super(BadChisq, self).__init__(message)
        self.chisq = chisq
        self.dof = dof
        self.message = message

class ImaginaryEigenvalue(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, expression='', message=''):
        print("***ERROR***")
        print('imaginary eigenvalue found')
        super(ImaginaryEigenvalue, self).__init__(message)
        self.expression = expression
        self.message = message

class XmaxError(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, problemx=None, message=''):
        print("***ERROR***")
        print('xmax likely too large, decreasing')
        super(XmaxError, self).__init__(message)
        self.problemx = problemx
        self.message = message

class PrecisionLossError(Exception):
    """Error if precision loss in eps prescription"""
    def __init__(self, message=''):
        print("***ERROR***")
        print("Precision loss.")
        super(PrecisionLossError, self).__init__(message)
        self.message = message

class ZetaError(Exception):
    """Define an error for generic phase shift calc failure"""
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

class RelGammaError(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, gamma=None, epipi=None, message=''):
        print("***ERROR***")
        print("gamma < 1: gamma=", gamma, "Epipi=", epipi)
        super(RelGammaError, self).__init__(message)
        self.gamma = gamma
        self.epipi = epipi
        self.message = message
