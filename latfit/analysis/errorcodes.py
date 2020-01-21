"""All error classes for custom error handling"""
import mpi4py
from mpi4py import MPI
MPIRANK = MPI.COMM_WORLD.rank
#MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

PRIN = False

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

class BoolThrowErr:
    """object which throws an error if it's 'boolness' is examined"""
    def __bool__(self):
        assert None, "This bool has not been initialized."

class AvgCovSingular(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, message=''):
        print("***ERROR***")
        print("Average covariance matrix is singular")
        super(AvgCovSingular, self).__init__(message)
        self.message = message

class FitFail(Exception):
    """Exception for bad jackknife distribution"""
    def __init__(self, message='', prin=PRIN):
        if prin:
            print("***ERROR***")
            print("No fits to given fit window succeeded")
        super(FitFail, self).__init__(message)
        self.message = message


class MpiSkip(Exception):
    """Skip something due to parallelism"""
    def __init__(self, message='', prin=PRIN):
        if prin:
            print("Skipping fit, rank:", MPIRANK)
        super(MpiSkip, self).__init__(message)
        self.message = message

class FitRangeInconsistency(Exception):
    """Error if too many jackknifed fits have a large chi^2 (t^2)"""
    @PROFILE
    def __init__(self, message=''):
        print("***ERROR***")
        print("fit ranges give inconsistent results")
        super(FitRangeInconsistency, self).__init__(message)
        self.message = message



class TooManyBadFitsError(Exception):
    """Error if too many jackknifed fits have a large chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, pvalue=None, uncorr=BoolThrowErr(),
                 message=''):
        print("***ERROR***")
        if uncorr:
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
    def __init__(self, message='', uncorr=BoolThrowErr()):
        print("***ERROR***")
        if uncorr:
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
    def __init__(self, dof=None, message='', excl=None, prin=PRIN):
        if prin:
            print("***ERROR***")
        if dof is not None:
            if prin:
                print("dof < 1: dof=", dof)
        else:
            if prin:
                print("dof < 1")
        if excl is not None:
            if prin:
                print("FIT_EXCL=", excl)
        super(DOFNonPos, self).__init__(message)
        self.dof = dof
        self.message = message

class DOFNonPosFit(Exception):
    """Exception for dof < 0 (within fit; after getting coords)"""
    @PROFILE
    def __init__(self, dof=None, message='', excl=None, prin=PRIN):
        if prin:
            print("***ERROR***")
            print("dof < 1: dof=", dof)
            print("FIT_EXCL=", excl)
        super(DOFNonPosFit, self).__init__(message)
        self.dof = dof
        self.message = message

class BadChisq(Exception):
    """Exception for bad chi^2 (t^2)"""
    @PROFILE
    def __init__(self, chisq=None, message='', uncorr=BoolThrowErr(),
                 dof=None, prin=PRIN):
        if prin:
            print("***ERROR***")
        if uncorr:
            if prin:
                print("chisq/dof >> 1 or p-value >> 0.5 chi^2/dof =",
                      chisq, "dof =", dof)
        else:
            if prin:
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

class NegativeEnergy(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, expression='', message=''):
        print("***ERROR***")
        print('negative energy found')
        super(NegativeEnergy, self).__init__(message)
        self.expression = expression
        self.message = message



class NegativeEigenvalue(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, expression='', message=''):
        print("***ERROR***")
        print('negative eigenvalue found')
        super(NegativeEigenvalue, self).__init__(message)
        self.expression = expression
        self.message = message

class EigenvalueSignInconsistency(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, expression='', message='', prin=PRIN):
        if prin:
            print("***ERROR***")
            print('negative eigenvalue found (sign inconsistency)')
        super(EigenvalueSignInconsistency, self).__init__(message)
        self.expression = expression
        self.message = message

class XminError(Exception):
    """Exception for early time inconsistencies"""
    def __init__(self, problemx=None, message='', prin=PRIN):
        if prin:
            print("***ERROR***")
            print('xmin likely too small, increasing')
        super(XminError, self).__init__(message)
        self.problemx = problemx
        self.message = message


class XmaxError(Exception):
    """Exception for imaginary GEVP eigenvalue"""
    def __init__(self, problemx=None, message='', prin=PRIN):
        if prin:
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

class FinishedSkip(Exception):
    """Exception if the results are already finished"""
    def __init__(self, message='', prin=PRIN):
        if prin:
            print("Results finished")
        super(FinishedSkip, self).__init__(message)
