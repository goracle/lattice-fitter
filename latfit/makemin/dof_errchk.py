"""Check to make sure dof>0"""
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT
from latfit.config import FIT
from latfit.analysis.errorcodes import DOFNonPos
import latfit.config


def dof_errchk(dimcov, dimops=1):
    """Check to make sure degrees of freedom > 0."""
    dof = -1*len(START_PARAMS) + dimcov*dimops
    if dof <= 0 and not AUTO_FIT and FIT:
        print("***ERROR***")
        print("Degrees of freedom <= 0.")
        print("dimcov =", dimcov)
        print("dimops =", dimops)
        print("Rerun with a different number of fit parameters.")
        raise DOFNonPos(dof=0, excl=latfit.config.FIT_EXCL)
    return 0
