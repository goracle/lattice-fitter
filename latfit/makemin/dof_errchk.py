"""Check to make sure dof>0"""
import sys
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT
from latfit.config import FIT


def dof_errchk(dimcov, dimops=1):
    """Check to make sure degrees of freedom > 0."""
    if len(START_PARAMS) >= dimcov*dimops and not AUTO_FIT and FIT:
        print("***ERROR***")
        print("Degrees of freedom <= 0.")
        print("dimcov =", dimcov)
        print("dimops =", dimops)
        print("Rerun with a different number of fit parameters.")
        raise
        sys.exit(1)
    return 0
