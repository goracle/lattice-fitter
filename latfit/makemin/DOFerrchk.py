import sys
from latfit.config import START_PARAMS
from latfit.config import AUTO_FIT

def DOFerrchk(dimcov):
    """Check to make sure degrees of freedom > 0."""
    if len(START_PARAMS) >= dimcov and not AUTO_FIT:
        print "***ERROR***"
        print "Degrees of freedom <= 0."
        print "Rerun with a different number of fit parameters."
        sys.exit(1)
    return 0
