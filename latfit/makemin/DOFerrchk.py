import sys

def DOFerrchk(start_params, dimcov):
    """Check to make sure degrees of freedom > 0."""
    if len(start_params) >= dimcov:
        print "***ERROR***"
        print "Degrees of freedom <= 0."
        print "Rerun with a different number of fit parameters."
        sys.exit(1)
    return 0
