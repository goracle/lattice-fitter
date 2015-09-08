from collections import namedtuple

from latfit.checks.pos_def_check import pos_def_check
from latfit.checks.sym_check import sym_check
from latfit.checks.too_small_check import too_small_check
from latfit.extract.get_ccovandcoords import get_ccovandcoords

def simple_proc_file(kfile, cxmin, cxmax, eigcut = 10**(-10)):
    """Process file with precomputed covariance matrix."""
    rets = namedtuple('rets', ['coord', 'covar', 'numblocks'])
    ccov, proccoords = get_ccovandcoords(kfile, cxmin, cxmax)
    ##Checks
    #check symmetry
    sym_check(ccov)
    #check pos-definitiveness
    pos_def_check(ccov)
    #are the eigenvalues too small? check.
    too_small_check(ccov, eigcut)
    #checks done.  return results
    return rets(coord=proccoords, covar=ccov, numblocks=len(ccov))
