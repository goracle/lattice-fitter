"""Perform cache invalidation"""
import numpy as np

from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT

import latfit.extract.extract as ext
import latfit.extract.getblock.gevp_pionratio as grat
import latfit.extract.getblock.gevp_linalg as glin
import latfit.singlefit as sfit
import latfit.config

EXCL_ORIG = np.copy(EXCL_ORIG_IMPORT)

def reset_main(mintol):
    """Reset all dynamic variables"""
    latfit.config.MINTOL = mintol
    latfit.config.FIT_EXCL = np.copy(EXCL_ORIG)
    latfit.config.FIT_EXCL = tuple(tuple(i) for i in latfit.config.FIT_EXCL)
    partial_reset()

def partial_reset():
    """Partial reset during tloop. Only partially clear some caches.
    """
    glin.reset_sortevals() # this deserves some scrutiny #todo
    sfit.singlefit_reset()

def reset_cache():
    """Until the globals in the loop are removed,
    caches must be manually cleared"""
    ext.reset_extract()
    grat.reset()
    partial_reset()