"""Perform cache invalidation"""
import numpy as np

from latfit.config import FIT_EXCL as EXCL_ORIG_IMPORT

import latfit.extract.extract as ext
import latfit.extract.getblock.gevp_pionratio as grat
import latfit.extract.getblock.gevp_linalg as glin
import latfit.extract.getblock.getblock as gblock
import latfit.extract.getfiles as getf
import latfit.mathfun.proc_meff as pmeff
import latfit.singlefit as sfit
import latfit.config

EXCL_ORIG = np.copy(list(EXCL_ORIG_IMPORT))

def reset_main(mintol, check=False):
    """Reset all dynamic variables"""
    latfit.config.MINTOL = mintol
    latfit.config.FIT_EXCL = np.copy(list(EXCL_ORIG))
    latfit.config.FIT_EXCL = tuple(tuple(i) for i in list(latfit.config.FIT_EXCL))
    partial_reset(check=check)

def partial_reset(check=False):
    """Partial reset during tloop. Only partially clear some caches.
    """
    sfit.singlefit_reset()
    if check:
        assert ext.iscomplete()

def reset_cache():
    """Until the globals in the loop are removed,
    caches must be manually cleared"""
    partial_reset()
    reset_processing()

def reset_processing():
    """Reset all caches used in processing data
    into sets of fit points"""
    glin.reset_sortevals()
    ext.reset_extract()
    getf.file_reset()
    grat.reset()
    pmeff.EFF_MASS_TOMIN = pmeff.create_funcs()
    gblock.grd_inc_reset()
