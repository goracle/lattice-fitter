"""Get a single entry in the covariance matrix."""
import numpy as np
from accupy import kdot

from latfit.config import UNCORR, GEVP
from latfit.config import JACKKNIFE_BLOCK_SIZE
from latfit.utilities import exactmean as em
from latfit.mathfun.block_ensemble import block_ensemble

if UNCORR:
    def get_coventry_gevp(reuse_blocked, sameblk, avgi):
        """Get entry in cov. mat., GEVP, uncorr"""
        dimops = len(avgi)
        coventry = np.zeros((dimops, dimops))
        num_configs = len(reuse_blocked['i'])
        if sameblk:
            for opa in range(dimops):
                coventry[opa][opa] = em.acsum(
                    [(avgi[opa]-reuse_blocked['i'][k][opa])*(
                        avgi[opa]-reuse_blocked['i'][k][opa])
                     for k in range(num_configs)], axis=0)
        else:
            pass  # keep it zero, off diagonals are zero
        return coventry

else:
    def get_coventry_gevp(reuse_blocked, sameblk, avgi):
        """Get entry in cov. mat., GEVP"""
        dimops = len(avgi)
        coventry = np.zeros((dimops, dimops))
        num_configs = len(reuse_blocked['i'])
        if sameblk:
            coventry = em.acsum([np.outer(
                (avgi-reuse_blocked['i'][k]),
                (avgi-reuse_blocked['i'][k])) for k in range(
                    num_configs)], axis=0)
        else:
            coventry = em.acsum([np.outer(
                (avgi-reuse_blocked['i'][k]),
                (em.acmean(reuse_blocked['j'], axis=0)-reuse_blocked['j'][
                    k])) for k in range(num_configs)], axis=0)
        return coventry

if UNCORR:
    def get_coventry_simple(reuse_blocked, sameblk, avgi):
        """Get entry in cov. mat., uncorr"""
        if sameblk:
            coventry = kdot(reuse_blocked['i']-avgi, reuse_blocked[
                'i']-avgi)
        else:
            coventry = 0
        return coventry

else:
    def get_coventry_simple(reuse_blocked, sameblk, avgi):
        """Get entry in cov. mat."""
        if sameblk:
            coventry = kdot(reuse_blocked['i']-avgi, reuse_blocked['i']-avgi)
        else:
            coventry = kdot(reuse_blocked['i']-avgi,
                            reuse_blocked['i']-em.acmean(
                                reuse_blocked['j']))
        return coventry

if GEVP:
    def get_coventry(reuse, sameblk, avgi):
        """get the entry in the cov. mat. (GEVP)"""
        return get_coventry_gevp(get_blocked(reuse), sameblk, avgi)
else:
    def get_coventry(reuse, sameblk, avgi):
        """get the entry in the cov. mat."""
        return get_coventry_simple(get_blocked(reuse), sameblk, avgi)

def get_blocked(reuse):
    """Get the blocked ensemble from the dict"""
    assert len(reuse['i']) == len(reuse['j'])
    nconfigs = len(reuse['i'])/JACKKNIFE_BLOCK_SIZE
    assert nconfigs == int(nconfigs), str(nconfigs)
    nconfigs = int(nconfigs)
    reuse_blocked = {}
    reuse_blocked['i'] = block_ensemble(nconfigs, reuse['i'])
    reuse_blocked['j'] = block_ensemble(nconfigs, reuse['j'])
    return reuse_blocked
