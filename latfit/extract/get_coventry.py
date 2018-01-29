"""Get a single entry in the covariance matrix."""
import numpy as np

from latfit.config import UNCORR, GEVP

if UNCORR:
    def get_coventry_gevp(reuse, sameblk, avgi):
        """Get entry in cov. mat., GEVP, uncorr"""
        dimops = len(avgi)
        coventry = np.zeros((dimops, dimops))
        if sameblk:
            for opa in range(dimops):
                coventry[opa][opa] = np.sum(
                    [(avgi[opa]-reuse['i'][k][opa])*(
                        avgi[opa]-reuse['i'][k][opa])
                     for k in range(dimops)], axis=0)
        else:
            pass  # keep it zero, off diagonals are zero
        return coventry

else:
    def get_coventry_gevp(reuse, sameblk, avgi):
        """Get entry in cov. mat., GEVP"""
        dimops = len(avgi)
        coventry = np.zeros((dimops, dimops))
        num_configs = len(reuse['i'])
        if sameblk:
            coventry = np.sum([np.outer(
                (avgi-reuse['i'][k]),
                (avgi-reuse['i'][k])) for k in range(num_configs)], axis=0)
        else:
            coventry = np.sum([np.outer(
                (avgi-reuse['i'][k]),
                (np.mean(reuse['j'], axis=0)-reuse['j'][k]))
                               for k in range(num_configs)], axis=0)
        return coventry

if UNCORR:
    def get_coventry_simple(reuse, sameblk, avgi):
        """Get entry in cov. mat., uncorr"""
        if sameblk:
            coventry = np.dot(reuse['i']-avgi, reuse['i']-avgi)
        else:
            coventry = 0
        return coventry

else:
    def get_coventry_simple(reuse, sameblk, avgi):
        """Get entry in cov. mat."""
        if sameblk:
            coventry = np.dot(reuse['i']-avgi, reuse['i']-avgi)
        else:
            coventry = np.dot(reuse['i']-avgi,
                              reuse['i']-np.mean(reuse['j']))
        return coventry

if GEVP:
    def get_coventry(reuse, sameblk, avgi):
        """get the entry in the cov. mat. (GEVP)"""
        return get_coventry_gevp(reuse, sameblk, avgi)
else:
    def get_coventry(reuse, sameblk, avgi):
        """get the entry in the cov. mat."""
        return get_coventry_simple(reuse, sameblk, avgi)
