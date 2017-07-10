"""Get a single entry in the covariance matrix."""
import numpy as np

from latfit.config import UNCORR, GEVP

if UNCORR:
    def get_coventry_gevp(reuse, sameblk, avgI):
        dimops = len(avgI)
        coventry = np.zeros((dimops,dimops))
        if sameblk:
            num_configs = len(reuse['i'])
            for opa in range(dimops):
                coventry[opa][opa]=np.sum([(avgI[opa]-reuse['i'][k][opa])*(avgI[opa]-reuse['i'][k][opa]) for k in range(dimops)],axis=0)
        else:
            pass #keep it zero, off diagonals are zero
        return coventry

else:
    def get_coventry_gevp(reuse, sameblk, avgI):
        dimops = len(avgI)
        coventry = np.zeros((dimops,dimops))
        num_configs=len(reuse['i'])
        if sameblk:
            coventry=np.sum([np.outer((avgI-reuse['i'][k]),(avgI-reuse['i'][k])) for k in range(num_configs)],axis=0)
        else:
            coventry=np.sum([np.outer((avgI-reuse['i'][k]),(np.mean(reuse['j'],axis=0)-reuse['j'][k])) for k in range(num_configs)],axis=0)
        return coventry

if UNCORR:
    def get_coventry_simple(reuse, sameblk, avgI):
        if sameblk:
            coventry = np.dot(reuse['i']-avgI, reuse['i']-avgI)
        else:
            coventry = 0
        return coventry

else:
    def get_coventry_simple(reuse, sameblk, avgI):
        if sameblk:
            coventry = np.dot(reuse['i']-avgI, reuse['i']-avgI)
        else:
            coventry = np.dot(reuse['i']-avgI, reuse['i']-np.mean(reuse['j']))
        return coventry

if GEVP:
    def get_coventry(reuse, sameblk, avgI):
        """get the entry in the cov. mat."""
        return get_coventry_gevp(file_tup, reuse, ij_str)
else:
    def get_coventry(reuse, sameblk, avgI):
        """get the entry in the cov. mat."""
        return get_coventry_simple(file_tup, reuse, ij_str)
