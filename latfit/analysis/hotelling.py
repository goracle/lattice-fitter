"""Compute the hotelling t^2 distribution given dof"""
from latfit.config import BOOTSTRAP_BLOCK_SIZE
import latfit.config

def getm(nconf):
    """"Get the m parameter"""
    block = 1
    if latfit.config.BOOTSTRAP:
        block = BOOTSTRAP_BLOCK_SIZE
    em1 = nconf/block
    return em1


def var(dof, nconf):
    """Compute the variance of the hotelling t^2 dist."""
    em1 = getm(nconf)
    dee2 = em1-dof+1
    cor = dof*em1/dee2
    ret = 2*dee2**2*(dof+dee2-2)/dof/((dee2-2)**2)/(dee2-4)
    ret *= cor**2
    return ret
    

def avg(dof, nconf):
    """Compute the average of the hotelling t^2 dist."""
    em1 = getm(nconf)
    dee2 = em1-dof+1
    cor = dof*em1/dee2
    ret = dee2/(dee2-2)
    ret *= cor
    return ret

def hstr(dof, nconf):
    """Get a usable string for printing purposes"""
    vare = var(dof, nconf)
    mean = avg(dof, nconf)
    ret = 'Hotelling (variance, mean):'+str(vare)+", "+str(mean)
    return ret

