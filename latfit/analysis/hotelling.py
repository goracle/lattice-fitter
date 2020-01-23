"""Compute the hotelling t^2 distribution given dof"""
from scipy.optimize import fsolve
from scipy import stats
from latfit.config import BOOTSTRAP_BLOCK_SIZE, SUPERJACK_CUTOFF
from latfit.config import UNCORR, PVALUE_MIN
import latfit.config

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


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
    cor = getcor(dof, em1, dee2)
    ret = 2*dee2**2*(dof+dee2-2)/dof/((dee2-2)**2)/(dee2-4)
    ret *= cor**2
    return ret

def getcor(dof, em1, dee2):
    """Get correction to F distribution"""
    cor = dof*em1/dee2
    return cor


def avg(dof, nconf):
    """Compute the average of the hotelling t^2 dist."""
    em1 = getm(nconf)
    dee2 = em1-dof+1
    cor = getcor(dof, em1, dee2)
    ret = dee2/(dee2-2)
    ret *= cor
    return ret

def hstr(dof, nconf):
    """Get a usable string for printing purposes"""
    vare = var(dof, nconf)
    mean = avg(dof, nconf)
    ret = 'Hotelling (variance, mean):'+str(vare)+", "+str(mean)
    return ret

@PROFILE
def overfit_chisq_fiduc(num_configs, dof, guess=None):
    """Find the overfit 5 sigma cut
    (see chisqfiduc for the lower cut on the upper bound)
    """
    key = (num_configs, dof)
    t2correction = (num_configs-dof)/(num_configs-1)/dof
    cor = t2correction
    if key in overfit_chisq_fiduc.cache:
        ret = overfit_chisq_fiduc.cache[key]
    else:
        cut = stats.f.cdf(dof*cor, dof, num_configs-dof)
        lbound = 3e-7
        func = lambda tau: ((1-cut*lbound)-(
            stats.f.sf(abs(tau)*cor, dof, num_configs-dof)))**2
        sol = abs(float(fsolve(func, 1e-5 if guess is None else guess)))
        sol2 = dof
        assert abs(func(sol)) < 1e-12, "fsolve failed:"+str(num_configs)+\
            " "+str(dof)
        diff = (sol2-sol)/(num_configs-SUPERJACK_CUTOFF-1)
        assert diff > 0,\
            "bad solution to p-value solve, chi^2(/t^2)/dof solution > 1"
        ret = sol2-diff
        overfit_chisq_fiduc.cache[key] = ret
    return ret
overfit_chisq_fiduc.cache = {}

@PROFILE
def chisqfiduc(num_configs, dof):
    """Find the chi^2/dof (t^2/dof) cutoff (acceptance upper bound)
    defined as > 5 sigma away from an acceptable pvalue
    2*dof is the variance in chi^2 (t^2)
    """
    key = (num_configs, dof)
    t2correction = (num_configs-dof)/(num_configs-1)/dof
    cor = t2correction
    if key in chisqfiduc.mem:
        ret = chisqfiduc.mem[key]
    else:
        func = lambda tau: PVALUE_MIN*3e-7-(stats.f.sf(tau*cor, dof,
                                                       num_configs-dof))
        func2 = lambda tau: PVALUE_MIN-(stats.f.sf(tau*cor, dof,
                                                   num_configs-dof))
        # guess about 2 for the max chi^2/dof
        sol = float(fsolve(func, dof))
        sol2 = float(fsolve(func2, dof))
        assert abs(func(sol)) < 1e-8, "fsolve failed."
        assert abs(func2(sol2)) < 1e-8, "fsolve2 failed."
        # known variance of chi^2 is 2*dof,
        # but skewed at low dof (here chosen to be < 10)
        # thus, the notion of a "5 sigma fluctuation" is only defined
        # as dof->inf
        # so we have a factor of 2 to make low dof p-value cut less aggressive
        #ret = sol+5*(2 if dof < 10 else\
        # 1)*np.sqrt(2*dof)/(num_configs-SUPERJACK_CUTOFF)
        diff = (sol-sol2)/(num_configs-SUPERJACK_CUTOFF-1)
        ret = sol2+diff
        #print(ret/dof, sol/dof, num_configs, dof, PVALUE_MIN,
        #      1-stats.chi2.cdf(ret, dof), 1-stats.chi2.cdf(sol, dof))
        chisqfiduc.mem[key] = ret
    return ret
chisqfiduc.mem = {}

def torchi():
    """Are we calculating Hotelling's t^2 statistic or a true chi^2?
    return the corresponding string.
    """
    if UNCORR:
        ret = 'chisq/dof='
    else:
        ret = 't^2/dof='
    return ret
