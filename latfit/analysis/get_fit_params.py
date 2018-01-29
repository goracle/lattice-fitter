"""Get common fit parameters"""
from warnings import warn
from collections import namedtuple
import numpy as np

from latfit.config import JACKKNIFE


def get_fit_params(cov, reuse, xmin, fitrange, xstep):
    """Get a namedtuple of fit params
    prefactor is jackknife prefactor
    dimops is the dimension of the GEVP matrix
    num_configs is the number of configurations to average over.
    """
    params = namedtuple('fit_params', ['dimops', 'num_configs',
                                       'prefactor', 'time_range'])
    params.num_configs = len(reuse[xmin])
    try:
        params.dimops = len(cov[0][0])
    except (NameError, TypeError):
        params.dimops = 1
    if JACKKNIFE == 'YES':
        # applying jackknife correction of (count-1)^2
        warn("Applying jackknife correction to cov. matrix.")
        params.prefactor = (params.num_configs-1.0)/(1.0*params.num_configs)
    elif JACKKNIFE == 'NO':
        params.prefactor = (1.0)/(
            (params.num_configs-1.0)*(1.0*params.num_configs))
    params.time_range = np.arange(fitrange[0], fitrange[1]+1, xstep)
    return params
