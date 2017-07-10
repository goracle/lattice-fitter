from collections import namedtuple
from latfit.config import JACKKNIFE

def get_fit_params(cov, reuse):
    params = namedtuple('fit_params', [
        'dimops', 'num_configs', 'prefactor'])   
    params.num_configs = len(reuse[xmin])
    try:
        params.dimops = len(cov[0][0])
    except (NameError, TypeError):
        params.dimops = 1
    if JACKKNIFE == 'YES':
        #applying jackknife correction of (count-1)^2
        warn("Applying jackknife correction to cov. matrix.")
        params.prefactor = (num_configs-1.0)/(1.0*num_configs)
    elif JACKKNIFE == 'NO':
        params.prefactor = (1.0)/((num_configs-1.0)*(1.0*num_configs))
    return params
