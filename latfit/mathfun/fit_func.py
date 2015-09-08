from math import fsum, exp
from numpy import arange

def fit_func(ctime, trial_params, switch):
    """Give result of function computed to fit the data given in <inputfile>
    (See procargs(argv))
    """
    if switch == '0':
        #pade function
        return trial_params[0]+ctime*(trial_params[1]+trial_params[2]/(
            trial_params[3]+ctime)+fsum([trial_params[ci]/(
                trial_params[ci+1]+ctime) for ci in arange(
                    4, len(trial_params), 2)]))
    #return (trial_params[0]+trial_params[1]*ctime+
    #trial_params[3]*ctime*ctime)/(
    #           1+trial_params[2]*ctime)
    if switch == '1':
        #simple exponential
        return trial_params[0]*exp(-ctime*trial_params[1])
