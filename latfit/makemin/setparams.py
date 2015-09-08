from __future__ import division
from latfit.mathfun import chi_sq
#import global variables
from latfit.globs import START_PARAMS
from latfit.globs import MBOUND
from latfit.globs import AYONE
from latfit.globs import START_A_0
from latfit.globs import START_ENERGY

def setparams(switch, numpextra, start_params = START_PARAMS, mbound = MBOUND, ayone = AYONE):
    """Set start parameters and bounds on the initial parameters.
    Base initial values on global variables.
    """
    ####minimize 7ab
    #minimize chi squared
    if switch == '0':
        print "Pade fit."
        #m_rho = 770 MeV
        #m_pi = 140 MeV
        #b_i>2.3716
        #start_params = [-.18, 0.09405524, 1.21877187, 2.4]
        #mass of pion bound
        #mbound = 0.0779
        #mass of rho meson bound
        #mbound = 2.4025
        if numpextra == 0:
            ADDPARAMS = []
        else:
            ADDPARAMS = [ayone+i/1000.0 if i%2 == 0
                         else mbound*1.01+i/1000.0
                         for i in range(numpextra*2)]
        for i in ADDPARAMS:
            start_params.append(i)
        BINDS = [(None, None), (None, None), (None, None),
                 (mbound, None)]
        ADDBINDS = [(None, None) if i%2 == 0 else (mbound, None)
                    for i in range(numpextra*2)]
        for i in ADDBINDS:
            BINDS.append(i)
        BINDS = tuple(BINDS)
    if switch == '1':
        start_params = [START_A_0, START_ENERGY]
        BINDS = ((None, None), (0, None))
    return start_params, BINDS
