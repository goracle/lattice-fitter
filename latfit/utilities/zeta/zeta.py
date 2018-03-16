"""Calculate phase shift from Luscher zeta for given inputs."""
import sys
import subprocess
import inspect
import os
import math
import numpy as np
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT, START_PARAMS, PTOTSQ, AINVERSE

class ZetaError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)

if CALC_PHASE_SHIFT:
    def zeta(epipi):
        """Calculate the I=0 scattering phase shift given the pipi energy
        for that channel.
        """
        try:
            epipi = epipi[1]
        except IndexError:
            try:
                epipi = epipi[0]
            except IndexError:
                pass
        epipi = epipi*AINVERSE
        #epipi = math.sqrt(epipi**2-(2*np.pi/L_BOX)**2*PTOTSQ) //not correct
        binpath = os.path.dirname(inspect.getfile(zeta))+'/main.o'
        arglist = [binpath, str(epipi), str(PION_MASS), str(L_BOX)]
        try:
            out = subprocess.check_output(arglist)
        except FileNotFoundError:
            print("Error in zeta: main.C not compiled yet.")
            print(subprocess.check_output(['pwd']))
            print(inspect.getfile(zeta))
            sys.exit(1)
        except subprocess.CalledProcessError:
            print("Error in zeta: calc of phase shift error:")
            print(epipi)
            errstr = subprocess.Popen(arglist,
                                   stdout=subprocess.PIPE).stdout.read()
            raise ZetaError(errstr)
        if(epipi*epipi/4-PION_MASS**2<0):
            out = float(out)*1j
        else:
            out = complex(float(out))
        return out
else:
    def zeta(_):
        """Blank function; do not calculate phase shift"""
        return
