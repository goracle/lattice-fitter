"""Calculate phase shift from Luscher zeta for given inputs."""
import sys
import subprocess
import inspect
import os
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT, START_PARAMS

if CALC_PHASE_SHIFT:
    def zeta(epipi):
        """Calculate the I=0 scattering phase shift given the pipi energy
        for that channel.
        """
        try:
            epipi = epipi[1]
        except IndexError:
            pass
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
            print(subprocess.Popen(arglist,
                                   stdout=subprocess.PIPE).stdout.read())
            sys.exit(1)
        return float(out)
else:
    def zeta(_):
        """Blank function; do not calculate phase shift"""
        return
