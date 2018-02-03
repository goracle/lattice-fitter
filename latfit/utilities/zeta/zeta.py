"""Calculate phase shift from Luscher zeta for given inputs."""
import sys
import subprocess
from latfit.config import PION_MASS, L_BOX, CALC_PHASE_SHIFT

if CALC_PHASE_SHIFT:
    def zeta(epipi):
        """Calculate the I=0 scattering phase shift given the pipi energy
        for that channel.
        """
        arglist = ["./main.o", str(epipi), str(PION_MASS), str(L_BOX)]
        try:
            out = subprocess.check_output(arglist)
        except FileNotFoundError:
            print("Error in zeta: main.C not compiled yet.")
        except subprocess.CalledProcessError:
            print("Error in zeta: calc of phase shift error:")
            print(subprocess.Popen(arglist,
                                   stdout=subprocess.PIPE).stdout.read())
            sys.exit(1)
        return float(out)
else:
    def zeta(_):
        """Blank function; do not calculate phase shift"""
        return
