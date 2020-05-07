#!/usr/bin/env pypy3
"""Fit function to data.
Compute chi^2 (t^2) and errors.
Plot fit with error bars.
Save result to pdf.
usage note: MAKE SURE YOU SET THE Y LIMITS of your plot by hand!
usage note(2): MAKE SURE as well that you correct the other "magic"
parts of the graph routine
"""

# install pip3
# then sudo pip3 install numdifftools

import sys
import signal
import mpi4py
from mpi4py import MPI

from latfit.mainfunc.tloop import tloop

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

def main():
    """main"""
    try:
        tloop()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, "")

def signal_handler(sig, frame):
    """Handle ctrl+c"""
    print('Ctrl+C pressed; raising.')
    raise RuntimeError
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("__main__.py should not be called directly")
    print("install first with python3 setup.py install")
    print("then run: latfit <args>")
    sys.exit(1)
