"""Logs stdout to file."""
import sys
import os
import time
import subprocess as sp
import mpi4py
from mpi4py import MPI

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

class Logger:
    """log output from fit"""
    @PROFILE
    def __init__(self, fname=None):
        """initialize logger"""
        self.terminal = sys.stdout
        if fname is None:
            self.log = open("fit.log", "a")
        else:
            self.log = open("fit_"+str(fname)+"_"+str(
                MPIRANK)+".log", "a")
        self.flush()

    @PROFILE
    def write(self, message):
        """write to log"""
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    @PROFILE
    def flush(self):
        """this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        self.terminal.flush()


@PROFILE
def setup_logger():
    """Setup the logger"""
    print("BEGIN NEW OUTPUT")
    timedate = time.asctime(time.localtime(time.time()))
    print(timedate)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    gitlog = sp.check_output(['git', 'rev-parse', 'HEAD'])
    with open(cwd+'/config.log', 'a') as conflog:
        conflog.write("BEGIN NEW OUTPUT-------------\n")
        conflog.write(timedate+'\n')
        conflog.write("current git commit:"+str(gitlog)+'\n')
        filen = open(os.getcwd()+'/config.py', 'r')
        for line in filen:
            conflog.write(line)
        conflog.write("END OUTPUT-------------------\n")
    if len(gitlog.split()) == 1:
        print("current git commit:", gitlog)
    os.chdir(cwd)
