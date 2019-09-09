"""Logs stdout to file."""
import sys
import os
import time
import subprocess as sp

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

class Logger(object):
    """log output from fit"""
    @PROFILE
    def __init__(self):
        """initialize logger"""
        self.terminal = sys.stdout
        self.log = open("fit.log", "a")

    @PROFILE
    def write(self, message):
        """write to log"""
        self.terminal.write(message)
        self.log.write(message)

    @PROFILE
    def flush(self):
        """this flush method is needed for python 3 compatibility.
        this handles the flush command by doing nothing.
        you might want to specify some extra behavior here.
        """
        pass


sys.stdout = Logger()
sys.stderr = Logger()



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
