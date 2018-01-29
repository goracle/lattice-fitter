"""Perform basic error check on the number of trials."""
import sys

from latfit.procargs import procargs


def trials_err(ntrials):
    """Check trials given on the command line for errors.
    Return -1 if trials are not specified.
    """
    sent1 = object()
    ntrials = sent1
    if isinstance(ntrials, str):
        try:
            ntrials = int(ntrials)
        except ValueError:
            print("***ERROR***")
            print("Invalid number of trials.")
            procargs(["h"])
    if ntrials == sent1:
        return -1
    if ntrials <= 0:
        print("Number of trials should be greater than one")
        print("Check command line input.")
        sys.exit(1)
    return ntrials
