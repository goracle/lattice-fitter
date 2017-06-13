import sys

from latfit.procargs import procargs

def trials_err(ntrials):
    """Perform basic error check on the number of trials."""
    SENT1 = object()
    NTRIALS = SENT1
    if isinstance(ntrials, str):
        try:
            NTRIALS = int(ntrials)
        except ValueError:
            print("***ERROR***")
            print("Invalid number of trials.")
            procargs(["h"])
    if NTRIALS == SENT1:
        return -1
    if NTRIALS <= 0:
        print("Number of trials should be greater than one")
        print("Check command line input.")
        sys.exit(1)
    return NTRIALS
