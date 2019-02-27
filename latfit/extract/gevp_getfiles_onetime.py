"""Get GEVP files for a time slice."""
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder

from latfit.config import GEVP_DIRS


def gevp_getfiles_onetime(time):
    """Get matrix of files for a particular time slice
    (read from the gevp directories)
    """
    rdimops = range(len(GEVP_DIRS))
    files = [[proc_folder(GEVP_DIRS[op1][op2], time, opdim=(op1,op2))
              for op1 in rdimops] for op2 in rdimops]
    return [[pre_proc_file(files[op1][op2], GEVP_DIRS[op1][op2])
             for op1 in rdimops] for op2 in rdimops]
