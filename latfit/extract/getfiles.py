"""Get files"""
from numpy import ceil, floor

from latfit.extract.gevp_getfiles_onetime import gevp_getfiles_onetime
from latfit.extract.pencil_shift import pencil_shift_lhs, pencil_shift_rhs
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder

from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS
from latfit.config import GEVP


if EFF_MASS:
    def getfiles_simple(time, input_f, xstep):
        """Get files for a given time slice."""
        # extract file
        ijfile = proc_folder(input_f, time)
        # check for errors
        ijfile = pre_proc_file(ijfile, input_f)
        ij2file = proc_folder(input_f, time+xstep)
        ij3file = proc_folder(input_f, time+2*xstep)
        ij4file = proc_folder(input_f, time+3*xstep)
        ij2file = pre_proc_file(ij2file, input_f)
        ij3file = pre_proc_file(ij3file, input_f)
        ij4file = pre_proc_file(ij4file, input_f)
        return (ijfile, ij2file, ij3file, ij4file)

else:
    def getfiles_simple(time, input_f, _):
        """Get files for a given time slice."""
        # extract file
        ijfile = proc_folder(input_f, time)
        # check for errors
        ijfile = pre_proc_file(ijfile, input_f)
        return ijfile

if EFF_MASS:
    def getfiles_gevp(time, time2, xstep):
        """Get files, gevp, eff_mass"""
        # extract files
        if NUM_PENCILS < 1:
            files = gevp_getfiles_onetime(time)
            files2 = gevp_getfiles_onetime(time2)
            files3 = gevp_getfiles_onetime(time+xstep)
            files4 = gevp_getfiles_onetime(time+2*xstep)
            files5 = gevp_getfiles_onetime(time+3*xstep)
        else:
            files = pencil_shift_lhs(time, xstep)
            files2 = pencil_shift_rhs(time2, xstep)
            files3 = pencil_shift_lhs(time+xstep, xstep)
            files4 = pencil_shift_lhs(time+2*xstep, xstep)
            files5 = pencil_shift_lhs(time+3*xstep, xstep)
        # eff mass stuff
        return (files, files2, files3, files4, files5)
else:
    def getfiles_gevp(time, time2, xstep):
        """Get files, gevp"""
        # extract files
        if NUM_PENCILS < 1:
            files = gevp_getfiles_onetime(time)
            files2 = gevp_getfiles_onetime(time2)
        else:
            files = pencil_shift_lhs(time, xstep)
            files2 = pencil_shift_rhs(time2, xstep)
        # eff mass stuff
        return (files, files2)

def roundup(time, xstep, xmin):
    """ceil(t/2) with xstep factored in"""
    time2 = ceil(float(time)/2.0/xstep)*xstep if ceil(
        float(time)/2.0) != time else max(
            floor(float(time)/2.0/xstep)*xstep, xmin)
    return time2

if GEVP:
    def getfiles(time, xstep, xmin, _):
        """Get files, gevp (meta)"""
        time2 = 3
        time2 = roundup(time, xstep, xmin)
        return getfiles_gevp(time, time2, xstep)
else:
    def getfiles(time, xstep, _, input_f):
        """Get files, (meta)"""
        return getfiles_simple(time, input_f, xstep)
