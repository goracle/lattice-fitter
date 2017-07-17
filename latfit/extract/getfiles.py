"""Get files"""
#from numpy import ceil, floor

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
        #extract file
        ijfile = proc_folder(input_f, time)
        #check for errors
        ijfile = pre_proc_file(ijfile, input_f)
        ij2file = proc_folder(input_f, time+xstep)
        ij3file = proc_folder(input_f, time+2*xstep)
        ij2file = pre_proc_file(ij2file, input_f)
        ij3file = pre_proc_file(ij3file, input_f)
        return (ijfile, ij2file, ij3file)

else:
    def getfiles_simple(time, input_f, *args):
        """Get files for a given time slice."""
        #extract file
        ijfile = proc_folder(input_f, time)
        #check for errors
        ijfile = pre_proc_file(ijfile, input_f)
        if args:
            pass
        return tuple(ijfile)

if EFF_MASS:
    def getfiles_gevp(time, time2, xstep):
        """Get files, gevp, eff_mass"""
        #extract files
        if NUM_PENCILS < 1:
            files = gevp_getfiles_onetime(time)
            files2 = gevp_getfiles_onetime(time2)
            files3 = gevp_getfiles_onetime(time+xstep)
            files4 = gevp_getfiles_onetime(time+2*xstep)
        else:
            files = pencil_shift_lhs(time, xstep)
            files2 = pencil_shift_rhs(time2, xstep)
            files3 = pencil_shift_lhs(time+xstep, xstep)
            files4 = pencil_shift_lhs(time+2*xstep, xstep)
        #eff mass stuff
        return (files, files2, files3, files4)
else:
    def getfiles_gevp(time, time2, xstep):
        """Get files, gevp"""
        #extract files
        if NUM_PENCILS < 1:
            files = gevp_getfiles_onetime(time)
            files2 = gevp_getfiles_onetime(time2)
        else:
            files = pencil_shift_lhs(time, xstep)
            files2 = pencil_shift_rhs(time2, xstep)
        #eff mass stuff
        return (files, files2)

if GEVP:
    def getfiles(time, xstep, xmin, input_f):
        """Get files, gevp (meta)"""
        if input_f:
            pass
        #time2 = ceil(float(time)/2.0/xstep)*xstep if ceil(
        #    float(time)/2.0) != time else max(
        #        floor(float(time)/2.0/xstep)*xstep, xmin)
        time2 = xmin-2*xstep
        return time2, getfiles_gevp(time, xstep, xmin, time2)
else:
    def getfiles(time, xstep, xmin, input_f):
        """Get files, (meta)"""
        if xmin:
            pass
        return getfiles_simple(time, input_f, xstep)
