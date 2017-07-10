from numpy import ceil,floor

from latfit.extract.gevp_getfiles_onetime import gevp_getfiles_onetime
from latfit.extract.pencil_shift import pencil_shift_lhs,pencil_shift_rhs
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder

from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS
from latfit.config import GEVP


if EFF_MASS:
    def getfiles_simple(time, xstep, input_f):
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
    def getfiles_simple(time, xstep, input_f):
        """Get files for a given time slice."""
        #extract file
        ijfile = proc_folder(input_f, time)
        #check for errors
        ijfile = pre_proc_file(ijfile, input_f)
        return (ijfile)

if EFF_MASS:
    def getfiles_gevp(time,XSTEP,xmin):
        time2=ceil(float(time)/2.0/XSTEP)*XSTEP if ceil(float(time)/2.0)!=time else max(floor(float(time)/2.0/XSTEP)*XSTEP,xmin)
        #extract files
        if NUM_PENCILS < 1:
            FILES=gevp_getfiles_onetime(time)
            FILES2 =gevp_getfiles_onetime(time2)
            FILES3 = gevp_getfiles_onetime(time+XSTEP)
            FILES4 = gevp_getfiles_onetime(time+2*XSTEP)
        else:
            FILES=pencil_shift_lhs(time,XSTEP)
            FILES2=pencil_shift_rhs(time2,XSTEP)
            FILES3 =pencil_shift_lhs(time+XSTEP,XSTEP)
            FILES4 = pencil_shift_lhs(time+2*XSTEP,XSTEP)
        #eff mass stuff
        return time2,(FILES,FILES2,FILES3,FILES4)
else:
    def getfiles_gevp(time,XSTEP,xmin):
        time2=ceil(float(time)/2.0/XSTEP)*XSTEP if ceil(float(time)/2.0)!=time else max(floor(float(time)/2.0/XSTEP)*XSTEP,xmin)
        #extract files
        if NUM_PENCILS < 1:
            FILES=gevp_getfiles_onetime(time)
            FILES2 =gevp_getfiles_onetime(time2)
        else:
            FILES=pencil_shift_lhs(time,XSTEP)
            FILES2=pencil_shift_rhs(time2,XSTEP)
        #eff mass stuff
        return time2,(FILES,FILES2)

if GEVP:
    def getfiles(time, xstep, xmin, input_f):
        return getfiles_gevp(time, xstep, xmin)
else:
    def getfiles(time, xstep, xmin, input_f):
        return getfiles_simple(time, xstep, input_f)
