from latfit.config import EFF_MASS
from numpy import ceil,floor

from latfit.extract.gevp_getfiles_onetime import gevp_getfiles_onetime
from latfit.extract.pencil_shift import pencil_shift_lhs,pencil_shift_rhs

from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS
 
if EFF_MASS:
    def gevp_getfiles(time,xstep,xmin):
        #time2=ceil(float(time)/2.0/xstep)*xstep if ceil(float(time)/2.0)!=time else max(floor(float(time)/2.0/xstep)*xstep,xmin)
        #time2=time-1
        time2=xmin-2*xstep
        #extract files
        if NUM_PENCILS < 1:
            FILES=gevp_getfiles_onetime(time)
            FILES2 =gevp_getfiles_onetime(time2)
            FILES3 = gevp_getfiles_onetime(time+xstep)
            FILES4 = gevp_getfiles_onetime(time+2*xstep)
        else:
            FILES=pencil_shift_lhs(time,xstep)
            FILES2=pencil_shift_rhs(time2,xstep)
            FILES3 =pencil_shift_lhs(time+xstep,xstep)
            FILES4 = pencil_shift_lhs(time+2*xstep,xstep)
        #eff mass stuff
        return time2,FILES,FILES2,FILES3,FILES4
else:
    def gevp_getfiles(time,xstep,xmin):
        #time2=ceil(float(time)/2.0/xstep)*xstep if ceil(float(time)/2.0)!=time else max(floor(float(time)/2.0/xstep)*xstep,xmin)
        time2=xmin-2*xstep
        #extract files
        if NUM_PENCILS < 1:
            FILES=gevp_getfiles_onetime(time)
            FILES2 =gevp_getfiles_onetime(time2)
        else:
            FILES=pencil_shift_lhs(time,xstep)
            FILES2=pencil_shift_rhs(time2,xstep)
        #eff mass stuff
        return time2,FILES,FILES2
