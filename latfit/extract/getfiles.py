"""Get files"""
import math
import copy
import numpy as np

from latfit.extract.gevp_getfiles_onetime import gevp_getfiles_onetime
from latfit.extract.pencil_shift import pencil_shift_lhs, pencil_shift_rhs
from latfit.extract.pre_proc_file import pre_proc_file
from latfit.extract.proc_folder import proc_folder

from latfit.config import EFF_MASS
from latfit.config import NUM_PENCILS
from latfit.config import GEVP, IRREP
from latfit.config import MATRIX_SUBTRACTION
from latfit.config import DELTA_E_AROUND_THE_WORLD, DELTA_E2_AROUND_THE_WORLD
import latfit.config

if DELTA_E2_AROUND_THE_WORLD is not None:
    assert 'mom000' not in IRREP,\
        "only one subtraction required for center of mass frame"

if EFF_MASS:
    def getfiles_simple(time, input_f, xstep=1):
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

def get_extra_times(time, time2, dt1, xstep=1):
    """get extra time slices for around the world subtraction
    """
    if EFF_MASS:
        extra_lhs_times = [time-dt1, time-dt1+xstep, time-dt1+2*xstep,
                           time-dt1+3*xstep]
    else:
        extra_lhs_times = [time-dt1]
    ret = []
    for timei in extra_lhs_times:
        ret.append(max(timei, 0))
    return ret, max(time2-dt1, 0)

def getfiles_gevp(time, time2, xstep=1):
    """Loop over time2"""
    rhs_files = []
    if hasattr(time2, '__iter__'):
        tupret = getfiles_gevp_singlerhs(time, time2[0], xstep)
        assert tupret[1] is not None, "rhs files not found."
        for time_rhs in time2:
            assert time >= time_rhs, "Bad rhs time. t values (lhs, rhs):"+\
                str(time)+", "+str(time_rhs)
            if time_rhs not in getfiles_gevp_singlerhs.mats:
                tuptemp = getfiles_gevp_singlerhs(time, time_rhs, xstep)
                file_rhs = tuptemp[1]
            else:
                file_rhs = getfiles_gevp_singlerhs.mats[time_rhs]
            assert file_rhs is not None, "rhs files not found."
            rhs_files.append(file_rhs)
        assert len(time2) == len(rhs_files), "rhs times dimension mismatch"
        ret = (tupret[0], rhs_files, *tupret[2:])
    else:
        assert time >= time2, "Bad rhs time. t values (lhs, rhs):"+\
            str(time)+", "+str(time2)+", "+str(xstep)
        ret = getfiles_gevp_singlerhs(time, time2, xstep)
    assert len(ret) == 5, "return tuple is wrong length:"+str(len(ret))
    return ret

def getfiles_gevp_singlerhs(time, time2, xstep=1):
    """Get files, gevp, eff_mass for single t0 (rhs time)"""
    # extract files
    files = {}
    sub = {}
    dt1 = latfit.config.DELTA_T_MATRIX_SUBTRACTION*xstep
    dt2 = latfit.config.DELTA_T2_MATRIX_SUBTRACTION*xstep
    extra_lhs_times, extra_rhs_time = get_extra_times(time, time2,
                                                      dt1, xstep)
    extra_lhs_times2, extra_rhs_time2 = get_extra_times(time, time2,
                                                        dt2, xstep)
    extra_lhs_times = [*extra_lhs_times, *extra_lhs_times2]
    extra_rhs_times = [extra_rhs_time, extra_rhs_time2]
    if NUM_PENCILS < 1:
        files[time] = gevp_getfiles_onetime(time, chkpos=True)
        files[time2] = gevp_getfiles_onetime(time2)
        if EFF_MASS:
            files[time+xstep] = gevp_getfiles_onetime(time+xstep)
            files[time+2*xstep] = gevp_getfiles_onetime(time+2*xstep)
            files[time+3*xstep] = gevp_getfiles_onetime(time+3*xstep)
        for timeidx in [*extra_lhs_times, *extra_rhs_times]:
            sub[timeidx] = gevp_getfiles_onetime(timeidx) if\
                timeidx not in files and MATRIX_SUBTRACTION else None
    else:
        files[time] = pencil_shift_lhs(time, xstep)
        files[time2] = pencil_shift_rhs(time2, xstep)
        if EFF_MASS:
            files[time+xstep] = pencil_shift_lhs(time+xstep, xstep)
            files[time+2*xstep] = pencil_shift_lhs(time+2*xstep, xstep)
            files[time+3*xstep] = pencil_shift_lhs(time+3*xstep, xstep)
        if MATRIX_SUBTRACTION:
            for timeidx in extra_lhs_times:
                if timeidx not in files:
                    sub[timeidx] = pencil_shift_lhs(timeidx, xstep)
            for timeidx in extra_rhs_times:
                if timeidx not in files:
                    sub[timeidx] = pencil_shift_rhs(timeidx, xstep)

    # do matrix subtraction to eliminate leading order around the world term
    if MATRIX_SUBTRACTION:
        files, sub = matsub(files, sub, dt1)
        files, sub = matsub(files, sub, dt2, 'Two') if\
            DELTA_E2_AROUND_THE_WORLD is not None else (files, sub)

    ret = (files[time], files[time2], files[time+xstep],
           files[time+2*xstep], files[time+3*xstep]) if EFF_MASS else (
               files[time], files[time2])

    #for savedt in files:
    #    break
    #getfiles_gevp_singlerhs.mats[savedt] = copy.deepcopy(np.array(
    #files[savedt])) if savedt not in getfiles_gevp_singlerhs.mats\
    #        else getfiles_gevp_singlerhs.mats[savedt]
    return ret
getfiles_gevp_singlerhs.mats = {}

def file_reset():
    """Reset file cache"""
    getfiles_gevp_singlerhs.mats = {}

def mult_sub_by_delta(sub_timeidx, delta_e, timeidx):
    """Multiply subtraction term by exponential to shift energy"""
    if sub_timeidx is not None:
        sub_timeidx = copy.deepcopy(np.asarray(sub_timeidx))
        if hasattr(delta_e, '__iter__'):
            for i, _ in enumerate(delta_e):
                sub_timeidx[:, :, i] *= math.exp(delta_e[i]*timeidx)
        else:
            sub_timeidx *= math.exp(delta_e*timeidx)
    return sub_timeidx

def matsub(files, sub, dt1, dt12='One'):
    """Do the around the world subtraction"""
    subterm = {}
    assert MATRIX_SUBTRACTION
    delta_e = list(DELTA_E_AROUND_THE_WORLD) if dt12 == 'One' else\
        list(DELTA_E2_AROUND_THE_WORLD)
    for timeidx in sub:
        sub[timeidx] = mult_sub_by_delta(sub[timeidx], delta_e, timeidx)
    for timeidx in files:
        files[timeidx] = copy.deepcopy(np.asarray(files[timeidx]))
        if hasattr(delta_e, '__iter__'):
            for i, _ in enumerate(delta_e):
                files[timeidx][:, :, i] *= math.exp(delta_e[i]*timeidx)
        else:
            files[timeidx] *= math.exp(delta_e*timeidx)
    for timeidx in files:
        subidx = max(timeidx-dt1, 0)
        subterm[timeidx] = files[subidx] if subidx in files else sub[subidx]
        subterm[timeidx] = copy.deepcopy(subterm[timeidx])
        #print("C_weighted(", timeidx, ") -= C_weighted(", subidx, ")", "check with delta_t=", dt1)
    for timeidx in files:
        subtraction = copy.deepcopy(subterm[timeidx])
        files[timeidx] -= subtraction
    assert isinstance(files, dict)
    return (files, sub)


def roundup(time, xstep, xmin):
    """ceil(t/2) with xstep factored in"""
    time2 = int(time/2/xstep)*xstep
    time2 = np.ceil(float(time)/2.0/xstep)*xstep if np.ceil(
        float(time)/2.0) != time else max(
            np.floor(float(time)/2.0/xstep)*xstep, xmin)
    #time2 = 2 if time == 5 else time2
    #time2 = 9 if time == 11 else time2
    #time2 = time-2 if time >= 11 else time2
    return time2

if GEVP:
    def getfiles(time, xstep, xmin, _):
        """Get files, gevp (meta)"""
        if latfit.config.T0 == 'ROUND':
            time2 = roundup(time, xstep, xmin)
        elif 'TMINUS' in latfit.config.T0:
            time2 = time-xstep*int(latfit.config.T0[6:])
        elif isinstance(latfit.config.T0, int):
            time2 = latfit.config.T0
        elif latfit.config.T0 == 'LOOP':
            time2 = sorted(list(np.arange(roundup(
                time, xstep, xmin), time, xstep)), reverse=True)
            if not len(time2) > 1:
                time2 = roundup(time, xstep, xmin)
            #time2 = 4 if time == 5 else time2
        delta_t = time-time2
        ret = getfiles_gevp(time, time2, xstep), delta_t
        assert len(ret[0]) == 5, "bad file_tup length:"+str(len(ret[0]))
        return ret
else:
    def getfiles(time, xstep, _, input_f):
        """Get files, (meta)"""
        return getfiles_simple(time, input_f, xstep), None
