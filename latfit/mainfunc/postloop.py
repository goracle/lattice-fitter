"""Actions peformed after the fit"""
import sys
import re
from collections import namedtuple
import pickle
import numpy as np
import mpi4py
from mpi4py import MPI

# config
from latfit.config import EFF_MASS, MULT, GEVP, SUPERJACK_CUTOFF
from latfit.config import TLOOP, METHOD, ALTERNATIVE_PARALLELIZATION
from latfit.config import NOLOOP, UNCORR, LATTICE_ENSEMBLE
from latfit.config import SYS_ENERGY_GUESS
from latfit.checks.consistency import check_include

from latfit.makemin.mkmin import convert_to_namedtuple
from latfit.analysis.result_min import Param
from latfit.analysis.filename_windows import filename_plus_config_info
from latfit.finalout.printerr import printerr

# dynamic
import latfit.extract.extract as ext
from latfit.utilities import exactmean as em
import latfit.singlefit as sfit
import latfit.config
import latfit.finalout.mkplot as mkplot
import latfit.mainfunc.print_res as print_res
from latfit.analysis.superjack import jack_mean_err

# errors
from latfit.analysis.errorcodes import NegChisq
from latfit.analysis.errorcodes import RelGammaError, ZetaError
from latfit.analysis.errorcodes import BadChisq, FitFail
from latfit.analysis.errorcodes import BadJackknifeDist, NoConvergence
from latfit.analysis.errorcodes import EnergySortError, TooManyBadFitsError


try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

# for final representative fit.  dof should be deterministic, so should work
# (at least)
ACCEPT_ERRORS_FIN = (NegChisq, RelGammaError, NoConvergence,
                     OverflowError, EnergySortError, TooManyBadFitsError,
                     BadJackknifeDist, BadChisq, ZetaError)

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
mpi4py.rc.recv_mprobe = False

DOWRITE = ALTERNATIVE_PARALLELIZATION and not\
    MPIRANK or not ALTERNATIVE_PARALLELIZATION


def write_pickle_file_verb(filename, arr):
    """Write pickle file; print info"""
    # assert not os.path.exists(filename+'.p'), filename+'.p'
    if DOWRITE:
        print("writing pickle file", filename)
        pickle.dump(arr, open(filename+'.p', "wb"))

@PROFILE
def divbychisq(param_arr, pvalue_arr):
    """Divide a parameter by chisq (t^2)"""
    assert not any(np.isnan(pvalue_arr)), "pvalue array contains nan"
    ret = np.array(param_arr)
    assert ret.shape, str(ret)
    if len(ret.shape) > 1:
        assert param_arr[:, 0].shape == pvalue_arr.shape,\
            "Mismatch between pvalue_arr"+\
            " and parameter array (should be the number of configs):"+\
            str(pvalue_arr.shape)+", "+str(param_arr.shape)
        for i in range(len(ret[0])):
            try:
                assert not any(np.isnan(param_arr[:, i])),\
                    "parameter array contains nan"
            except AssertionError:
                print("found nan in dimension :", i)
                for j in param_arr:
                    print(j)
                sys.exit(1)
            ret[:, i] *= pvalue_arr
            assert not any(np.isnan(ret[:, i])),\
                "parameter array contains nan"
    else:
        try:
            assert not np.any(np.isnan(param_arr)),\
                "parameter array contains nan"
        except AssertionError:
            for i in param_arr:
                print(i)
            raise
        except TypeError:
            print("param_arr=", param_arr)
            raise
        try:
            ret *= pvalue_arr
        except ValueError:
            print("could not be broadcast together")
            print("ret=", ret)
            print("pvalue_arr=", pvalue_arr)
            raise
    assert ret.shape == param_arr.shape,\
        "return shape does not match input shape"
    return ret


@PROFILE
def closest_fit_to_avg(result_min_avg, min_arr):
    """Find closest fit to average fit
    (find the most common fit range)
    """
    minmax = np.nan
    ret_excl = []
    for i, fiti in enumerate(min_arr):
        minmax_i = max(abs(fiti[0].energy.val-result_min_avg))
        if i == 0:
            minmax = minmax_i
            ret_excl = list(fiti[2])
        else:
            minmax = min(minmax_i, minmax)
            if minmax == minmax_i:
                ret_excl = list(fiti[2])
    return ret_excl


@PROFILE
def dump_fit_range(meta, min_arr, name, res_mean, err_check):
    """Pickle the fit range result"""
    if DOWRITE:
        print("starting arg:", name)
    if 'energy' in name: # no clobber (only do this once)
        if MULT > 1:
            for i in range(len(res_mean)):
                dump_min_err_jackknife_blocks(meta, min_arr, mindim=i)
        else:
            dump_min_err_jackknife_blocks(meta, min_arr)
    pickl_res = pickle_res(name, min_arr)
    pickl_res_err = pickle_res_err(name, min_arr)
    pickl_excl = pickle_excl(meta, min_arr)
    pickl_res_fill = np.empty(4, object)
    try:
        pickl_res_fill[:] = [res_mean, err_check, pickl_res, pickl_excl]
        pickl_res = pickl_res_fill
    except ValueError:
        print(np.asarray(res_mean).shape)
        print(np.asarray(err_check).shape)
        print(np.asarray(pickl_res).shape)
        print(np.asarray(pickl_excl).shape)
        print(name)
        raise
    assert pickl_res_err.shape == pickl_res[2].shape[0::2], (
        "array mismatch:", pickl_res_err.shape, pickl_res[2].shape)
    assert len(pickl_res) == 4, "bad result length"

    if not GEVP:
        if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
            name = name+'_'+dump_fit_range.fn1
        name = re.sub('.jkdat', '', name)
    filename = filename_plus_config_info(meta, name)
    filename_err = filename_plus_config_info(meta, name+'_err')
    write_pickle_file_verb(filename, pickl_res)
    write_pickle_file_verb(filename_err, pickl_res_err)
dump_fit_range.fn1 = None

@PROFILE
def pickle_excl(meta, min_arr):
    """Get the fit ranges to be pickled
    append the effective mass points
    """
    ret = [print_res.inverse_excl(meta, i[2]) for i in min_arr]
    if DOWRITE:
        print("original number of fit ranges before effective mass append:",
              len(ret))
    if EFF_MASS:
        xcoord = list(sfit.singlefit.coords_full[:, 0])
        xcoordapp = [[i] for i in xcoord]
        ret = [*ret, *xcoordapp]
    ret = np.array(ret, dtype=object)
    if DOWRITE:
        print("final fit range amount:", len(ret))
    return ret

@PROFILE
def pickle_res_err(name, min_arr):
    """Append the effective mass errors to the """
    ret = [getattr(i[0], name).err for i in min_arr]
    if DOWRITE:
        print("debug:[getattr(i[0], name) for i in min_arr].shape",
              np.asarray(ret).shape)
        print("debug2:", np.asarray(sfit.singlefit.error2).shape)
    origl = len(ret)
    if GEVP and 'systematics' not in name:
        if len(np.asarray(ret).shape) > 1:
            # dimops check
            dimops1 = (np.array(ret).shape)[1]
            if name == 'min_params' and SYS_ENERGY_GUESS:
                dimops1 = int((dimops1-1)/2)
            dimops2 = (np.asarray(sfit.singlefit.error2).shape)[1]
            assert dimops1 == dimops2, (np.array(ret).shape,
                                        np.asarray(sfit.singlefit.error2),
                                        name)
    if 'energy' in name:
        _, erreff = min_eff_mass_errors(getavg=True)
        ret = np.array([*ret, *erreff])
    ret = np.asarray(ret)
    flen = len(ret)
    if DOWRITE:
        print("original error length (err):", origl,
              "final error length:", flen)
    return ret

@PROFILE
def pickle_res(name, min_arr):
    """Return the fit range results to be pickled,
    append the effective mass points
    """
    ret = [getattr(i[0], name).arr for i in min_arr]
    origlshape = np.asarray(ret, dtype=object).shape
    if DOWRITE:
        print("res shape", origlshape)
    origl = len(ret)
    if 'energy' in name:
        arreff, _ = min_eff_mass_errors()
        ret = [*ret, *arreff]
    ret = np.asarray(ret, dtype=object)
    assert len(origlshape) == len(ret.shape), str(origlshape)+","+str(
        ret.shape)
    flen = len(ret)
    if DOWRITE:
        print("original error length (res):", origl,
              "final error length:", flen)
    return ret

@PROFILE
def dump_min_err_jackknife_blocks(meta, min_arr, mindim=None):
    """Dump the jackknife blocks for the energy with minimum errors"""
    fname = "energy_min_"+str(LATTICE_ENSEMBLE)
    if dump_fit_range.fn1 is not None and dump_fit_range.fn1 != '.':
        fname = fname + '_'+dump_fit_range.fn1
    name = 'energy'
    err = np.array([getattr(i[0], name).err for i in min_arr])
    dimops = err.shape[1]
    if dimops == 1:
        err = err[:, 0]
        errmin = min(err)
        ind = list(err).index(min(err))
        arr = getattr(min_arr[ind][0], 'energy').arr
    else:
        assert mindim is not None, "needs specification of operator"+\
            " dimension to write min error jackknife blocks (unsupported)."
        if DOWRITE:
            print(err.shape)
        errmin = min(err[:, mindim])
        ind = list(err[:, mindim]).index(errmin)
        fname = fname+'_mindim'+str(mindim)
        arr = np.asarray(getattr(min_arr[ind][0], 'energy').arr[:, mindim])
    arr, errmin = compare_eff_mass_to_range(arr, errmin, mindim=mindim)

    fname = filename_plus_config_info(meta, fname)
    if DOWRITE:
        print("dumping jackknife energies with error:", errmin,
              "into file:", fname+'.p')
    # assert not os.path.exists(fname+'.p'), fname+'.p'
    if DOWRITE:
        pickle.dump(arr, open(fname+'.p', "wb"))

@PROFILE
def compare_eff_mass_to_range(arr, errmin, mindim=None):
    """Compare the error of err to the effective mass errors.
    In other words, find the minimum error of
    the errors on subsets of effective mass points
    and the error on the points themselves.
    """
    arreff, erreff = min_eff_mass_errors(mindim=mindim)
    if errmin == erreff:
        arr = arreff
    else:
        errmin = min(errmin, erreff)
        if errmin == erreff:
            arr = arreff
    # the error is not going to be a simple em.acstd if we do sample AMA
    # so skip the check in this case
    if not SUPERJACK_CUTOFF:
        errcheck = em.acstd(arr)*np.sqrt(len(arr-1))
        try:
            assert np.allclose(errcheck, errmin, rtol=1e-6)
        except AssertionError:
            print("error check failed")
            print(errmin, errcheck)
            sys.exit(1)
    return arr, errmin

def time_slice_list():
    """Get list of time slices from reuse dictionary keys"""
    # time slice list
    times = []
    for key in sfit.singlefit.reuse:
        if not isinstance(key, float) and not isinstance(key, int):
            continue
        if int(key) == key:
            times.append(int(key))
    times = sorted(times)
    return times

def get_dimops_at_time(time1):
    """ get the dimension of the GEVP (should be == MULT)
    (also, dimops should be == first config's energy values at time1)"""
    dimops = len(sfit.singlefit.reuse[time1][0])
    assert dimops == MULT, (dimops, MULT)
    assert dimops == len(sfit.singlefit.reuse[time1][0])


if EFF_MASS:
    @PROFILE
    def min_eff_mass_errors(mindim=None, getavg=False):
        """Append the errors of the effective mass points
        to errarr"""

        # time slice list
        xcoord = list(sfit.singlefit.coords_full[:, 0])
        assert mindim is None or isinstance(mindim, int),\
            "type check failed"

        # build the time slice lists of eff mass, errors
        dimops = None
        errlist = []
        arrlist = []
        for _, time1 in enumerate(time_slice_list()):

            # we may have extra time slices
            # (due to fit window cuts from TLOOP)
            if time1 not in xcoord:
                continue

            # check dimensionality
            dimops = get_dimops_at_time(time1) if dimops is None else dimops
            assert dimops == get_dimops_at_time(time1), (
                get_dimops_at_time(time1), dimops)

            # masses and errors at this time slice
            arr = sfit.singlefit.reuse[time1]
            err = sfit.singlefit.error2[xcoord.index(time1)]

            # reduce to a specific GEVP dimension
            if mindim is not None:
                arr = arr[:, mindim]
                err = err[mindim]
                assert isinstance(err, float), str(err)
            arrlist.append(arr)
            errlist.append(err)

        if getavg and mindim is None:
            arr, err = config_avg_eff_mass(arrlist, errlist)
        elif not getavg and mindim is not None:
            arr, err = mindim_eff_mass(arrlist, errlist, xcoord)
        elif not getavg and mindim is None:
            arr, err = np.array(arrlist), np.array(errlist)
        else:
            assert None, ("mode check fail:", getavg, mindim)

        assert isinstance(err, float) or mindim is None, "index bug"
        return arr, err

    def mindim_eff_mass(arrlist, errlist, xcoord):
        """find the time slice which gives the minimum error
        then get the energy and error at that point
        this only makes sense if we are at a particular GEVP dimension
        """
        assert isinstance(errlist[0], float),\
            (errlist, " ", sfit.singlefit.error2[xcoord.index(10)],
             np.asarray(errlist).shape, np.asarray(errlist[0]).shape)
        err = min(errlist)
        arr = arrlist[errlist.index(err)]
        return arr, err

    def config_avg_eff_mass(arrlist, errlist):
        """Get the config average of the effective mass points.
        Add structure to non-GEVP points to make files
        dumped like a 1x1 GEVP
        """
        err = np.asarray(errlist)
        # average over configs
        arr = em.acmean(np.asarray(arrlist), axis=1)

        # add structure in arr for backwards compatibility
        if isinstance(arr[0], float): # no GEVP
            assert MULT == 1, MULT
            arr = np.asarray([[i] for i in arr])
            assert isinstance(err[0], float),\
                "error array does not have same structure as"+\
                " eff mass array"
            err = np.asarray([[i] for i in err])

        # index checks
        assert len(arr.shape) == 2, (
            arr, "first dim is time, second dim is operator", arr.shape)
        assert len(err.shape) == 2, (
            err, "first dim is time, second dim is operator", err.shape)
        assert len(err) == len(arr), (len(err), len(arr))
        return arr, err

else:
    @PROFILE
    def min_eff_mass_errors(_):
        """blank"""
        return (None, np.inf)

def dump_single_fit(meta, min_arr):
    """Dump result if there's only one"""
    # dump the results to file
    # if not (ISOSPIN == 0 and GEVP):
    for name in min_arr[0][0].__dict__:

        if mean_and_err_loop_continue(name, min_arr):
            continue

        res_mean, err_check = get_first_res(name, min_arr)

        # presumably we don't want to save files in this situation
        # probably there should be a separate switch
        if not NOLOOP:
            dump_fit_range(meta, min_arr, name, res_mean, err_check)

@PROFILE
def find_mean_and_err(meta, min_arr):
    """Find the mean and error from results of fit"""
    result_min = {}
    weight_sum = em.acsum([getattr(
        i[0], "pvalue").arr for i in min_arr], axis=0)
    for name in min_arr[0][0].__dict__:

        if mean_and_err_loop_continue(name, min_arr):
            continue

        # find the name of the array
        if DOWRITE:
            print("finding error in", name, "which has shape=",
                  np.asarray(min_arr[0][0].__dict__[name].val).shape)

        # compute the jackknife errors as a check
        # (should give same result as error propagation)
        res_mean, err_check = jack_mean_err(em.acsum([
            divbychisq(getattr(i[0], name).arr, getattr(
                i[0], 'pvalue').arr/weight_sum) for i in min_arr], axis=0))

        # dump the results to file
        # if not (ISOSPIN == 0 and GEVP):
        if len(min_arr) > 1 or (meta.lenprod == 1 and len(min_arr) == 1):
            if not NOLOOP:
                dump_fit_range(meta, min_arr, name, res_mean, err_check)

        # error propagation check
        result_min = parametrize_entry(result_min, name)
        result_min[name].err = fill_err_array(min_arr, name, weight_sum)
        try:
            result_min[name].err = np.sqrt(result_min[name].err)
        except FloatingPointError:
            print("floating point error in", name)
            print(result_min[name].err)
            if hasattr(result_min[name].err, '__iter__'):
                for i, res in enumerate(result_min[name].err):
                    if np.isreal(res) and res < 0:
                        result_min[name].err[i] = np.nan
            else:
                if np.isreal(result_min[name].err):
                    if res < 0:
                        result_min[name].err = np.nan
            result_min[name].err = np.sqrt(result_min[name].err)

        # perform the comparison
        try:
            assert np.allclose(
                err_check, result_min[name].err, rtol=1e-8)
        except AssertionError:
            print("jackknife error propagation"+\
                    " does not agree with jackknife"+\
                    " error.")
            print(result_min[name].err)
            print(err_check)
            if hasattr(err_check, '__iter__'):
                for i, ress in enumerate(zip(
                        result_min[name].err, err_check)):
                    res1, res2 = ress
                    print(res1, res2, np.allclose(res1, res2,
                                                  rtol=1e-8))
                    if not np.allclose(res1, res2, rtol=1e-8):
                        result_min[name][i].err = np.nan
                        err_check[i] = np.nan

    # find the weighted mean
        result_min[name].val = em.acsum(
            [getattr(i[0], name).val*getattr(i[0], 'pvalue').val
             for i in min_arr],
            axis=0)/em.acsum([getattr(i[0], 'pvalue').val
                              for i in min_arr])
    param_err = np.array(result_min['energy'].err)
    assert not any(np.isnan(param_err)), \
        "A parameter error is not a number (nan)"
    return result_min

def mean_and_err_loop_continue(name, min_arr):
    """should we continue in the loop
    """
    ret = False
    if 'misc' in name or '__paramlist' in name:
        ret = True
    else:
        try:
            val, _ = get_first_res(name, min_arr)
        except AttributeError:
            print("a Param got overwritten.  name:", name)
            raise
        if val is None:
            ret = True
    return ret

@PROFILE
def combine_results(result_min, result_min_close,
                    meta, param_err_avg, param_err_close):
    """use the representative fit's goodness of fit in final print
    """
    if meta.skip_loop() or not isinstance(result_min, dict):
        result_min, param_err_avg = result_min_close, param_err_close
    else:
        result_min['chisq'].val = result_min_close.chisq.val
        result_min['chisq'].err = result_min_close.chisq.err
        result_min['misc'] = result_min_close.misc
        result_min['pvalue'] = result_min_close.pvalue
        #result_min['pvalue'].err = result_min_close.pvalue.err
        if DOWRITE:
            print("closest representative fit result (lattice units):")
        # convert here since we can't set attributes afterwards
        result_min = convert_to_namedtuple(result_min)
        printerr(result_min_close.energy.val, param_err_close)
        print_res.print_phaseshift(result_min_close)
    return result_min, param_err_avg


@PROFILE
def loop_result(min_arr, overfit_arr):
    """Test if fit range loop succeeded"""
    if min_arr and DOWRITE:
        print(min_arr[0], np.array(min_arr).shape)
    min_arr = collapse_filter(min_arr)
    if overfit_arr and DOWRITE:
        print(overfit_arr[0])
    overfit_arr = collapse_filter(overfit_arr)
    try:
        assert min_arr, "No fits succeeded."+\
            "  Change fit range manually:"+str(min_arr)
    except AssertionError:
        min_arr = overfit_arr
        try:
            assert overfit_arr, "No fits succeeded."+\
                "  Change fit range manually:"+str(min_arr)
        except AssertionError:
            raise FitFail
    return min_arr


@PROFILE
def collapse_filter(arr):
    """collapse the results array and filter out duplicates"""
    # collapse the array structure introduced by mpi
    shape = np.asarray(arr).shape
    if shape:
        if len(shape) > 1:
            if shape[1] != 3:
                arr = [x for b in arr for x in b]

    # filter out duplicated work (find unique results)
    arr = getuniqueres(arr)
    return arr



def get_first_res(name, min_arr):
    """Get the first mean for name"""
    mean = min_arr[0][0].__dict__[name].val
    err = min_arr[0][0].__dict__[name].err
    return mean, err

def post_loop(meta, loop_store,
              retsingle_save, test_success):
    """After loop over fit ranges"""
    result_min = {}
    min_arr, overfit_arr = loop_store
    min_arr = loop_result(min_arr, overfit_arr)
    # did anything succeed?
    # test = False if not list(min_arr) and not meta.random_fit else True
    test = list(min_arr) or meta.random_fit
    if len(min_arr) > 1:

        result_min_avg = find_mean_and_err(meta, min_arr)
        param_err_avg = result_min['energy'].err

    elif len(min_arr) == 1:
        result_min_avg = min_arr[0]
        param_err_avg = result_min[1]
        dump_single_fit(meta, min_arr)

    print_res.print_fit_results(meta, min_arr)

    if DOWRITE and not TLOOP:
        makerep(meta, min_arr, result_min_avg, param_err_avg)

    return test

def makerep(meta, min_arr, result_min_avg, param_err_avg):
    """Plot representative fit"""
    plotdata = namedtuple('data', ['coords', 'cov', 'fitcoord'])

    # set the fit range
    if len(min_arr) > 1:
        latfit.config.FIT_EXCL = list(closest_fit_to_avg(
            result_min_avg['energy'].val, min_arr))
        # do the best fit again, with good stopping condition
        # latfit.config.FIT_EXCL = min_excl(min_arr)
    elif len(min_arr) == 1:
        latfit.config.FIT_EXCL = list(min_arr[0][2])
    print("fit excluded points (indices):",
          latfit.config.FIT_EXCL)
    print("fit window = ", meta.fitwindow)

    if len(min_arr) > 1:
        print("fitting for representative fit")
        latfit.config.MINTOL = True
        assert ext.iscomplete()
        try:
            retsingle = sfit.singlefit(meta, meta.input_f)
        except ACCEPT_ERRORS_FIN:
            print("reusing first successful fit"+\
                  " since representative fit failed (NoConvergence)")
            retsingle = retsingle_save
            #param_err = retsingle_save[1]
    else:
        print("reusing first successful fit result for representative fit")
        retsingle = retsingle_save
        #param_err = retsingle_save[1]
    result_min_close, param_err_close, \
        plotdata.coords, plotdata.cov = retsingle

    result_min, param_err = combine_results(
        result_min, result_min_close,
        meta, param_err_avg, param_err_close)

    # plot the result
    plotdata.fitcoord = meta.fit_coord()
    if check_include(result_min) and DOWRITE:
        mkplot.mkplot(plotdata, meta.input_f, result_min,
                      param_err, meta.fitwindow)



def fill_err_array(min_arr, name, weight_sum):
    """Fill the error array"""
    fill = []
    for i in min_arr:
        for j in min_arr:
            fill.append(jack_mean_err(
                divbychisq(getattr(i[0], name).arr,
                           getattr(i[0], 'pvalue').arr/weight_sum),
                divbychisq(getattr(j[0], name).arr,
                           getattr(j[0], 'pvalue').arr/weight_sum),
                nosqrt=True)[1])
    fill = em.acsum(fill, axis=0)
    return fill

def parametrize_entry(result_min, name):
    """Make into blank Param object"""
    if name not in result_min:
        result_min[name] = Param()
    return result_min

@PROFILE
def getuniqueres(min_arr):
    """Find unique fit ranges"""
    ret = []
    keys = set()
    for i in min_arr:
        key = str(i[2])
        if key not in keys:
            ret.append(i)
            keys.add(key)
    return ret

# obsolete, we should simply pick the model with the smallest errors
# and an adequate chi^2 (t^2)
@PROFILE
def min_excl(min_arr):
    """Find the minimum reduced chisq (t^2) from all the fits considered"""
    minres = sorted(min_arr, key=lambda row: row[0])[0]
    if UNCORR:
        print("min chisq/dof=", minres[0])
    else:
        print("min t^2/dof=", minres[0])
    print("best times to exclude:", minres[1])
    return minres[1]
