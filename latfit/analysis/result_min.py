"""Contains result min class used in jackknife fit loop; stores results
 of fit"""
from collections import namedtuple
import pickle
from random import randint
from mpi4py import MPI
import mpi4py
import numpy as np
from scipy import stats
from latfit.config import START_PARAMS, UNCORR, ALTERNATIVE_PARALLELIZATION
from latfit.config import GEVP, NBOOT, VERBOSE
from latfit.analysis.errorcodes import DOFNonPosFit
from latfit.analysis.filename_windows import filename_plus_config_info
import latfit.config
import latfit.analysis.hotelling as hotelling

MPIRANK = MPI.COMM_WORLD.rank
MPISIZE = MPI.COMM_WORLD.Get_size()
COMM = MPI.COMM_WORLD
mpi4py.rc.recv_mprobe = False
DOWRITE = ALTERNATIVE_PARALLELIZATION and not MPIRANK\
    or not ALTERNATIVE_PARALLELIZATION

VERBOSE = VERBOSE and DOWRITE
try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

# misnomer, is actually just the actual fit range
WINDOW = []

NULL_CHISQ_ARRS = {}

def bootstrap_copy(arr, nboot):
    """Bootstrap array"""
    ret = []
    for _ in range(nboot):
        ret.append(arr[randint(0, len(arr)-1)])
    ret = np.array(ret)
    return ret

def bootstrap_adjust_pvalue(pvalue_arr_boot, nboot=20000):
    """Get adjusted p-value which is corrected using another bootstrap"""
    ensem = bootstrap_copy(pvalue_arr_boot, nboot)
    ret = {}
    for pval in pvalue_arr_boot:
        count = len([i for i in ensem if i <= pval])
        ret[pval] = count/nboot
    return ret


def chisq_arr_to_pvalue_arr(dof, nconf, chisq_arr_boot, chisq_arr):
    """Get the array of p-values"""
    chisq_arr_boot = sorted(list(chisq_arr_boot))
    chisq_arr_boot = np.asarray(chisq_arr_boot)
    if len(chisq_arr) > 1:
        assert len(np.asarray(chisq_arr).shape) == 1, str(
            np.asarray(chisq_arr))
        print("variance of null dist:", np.std(chisq_arr_boot)**2)
        print("mean of null dist:", np.mean(chisq_arr_boot))
        print(hotelling.hstr(dof, nconf))
    assert len(chisq_arr_boot) == NBOOT, str(len(chisq_arr_boot))
    chisq_arr = np.asarray(chisq_arr)
    pvalue_arr_boot = []
    for i, _ in enumerate(chisq_arr_boot):
        pvalue_arr_boot.append((NBOOT-i-1)/NBOOT)
    pvalue_arr_boot = np.array(pvalue_arr_boot)

    # now we need to do another bootstrap since we need a uniform dist for p
    # dict of adjusted p-values (DOES NOT WORK, PROBABLY, DO NOT TURN ON)
    # pvalue_arr_boot_adj = bootstrap_adjust_pvalue(pvalue_arr_boot)

    pvalue_arr = []
    for chisq1 in chisq_arr:
        subarr = np.abs(chisq1-chisq_arr_boot)
        minidx = list(subarr).index(min(subarr))
        pvalue_arr.append(pvalue_arr_boot[minidx])
    pvalue_arr = np.array(pvalue_arr)
    return pvalue_arr

def trash(item):
    """Is item 0, np.nan, or None?"""
    if hasattr(item, '__iter__') and not isinstance(item, str):
        raise ValueError
    else:
        if isinstance(item, str):
            ret = not item
        else:
            ret = not item or str(item) == 'nan'
    return ret

def dead(arr):
    """Is the array alive?
    Multiple tests to see if it's filled with real data
    or just some placeholder"""
    if hasattr(arr, '__iter__'):
        try:
            arr = list(arr)
            ret = np.all([trash(i) for i in arr])
        except ValueError:
            ret = np.all([dead(i) for i in arr])
        except TypeError:
            arr = str(arr)
            ret = dead(arr)
    else:
        ret = trash(arr)
    assert ret == True or ret == False, (ret, arr)
    return ret


def blank(arr):
    """is the array just zeros?"""
    arr = np.asarray(arr)
    comp = np.zeros(arr.shape)
    return np.all(comp == arr)

class Param:
    """Storage for the average param, the array of the param,
    and the error on the param
    """
    def __init__(self):
        self.arr = np.array(None)
        self.err = None
        self.val = None
        self.__gathered = False

    def gather(self):
        if not self.__gathered and self.arr.shape:
            self.__callgather()
            self.__gathered = True

    def __callgather(self):
        """MPI gather array"""
        COMM.barrier()
        arr = self.arr
        gat = COMM.allgather(arr)
        ret = []
        shape = None

        # shape check
        for item in gat:
            item = np.array(item)
            if shape is None:
                shape = item.shape
            else:
                assert shape == item.shape, (
                    shape, item.shape)

        for cfig in range(len(gat[0])):
            app = False
            prev = None
            for rank in range(len(gat)):
                assert len(gat) == MPISIZE
                item = gat[rank][cfig]
                #print("it", item, rank, cfig)
                if blank(item): 
                    # we did no work for this rank,
                    # config combination
                    continue
                if not app: # append once
                    ret.append(item)
                    prev = item
                    app = True
                else:
                    # if we duplicated some work, then make sure
                    # every rank got the same result
                    assert np.all(prev == item), (prev, item)
            assert app, "we missed a config with our parallelization"
        ret = np.array(ret)
        # final shape check
        assert ret.shape == arr.shape, (ret, arr)
        COMM.barrier()
        self.arr = ret

    def swapidx(self, idx1, idx2):
        """Swap axes method"""
        if idx1 != idx2:
            if hasattr(self.val, '__iter__'):
                self.val[idx1], self.val[idx2] = self.val[idx2], self.val[idx1]
            if hasattr(self.arr, '__iter__'):
                self.arr[:, idx1], self.arr[:, idx2] = self.arr[
                    :, idx2], self.arr[:, idx1]
            if hasattr(self.err, '__iter__'):
                self.err[idx1], self.err[idx2] = self.err[idx2], self.err[idx1]
        return self

    def zero(self, num_configs=None):
        """Zero out the array, value, and error"""
        self.val = 0
        self.err = 0
        if num_configs is not None:
            self.arr = np.zeros(num_configs)

    def mean(self, axis=0):
        """Take average of array"""
        #return em.acmean(self.arr, axis=axis)
        return np.mean(self.arr, axis=axis)

class ResultMin:
    """Store fit results for an individual fit range in this class"""
    def __init__(self, meta, params, coords):

        # the Params
        self.energy = Param()
        self.systematics = Param()
        self.pvalue = Param()
        self.chisq = Param()
        self.phase_shift = Param()
        self.scattering_length = Param()
        self.min_params = Param()

        self.misc = namedtuple(
            'misc', ['error_bars', 'dof',
                     'status', 'num_configs'])
        self.misc.error_bars = None
        self.misc.dof = None
        self.misc.num_configs = None
        self.misc.status = 0
        self.alloc_phase_shift(params)
        self.alloc_sys_arr(params)
        meta.actual_range() # to set WINDOW
        self.compute_dof(params, coords)
        self.alloc_errbar_arr(params, len(coords))
        self.pvalue.zero(params.num_configs)
        self.min_params.arr = np.zeros((params.num_configs,
                                        len(START_PARAMS)))
        self.energy.arr = np.zeros((params.num_configs,
                                    len(START_PARAMS)
                                    if not GEVP else params.dimops))
        self.__paramlist = {'energy': self.energy,
                            'systematics': self.systematics,
                            'pvalue': self.pvalue,
                            'chisq': self.chisq,
                            'phase_shift': self.phase_shift,
                            'scattering_length': self.scattering_length,
                            'min_params': self.min_params}

    def gather(self):
        """MPI gather data from parallelized jackknife loop"""
        for item in self.__paramlist:
            if VERBOSE:
                pass
                #print("gathering:", item)
            self.__paramlist[item].gather()

    def printjack(self, meta):
        """Prints out the jackknife blocks"""
        for i in self.__paramlist:
            if not DOWRITE:
                break
            print("jackknife blocks for", i, ":")
            name = i+'_single'
            fname = filename_plus_config_info(meta, name)+'.jkdat'
            topr = self.__paramlist[i].arr
            if not dead(topr):
                print("writing jackknife blocks in:", fname)
                fn1 = open(fname, 'wb')
                pickle.dump(topr, fn1)


    @PROFILE
    def alloc_errbar_arr(self, params, time_length):
        """Allocate an array. Each config gives us a jackknife fit,
        and a set of error bars.
        We store the error bars in this array, indexed
        by config.
        """
        if params.dimops > 1 or GEVP:
            errbar_arr = np.zeros((params.num_configs, time_length,
                                   params.dimops),
                                  dtype=np.float)
        else:
            errbar_arr = np.zeros((params.num_configs, time_length),
                                  dtype=np.float)
        self.misc.error_bars = errbar_arr



    @PROFILE
    def alloc_sys_arr(self, params):
        """alloc array for systematics"""
        syslength = len(START_PARAMS)-params.dimops*(1 if GEVP else 0)
        syslength = max(1, syslength)
        self.systematics.arr = np.zeros((params.num_configs, syslength))
        self.misc.num_configs = params.num_configs
        # storage for fit by fit chi^2 (t^2)
        self.chisq.arr = np.zeros(params.num_configs)

    @PROFILE
    def funpvalue(self, chisq):
        """Give pvalue from Hotelling t^2 stastistic
        (often referred to incorrectly as chi^2;
        is actually a sort of correlated chi^2)
        """
        ret = None
        nar = NULL_CHISQ_ARRS
        if self.misc.dof is not None and\
           self.misc.dof not in NULL_CHISQ_ARRS:
            correction = (self.misc.num_configs-self.misc.dof)/(
                self.misc.num_configs-1)
            correction /= self.misc.dof
            correction = 1 if UNCORR else correction
            cor = correction
            ret = stats.f.sf(chisq*cor, self.misc.dof,
                             self.misc.num_configs-self.misc.dof)
        elif self.misc.dof in nar:
            ret = chisq_arr_to_pvalue_arr(self.misc.dof,
                                          self.misc.num_configs,
                                          nar[self.misc.dof],
                                          np.asarray([chisq]))[0]
        return ret

    @PROFILE
    def alloc_phase_shift(self, params):
        """Get an empty array for Nconfig phase shifts"""
        nphase = 1 if not GEVP else params.dimops
        if hasattr(self.phase_shift.arr, '__iter__'):
            assert not np.asarray(self.phase_shift.arr).shape
        if GEVP:
            self.phase_shift.arr = np.zeros((params.num_configs,
                                             nphase), dtype=np.complex)
        else:
            self.phase_shift.arr = np.zeros((params.num_configs),
                                            dtype=np.complex)



    @PROFILE
    def compute_dof(self, params, coords):
        """Correct the degrees of freedom based on the chosen fit range"""
        # compute degrees of freedom
        try:
            assert len(coords) == len(WINDOW)
        except AssertionError:
            if VERBOSE:
                print("dof from WINDOW", WINDOW)
                print("dof from coords", coords)
            raise
        self.misc.dof = len(coords)*params.dimops-len(START_PARAMS)
        for i in coords[:, 0]:
            for j in latfit.config.FIT_EXCL:
                if i in j and i in WINDOW:
                    self.misc.dof -= 1
        if self.misc.dof < 1:
            if VERBOSE:
                print("dof < 1. dof =", self.misc.dof)
                print("actual fit range:", WINDOW)
                print("excl:", latfit.config.FIT_EXCL)
            raise DOFNonPosFit(dof=self.misc.dof,
                               excl=list(latfit.config.FIT_EXCL))
