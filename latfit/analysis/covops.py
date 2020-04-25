"""Operations on covariance matrix"""
import sys
import copy
from collections import namedtuple
from numpy import swapaxes as swap
from accupy import kdot
import numpy as np
from numpy import ma
from numpy.linalg import inv, tensorinv

from latfit.utilities import exactmean as em
from latfit.utilities.actensordot import actensordot
from latfit.analysis.errorcodes import PrecisionLossError
from latfit.mathfun.block_ensemble import delblock
from latfit.mathfun.block_ensemble import bootstrap_ensemble
from latfit.extract.inverse_jk import inverse_jk
from latfit.config import JACKKNIFE_FIT, RANDOMIZE_ENERGIES
from latfit.config import UNCORR_OP, CORRMATRIX, JACKKNIFE_BLOCK_SIZE
from latfit.config import UNCORR, GEVP
import latfit.config

CONST_SHIFT = 0

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

def dropimag(arr):
    """Drop the imaginary part if it's all 0"""
    if np.all(np.imag(arr) == 0.0):
        ret = np.real(arr)
    else:
        ret = arr
    return ret


if JACKKNIFE_FIT == 'SINGLE':
    def getcovfactor(_, reuse_blocked, config_num, reuse_inv_red):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        block == reuse
        SINGLE elimination jackknife
        SINGLE case is just the usual way to calculate the covariance matrix
        (no jackknifing after the first jackknife)
        """
        return reuse_inv_red-reuse_blocked[config_num]

elif JACKKNIFE_FIT == 'DOUBLE':
    @PROFILE
    def getcovfactor(params, reuse_blocked, config_num,
                     reuse_inv_red, bsize=JACKKNIFE_BLOCK_SIZE):
        """Get the factor which will be squared
        when computing jackknife covariance matrix
        (for this config number == config)
        inverse block == reuse_inv
        inverse block with the ith config
        or block of configs deleted == reuse_inv_red.
        block == reuse
        DOUBLE elimination jackknife
        """
        lcon = len(reuse_blocked) if RANDOMIZE_ENERGIES else 1
        if latfit.config.BOOTSTRAP or RANDOMIZE_ENERGIES:
            ret = (reuse_blocked-em.acmean(
                reuse_blocked, axis=0))/np.sqrt(lcon)
        else:
            num_configs_reduced = (params.num_configs-1)*bsize
            assert len(reuse_inv_red) == num_configs_reduced
            try:
                assert np.allclose(em.acmean(reuse_inv_red, axis=0),
                                   reuse_blocked[config_num], rtol=1e-14)
            except AssertionError:
                print(em.acmean(reuse_inv_red, axis=0))
                print(reuse_blocked[config_num])
                raise
            ret = np.array([
                np.mean(
                    np.delete(
                        reuse_inv_red, i, axis=0), axis=0)
                for i in range(num_configs_reduced)]) -\
                    reuse_blocked[config_num]
            assert len(ret) == num_configs_reduced
        return ret

def invp(matrix, invf):
    """Check if matrix is inverted"""
    assert np.allclose(np.dot(matrix, invf), np.eye(len(invf)), rtol=1e-14)

@PROFILE
def invertmasked(params, len_time, excl, covjack):
    """invert the covariance matrix with pruned operator basis
    and fit range"""
    dim = int(np.sqrt(np.prod(list(covjack.shape))))
    matrix = np.zeros((dim, dim))
    if params.dimops == 1 and not GEVP:
        matrix = np.copy(covjack)
    else:
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                if opa != opb and UNCORR_OP:
                    matrix[opa*len_time:(opa+1)*len_time,
                           opb*len_time:(opb+1)*len_time] = 0
                else:
                    matrix[opa*len_time:(opa+1)*len_time,
                           opb*len_time:(opb+1)*len_time] = covjack[
                               opa, opb, :, :]
    mask = np.zeros(matrix.shape)
    mask[excl, :] = 1
    mask[:, excl] = 1
    marray = ma.masked_array(matrix, dtype=float,
                             fill_value=0, copy=True, mask=mask)
    matrix = np.delete(matrix, excl, axis=0)
    matrix = np.delete(matrix, excl, axis=1)

    # hack
    #if invertmasked.params2 is None:
    invertmasked.params2.dimops = 1
    invertmasked.params2.num_configs = params.num_configs

    try:
        matrix_inv = invert_cov(matrix, invertmasked.params2)
        invp(matrix_inv, matrix)
        matrix = matrix_inv
        symp(matrix)
        marray[np.logical_not(marray.mask)] = normalize_covinv(
            matrix, invertmasked.params2).reshape(-1)
        flag = 0
    except np.linalg.linalg.LinAlgError as err:
        if str(err) == 'Singular matrix':
            print("Covariance matrix is singular",
                  "in jackknife fit.")
            print("Attempting to continue",
                  "fit with every other time slice",
                  "eliminated.")
            print("Plotted error bars should be " +
                  "considered suspect.")
            raise np.linalg.linalg.LinAlgError
            # flag = 1 # old way of handling error
        raise
    marray[marray.mask] = marray.fill_value
    return marray, flag
invertmasked.params2 = namedtuple('temp', ['dimops', 'num_configs'])



@PROFILE
def prune_covjack(params, covjack, coords_jack, flag):
    """Prune the covariance matrix based on aonfig excluded time slices"""
    excl = []
    len_time = len(params.time_range)
    # convert x-coordinates to index basis
    for i, xcoord in enumerate(coords_jack[:, 0]):
        for opa, _ in enumerate(latfit.config.FIT_EXCL):
            for j in range(len(latfit.config.FIT_EXCL[opa])):
                if xcoord == latfit.config.FIT_EXCL[opa][j]:
                    excl.append(opa*len_time+i)
    # rotate tensor basis to dimops, dimops, time, time
    # (or time, time if not GEVP)
    covjack = swap(covjack, len(covjack.shape)-1, 0)
    # invert the pruned covariance matrix
    marray, flag = invertmasked(params, len_time, excl, covjack)
    symp(marray)
    covinv_jack = np.zeros(covjack.shape, dtype=float)
    if params.dimops == 1 and not GEVP:
        covinv_jack = np.copy(marray.data)
    else:
        # fill in the pruned dimensions with 0's
        for opa in range(params.dimops):
            for opb in range(params.dimops):
                data = marray.data[
                    opa*len_time:(opa+1)*len_time,
                    opb*len_time:(opb+1)*len_time]
                assert np.asarray(data).shape == np.asarray(
                    covinv_jack[opa, opb]).shape
                covinv_jack[opa, opb, :, :] = data
        # to put things back in time, time, dimops, dimops basis
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-1, 0)
        covinv_jack = swap(covinv_jack, len(covinv_jack.shape)-2, 1)
    return covinv_jack, flag

def symp(matrix):
    """Assert matrix is symmetric"""
    try:
        assert np.allclose(matrix, matrix.T, rtol=1e-8)
    except AssertionError:
        for i, _ in enumerate(matrix):
            for j, _ in enumerate(matrix):
                if i <= j:
                    continue
                eva = matrix[i][j]
                evb = matrix[j][i]
                try:
                    assert np.allclose(eva, evb, rtol=1e-8)
                except AssertionError:
                    err = str(eva)+" "+str(evb)
                    print(err)
                    raise PrecisionLossError
        assert None, "bug; precision loss error should've been raised"


if CORRMATRIX:
    @PROFILE
    def invert_cov(covjack, params2):
        """Invert the covariance matrix via correlation matrix
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params2.dimops == 1:  # i.e. if not using the GEVP
            if UNCORR:
                covjack = np.diagflat(np.diag(covjack))
            covjack = dropimag(covjack)
            corrjack = np.zeros(covjack.shape)
            weightings = dropimag(np.sqrt(np.diag(covjack)))
            reweight = np.diagflat(1./weightings)
            corrjack = kdot(reweight, kdot(covjack, reweight))
            svd_check(corrjack)
            covinv_jack = kdot(kdot(reweight, inv(corrjack)), reweight)
        else:
            lent = len(covjack)  # time extent
            reweight = np.zeros((lent, params2.dimops,
                                 lent, params2.dimops))
            for i in range(lent):
                for j in range(params2.dimops):
                    reweight[i][j][i][j] = 1.0/np.sqrt(covjack[i][j][i][j])
            corrjack = actensordot(
                actensordot(reweight, covjack), reweight)
            if UNCORR:
                diagcorr = np.zeros(corrjack.shape)
                for i in range(lent):
                    for j in range(params2.dimops):
                        diagcorr[i][j][i][j] = corrjack[i][j][i][j]
                corrjack = diagcorr
            covinv_jack = swap(actensordot(reweight, actensordot(
                tensorinv(corrjack), reweight)), 1, 2)
        return covinv_jack

else:
    @PROFILE
    def invert_cov(covjack, params2):
        """Invert the covariance matrix,
        assumes shape is time, time or time, dimops, time dimops;
        returns time, time or time, time, dimops, dimops
        """
        if params2.dimops == 1:  # i.e. if not using the GEVP
            covinv_jack = inv(covjack)
        else:
            covinv_jack = swap(tensorinv(covjack), 1, 2)
        return covinv_jack

if JACKKNIFE_FIT == 'SINGLE':
    def normalize_covinv(covinv_jack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the covinv (single jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = ((nsize)*(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covinv_jack * norm

    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (single jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = 1/ ((nsize)*(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covjack * norm

elif JACKKNIFE_FIT == 'DOUBLE':
    @PROFILE
    def normalize_covinv(covinv_jack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the covinv (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = ((nsize)/(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covinv_jack * norm

    @PROFILE
    def normalize_cov(covjack, params, bsize=JACKKNIFE_BLOCK_SIZE):
        """do the proper normalization of the
        covariance matrix (double jk)"""
        total_configs = params.num_configs*bsize
        num_configs_minus_block = nconfigs_minus_block(total_configs, bsize)
        nsize = num_configs_minus_block
        norm = 1/((nsize)/(nsize-1))
        norm = norm if latfit.config.BOOTSTRAP else 1/norm
        return covjack * norm


@PROFILE
def get_covjack(cov_factor, params):
    """Get jackknife estimate of covariance matrix for this fit,
    note: no normalization is done here (it will need to be normalized
    depending on the jackknifing scheme)
    Also, the GEVP covjack has indexing that goes
    time, gevp index, time, gevp index
    """
    if params.dimops == 1 and not GEVP:  # i.e. if not using the GEVP
        covjack = np.einsum('ai, aj->ij', cov_factor, cov_factor)
    else:
        covjack = np.einsum('aim, ajn->imjn', cov_factor, cov_factor)
    return covjack

def nconfigs_minus_block(total_configs, bsize):
    """Number of configs less block size
    used for normalizing cov
    block reduction is eliminated if bootstrapping
    """
    if latfit.config.BOOTSTRAP:
        ret = total_configs
    else:
        ret = total_configs-bsize
    return ret

def svd_check(corrjack):
    """Check the correlation matrix eigenvalues
    cut on condition numbers > 10^10"""
    evals = np.linalg.eigvals(corrjack)
    evals = sorted(list(np.abs(evals)))
    assert len(evals) > 1, str(evals)
    cond = evals[-1]/evals[0]
    assert cond >= 1, str(evals)+" "+str(cond)
    if cond > 1e10:
        print("condition number of correlation matrix is > 1e10:", cond)
        raise PrecisionLossError


@PROFILE
def prune_fit_range(covinv_jack, coords_jack, debug=False):
    """Zero out parts of the inverse covariance matrix to exclude items
    from fit range.
    Thus, the contribution to chi^2 (or, usually, t^2) will be 0.
    """
    excl = list(latfit.config.FIT_EXCL)
    dimops1 = len(excl) == 1
    for i, xcoord in enumerate(coords_jack[:, 0]):
        if not debug:
            break
        for opa, _ in enumerate(excl):
            for j, _ in enumerate(excl[opa]):
                if xcoord == excl[opa][j]:
                    if not dimops1:
                        assert covinv_jack[
                            i, :, opa, :].all() == 0, "Prune failed."
                        assert covinv_jack[
                            :, i, opa, :].all() == 0, "Prune failed."
                        assert covinv_jack[
                            :, i, :, opa].all() == 0, "Prune failed."
                        assert covinv_jack[
                            i, :, :, opa].all() == 0, "Prune failed."
                    else:
                        assert covinv_jack[i, :].all() == 0, "Prune failed."
                        assert covinv_jack[:, i].all() == 0, "Prune failed."
                        assert covinv_jack[:, i].all() == 0, "Prune failed."
                        assert covinv_jack[i, :].all() == 0, "Prune failed."
    return covinv_jack


def update_coords(coords, reuse):
    """Update coords (for debugging only)"""
    mean = em.acmean(reuse)
    assert len(coords) == len(mean)
    for i, _ in enumerate(coords):
        coords[i][1] = mean[i]
    return coords

@PROFILE
def get_doublejk_data(params, coords_jack, reuse,
                      reuse_blocked, config_num):
    """Primarily, get the inverse covariance matrix for the particular
    double jackknife fit we are on (=config_num)
    reuse_inv is the original unjackknifed data
    coords_jack are the coordinates, we also return truncated coordinates
    if the fit fails.
    """

    reuse_new = np.array(copy.deepcopy(np.array(reuse)))
    reuse_blocked_new = np.array(copy.deepcopy(np.array(reuse_blocked)))
    reuse_new = apply_shift(reuse_new, coords_jack)
    reuse_blocked_new = apply_shift(reuse_blocked_new, coords_jack)

    # coords_jack_new = update_coords(coords_jack, reuse_new)

    # original data, obtained by reversing single jackknife procedure
    # we set the noise level to mimic the jackknifed data
    # so the random data is in some sense already "jackknifed"
    if RANDOMIZE_ENERGIES:
        reuse_inv = np.array(copy.deepcopy(np.array(reuse_new)))
    else:
        reuse_inv = inverse_jk(reuse_new, params.num_configs)

    # bootstrap ensemble
    reuse_blocked_new, coords_jack_new = bootstrap_ensemble(
        reuse_inv, coords_jack, reuse_blocked_new)

    # delete a block of configs
    reuse_inv_red = delblock(config_num, reuse_inv)

    flag = 2
    while flag > 0:
        cov_factor = getcovfactor(params, reuse_blocked_new,
                                  config_num, reuse_inv_red)
        covjack = get_covjack(cov_factor, params)
        covinv_jack_pruned, flag = prune_covjack(params, covjack,
                                                 coords_jack_new, flag)
    covinv_jack = prune_fit_range(covinv_jack_pruned, coords_jack_new)
    return coords_jack_new, covinv_jack, jack_errorbars(covjack, params)

def apply_shift(coords_jack_reuse, coords_jack):
    """apply the bootstrap constant shift to the averages"""
    coords_jack_reuse = copy.deepcopy(np.array(coords_jack_reuse))
    shift = 0 if not CONST_SHIFT else CONST_SHIFT
    shift_arr = convert_coord_dict(shift, coords_jack)
    sh1 = np.asarray(shift_arr).shape
    sh2 = coords_jack_reuse.shape
    check = copy.deepcopy(np.array(coords_jack_reuse))
    if not GEVP or np.all(sh2[1:] == sh1) or not np.array(shift).shape:
        shift = collapse_shift(shift_arr)
        try:
            coords_jack_reuse = coords_jack_reuse + shift
            assert np.allclose(check + shift, coords_jack_reuse,
                               rtol=1e-14)
        except TypeError:
            print(coords_jack_reuse)
            print(shift)
            print(sh1, sh2)
            raise
    else:
        for i, _ in enumerate(coords_jack_reuse):
            time = int(coords_jack_reuse[i][0])
            shiftc = shift if not shift else shift[time]
            coords_jack_reuse[i][1] += shiftc
            assert np.allclose(
                check[i][1] + shiftc,
                coords_jack_reuse[i][1], rtol=1e-14)
    return coords_jack_reuse

@PROFILE
def jack_errorbars(covjack, params):
    """Get error bars for this fit,
    given the covariance matrix
    """
    covjack = normalize_cov(covjack, params)
    lent = len(covjack)
    if len(covjack.shape) == 4:
        error_bars = np.zeros((covjack[0][0].shape), dtype=np.float)
        for i in range(lent):
            error_bars[i] = np.sqrt(np.diag(covjack[i, :, i, :]))
    elif len(covjack.shape) == 2:
        error_bars = np.sqrt(np.diag(covjack))
    else:
        print("***ERROR***")
        print("badly formed covariance matrix " +
              "provided to jackknife errorbar finder")
        print("shape =", covjack.shape)
        sys.exit(1)
    return error_bars

def convert_coord_dict(shift, coords):
    """Convert coord dict to array"""
    ret = shift
    allowedtimes = set()
    for i in coords:
        allowedtimes.add(i[0])
    if shift:
        ret = []
        keys = []
        for i in shift:
            if i in allowedtimes:
                keys.append(i)
        keys = sorted(keys)
        nextkey = keys[0]
        for i in keys:
            assert i == nextkey
            nextkey += 1
            ret.append(shift[i])
        ret = np.array(ret)
    return ret

if not GEVP:
    def collapse_shift(shift):
        """Collapse the shift structure for non GEVP fits"""
        if hasattr(shift, '__iter__'):
            ret = []
            for i in shift:
                ret.append(i[0])
            ret = np.array(ret)
        else:
            ret = shift
        return ret
else:
    def collapse_shift(shift):
        """Collapse the shift structure for non GEVP fits"""
        return shift
