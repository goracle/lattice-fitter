import numbers
import numpy as np
import latfit.utilities.exactmean as em
from latfit.config import SUPERJACK_CUTOFF

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile


@PROFILE
def jack_mean_err(arr, arr2=None, sjcut=SUPERJACK_CUTOFF, nosqrt=False, acc_sum=True, mean_arr=None):
    """Calculate error in arr over axis=0 via jackknife factor
    first n configs up to and including sjcut are exact
    the rest are sloppy.
    """
    len_total = len(arr)
    len_sloppy = len_total-sjcut

    # selection of algo. prec.
    sumf = em.acsum if acc_sum else np.sum
    meanf = em.acmean if acc_sum else np.mean

    mean1 = meanf(arr[:sjcut], axis=0)
    mean1a = meanf(arr[sjcut:], axis=0)
    if arr2 is None:
        mean2 = mean1
        mean2a = mean1a
    else:
        mean2 = meanf(arr2[:sjcut], axis=0)
        mean2a = meanf(arr2[sjcut:], axis=0)
    arr2 = arr if arr2 is None else arr2

    if sjcut == 0:
        assert not sjcut, "sjcut bug"
    if not sjcut:
        assert sjcut == 0, "sjcut bug"

    # get jackknife correction prefactors
    exact_prefactor = (sjcut-1)/sjcut if sjcut else 0
    exact_prefactor_inv = sjcut/(sjcut-1) if sjcut else 0
    assert not np.isnan(exact_prefactor), "exact prefactor is nan"
    sloppy_prefactor = (len_sloppy-1)/len_sloppy
    assert not np.isnan(sloppy_prefactor), "sloppy prefactor is nan"
    overall_prefactor = (len_total-1)/len_total
    assert not np.isnan(overall_prefactor), "sloppy prefactor is nan"
    if not sjcut:
        assert overall_prefactor == sloppy_prefactor, "bug"
    assert arr.shape == arr2.shape, "Shape mismatch"

    # calculate error on exact and sloppy
    if sjcut:
        errexact = exact_prefactor*sumf(
            (arr[:sjcut]-mean1)*(arr2[:sjcut]-mean2),
            axis=0)
    else:
        errexact = 0
        assert errexact == 0, "non-zero error in the non-existent"+\
            " exact samples"
    if isinstance(errexact, numbers.Number):
        assert not np.isnan(errexact), "exact err is nan"
    else:
        assert not any(np.isnan(errexact)), "exact err is nan"

    errsloppy = sloppy_prefactor*sumf(
        (arr[sjcut:]-mean1a)*(arr2[sjcut:]-mean2a),
        axis=0)
    if isinstance(errsloppy, numbers.Number):
        assert not np.isnan(errsloppy), "sloppy err is nan"
    else:
        assert not any(np.isnan(errsloppy)), "sloppy err is nan"

    # calculate the superjackknife errors
    # (redundant prefactor multiplies, but perhaps clearer)
    err = overall_prefactor*(errsloppy/sloppy_prefactor+\
                             errexact*exact_prefactor_inv)
    try:
        err = err if nosqrt else np.sqrt(err)
        flag = False
    except FloatingPointError:
        flag = True
    assert err.shape == np.array(arr)[0].shape,\
        "Shape is not preserved (bug)."

    # take advantage of pre-computed average
    if mean_arr is None:
        mean = meanf(arr, axis=0)
    else:
        mean = mean_arr

    return flagtonan(mean, err, flag)

@PROFILE
def flagtonan(mean, err, flag):
    """nan the mean and error if error flag is turned on"""
    # calculate the mean
    if isinstance(mean, numbers.Number):
        mean = float(mean)
        assert not np.isnan(mean), "mean is nan"
    else:
        assert not any(np.isnan(mean)), "mean is nan"
    if not flag:
        if isinstance(err, numbers.Number):
            err = float(err)
            assert not np.isnan(err), "err is nan"
        else:
            assert not any(np.isnan(err)), "err is nan"
    else:
        mean = np.nan
        err = np.nan
    return mean, err

