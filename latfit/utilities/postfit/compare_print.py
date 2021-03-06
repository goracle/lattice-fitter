"""Compare fit range results; also print results to screen for inspection"""
import numpy as np
import gvar
import latfit.utilities.exactmean as em
from latfit.analysis.errorcodes import FitRangeInconsistency
from latfit.analysis.superjack import jack_mean_err
from latfit.config import ISOSPIN, STRONG_CUTS
from latfit.utilities.postfit.fitwin import win_nan, wins_contained
from latfit.utilities.postfit.cuts import lencut, allow_cut
from latfit.utilities.postfit.cuts import statlvl
from latfit.utilities.postfit.strproc import errstr

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def diff_ind(res, arr, fit_range_arr):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxdiff = 0
    maxsyserr = 0
    maxrange = []
    maxerr = 0
    errstr1 = ''
    res, emean = res
    for i, gres in enumerate(arr):

        gres, gemean = gres
        # gsdev = gres[0].sdev
        gfit_range = fit_range_arr[i]

        # apply cuts
        #if allow_cut(gvar.gvar(gemean, gsdev),
        #             dim, cutstat=False, chk_consis=False):
        #continue
        # cuts are passed, calculate the discrepancy

        diff, err, syserr = discrep(res, gres, mean_diff=emean-gemean)

        maxsyserr = max(syserr, maxsyserr)
        #maxdiff = maxarr(diff, maxdiff)
        #if np.all(diff == maxdiff):
        #    maxerr = err

        #if maxsig == sig and maxsig:
        if syserr == maxsyserr:
            maxdiff = diff
            maxerr = np.sqrt(syserr**2+err**2)
            maxrange = gfit_range
            #mean = avg_gvar(gres)
            sdev = gres[0].sdev
            if len(fit_range_arr[i]) > 1:
                errstr1 = "disagree:"+str(i)+" "+str(
                    gvar.gvar(0, sdev))+" "+str(fit_range_arr[i])
            else:
                errstr1 = float(fit_range_arr[i][0])
    ret = gvar.gvar(maxdiff, maxerr)
    sig = statlvl(ret)
    return ret, sig, errstr1, maxsyserr, maxrange

@PROFILE
def maxarr(arr1, arr2):
    """Get the max between two arrays element by element"""
    if arr1 is None:
        ret = arr2
    elif arr2 is None:
        ret = arr1
    else:
        ret = []
        for i, j in zip(arr1, arr2):
            ret.append(max(i, j))
        ret = np.array(ret)
    return ret

@PROFILE
def discrep(res, gres, mean_diff=None):
    """Calculate the stat. sig of the disagreement"""
    assert len(res) == len(gres)
    diff = np.fromiter((i.val-j.val for i, j in zip(res, gres)),
                       count=len(res), dtype=np.float)
    # needs super jack
    mean, err = jack_mean_err(diff, acc_sum=False, mean_arr=mean_diff)
    mean = np.abs(mean)
    sys_err = np.sqrt(max((mean/1.5)**2-err**2, 0))
    #sys_err = max(0, mean-1.5*err)
    #sig = statlvl(gvar.gvar(em.acmean(diff), err))
    #maxsig = max(sig, maxsigcurr)
    return mean, err, sys_err
discrep.cache = {}

def clear_diff_cache():
    """Clear the diff cache"""
    print("clearing the pair difference cache")
    discrep.cache = {}

@PROFILE
def errfake(frdim, errstr1):
    """Is the disagreement with an effective mass point outside of this
    dimension's fit window?  Then regard this error as spurious"""
    tmin = min(frdim)
    tmax = max(frdim)
    ret = errstr1 < tmin or errstr1 > tmax
    if not ret:
        print(tmin, tmax, errstr1, frdim)
    return ret

def fit_wins_equality(wins1, wins2):
    """Compare two lists of fit windows for equality"""
    win = None
    for i, j in zip(wins1, wins2):
        i = list(i)
        j = list(j)
        inew = list(np.nan_to_num(i))
        jnew = list(np.nan_to_num(j))
        if i != inew or j != jnew:
            continue
        if win is None:
            win = i
        try:
            assert j[0] == win[0], (j, win)
            assert j[1] == win[1], (j, win)
            assert len(i) == len(j)
            assert len(i) == 2
        except (AssertionError, ValueError):
            print(i, j, win, wins1, wins2)
            raise

def slice_min_for_windows(mins):
    """Slice for minimum lists for the fit windows"""
    ret = []
    for i in mins:
        ret.append(i[2][1])
    return ret

@PROFILE
def print_compiled_res(cbest, min_en, min_ph):
    """Print the compiled results"""

    # total error results to plot
    min_enf = [(str(i), j, k, l) for i, j, k, l in min_en]
    min_phf = [(str(i), j, k, l) for i, j, k, l in min_ph]
    ret = list(zip(min_enf, min_phf))

    # perform check
    fitwins = slice_min_for_windows(min_en)
    fitwins2 = slice_min_for_windows(min_ph)
    if not cbest and STRONG_CUTS:
        if STRONG_CUTS:
            fit_wins_equality(fitwins, fitwins2)
        else:
            assert wins_contained(fitwins, fitwins2), (fitwins, fitwins2)
        if not win_nan(fitwins[0]) and not win_nan(fitwins2[0]):
            assert list(fitwins[0]) == list(fitwins2[0]), (
                fitwins[0], fitwins2[0])

    # res(stat)(sys) error string to print
    min_en = [errstr(i, j) for i, j, _, _ in min_en]
    min_ph = [errstr(i, j) for i, j, _, _ in min_ph]
    min_res = [
        list(i) for i in zip(min_en, min_ph) if list(i)]
    min_res_pr = [
        i for i in min_res if 'nan' not in str(i[0])]
    test = False
    if min_res_pr:
        print("minimized error results:", min_res_pr)
        test = True

    return ret, test, (fitwins[0], min_res_pr)

@PROFILE
def printres(effmass1, pval, syserr, fit_range, maxrange):
    """Print the result (and a header for the first result printed)"""
    #effmass1 = avg_gvar(effmass)
    #effmass1 = gvar.gvar(effmass1, effmass[0].sdev)
    if not printres.prt:
        print("val(err); syserr; pvalue; ind diff; median difference;",
              " avg difference; fit range; disagreeing fit range")
        printres.prt = True
    syst = trunc(syserr)
    print(effmass1, syst, pval, fit_range, maxrange)
printres.prt = False

@PROFILE
def update_effmass(effmass, errterm):
    """Replace the stat error with the
    total error in the effmass array"""
    ret = [gvar.gvar(i.val, errterm) for i in effmass]
    return ret


@PROFILE
def print_sep_errors(tot_pr):
    """Print results with separate errors"""
    for i in tot_pr:
        i = list(i)
        if i:
            fitwin, i = i
            print("fitwin", fitwin)
            print(i)

@PROFILE
def swap_err(gvar1, newerr):
    """Swap the errors in the gvar object"""
    err1 = gvar1.sdev
    #val = gvar1.val
    ret = gvar.gvar(gvar1, newerr)
    return ret, err1

@PROFILE
def trunc(val):
    """Truncate the precision of a number
    using gvar"""
    if isinstance(val, int) or 'e' in str(val):
        ret = val
    else:
        ret = float(str(gvar.gvar(val))[:-3])
    return ret

@PROFILE
def sort_check(median_err, reverse=False):
    """Make sure array is sorted by stat error size"""
    emax = 0
    for _, (effmass, _, _) in enumerate(median_err):
        sdev = effmass[0].sdev
        emax = max(sdev, emax)
        if reverse:
            assert sdev <= emax, (sdev, emax)
        else:
            assert sdev == emax, (sdev, emax)

@PROFILE
def output_loop(median_store, avg_dim, dim_idx, fit_range_arr, best):
    """The print loop
    """
    # dim, allowidx = dim_idx
    dim, _ = dim_idx
    print("dim:", dim)
    median_err, median = median_store
    maxsig = 0
    # usegevp = gevpp(freqarr)

    themin = None
    #fitrmin = None
    # pvalmin = None

    if not list(median_err):
        print("no results after cuts")
    else:
        print("number of results:", len(median_err))

    sort_check(median_err, reverse=False)

    effmass = []

    for idx, (effmass, pval, emean) in enumerate(median_err):

        sdev = effmass[0].sdev
        try:
            assert sdev, effmass
        except AssertionError:
            continue
        # skip if the error is already too big
        if themin is not None:
            if sdev >= themin[0].sdev:
                #print("proceeding stat errors are too large; breaking median loop")
                break

        fit_range = fit_range_arr[idx]
        assert not lencut(fit_range, dim) or (
            not ISOSPIN and not STRONG_CUTS), (fit_range, idx)

        # best known cut (stat comparison only)
        if allow_cut(gvar.gvar(emean, sdev),
                     dim, best, chk_consis=False):
            continue

        # mostly obsolete params
        pval = trunc(pval)
        assert list(effmass.shape) == list(median.shape), (
            list(effmass.shape), list(median.shape))
        #median_diff = effmass-median
        #median_diff = np.array(
        #    [gvar.gvar(i.val, max(sdev, i.sdev)) for i in median])
        avg_diff = effmass-avg_dim
        avg_diff = np.array(
            [gvar.gvar(i.val, max(
                sdev, avg_dim.sdev)) for i in avg_diff])

        # compare this result to all other results
        if list(np.array(median_err)[:, 0]):
            ind_diff, sig, errstr1, syserr, maxrange =\
                diff_ind((effmass, emean), np.array(
                    median_err)[:, 0::2], fit_range_arr)
        else:
            ind_diff, sig, errstr1, syserr, maxrange = (
                gvar.gvar(np.nan, np.nan),
                np.nan, '', np.nan, fit_range)
        assert ind_diff.sdev >= syserr

        errterm = np.sqrt(sdev**2+syserr**2)
        #effmass = update_effmass(effmass, errterm)

        noprint = False
        if themin is not None:
            if themin[0].sdev > errterm:
                themin = (gvar.gvar(emean, errterm), syserr, fit_range, effmass)
                #pvalmin = pval
                #fitrmin = fit_range
            else:
                noprint = True
        else:
            #pvalmin = pval
            #fitrmin = fit_range
            themin = (gvar.gvar(emean, errterm), syserr, fit_range, effmass)

        # if the max difference is not zero
        if ind_diff.val:

            #fake_err = False
            #if isinstance(errstr, float):
            #    fake_err = errfake(fit_range[dim], errstr)


            # print the result
            if not noprint:
                printres(themin[0], pval, syserr, fit_range, maxrange)

            # keep track of largest errors;
            # print the running max
            if maxsig < sig:
                maxsig = max(maxsig, sig)
                print("")
                print(ind_diff, '(', trunc(sig), 'sigma )')
                print("")

            # fit range inconsistency found;
            # error handling below
            # I=2 fits to a constant,
            # so we must look at all disagreements
            try:
                #assert sig < 1.5 or fake_err or not (
                #    disagree_multi_multi_point or ISOSPIN == 2)
                # more of a sanity check at this point, than
                # an actual cut, given that we preserve systematic error
                # information
                assert sig < 2.0
            except AssertionError:

                if isinstance(errstr1, float):
                    if errstr1:
                        print("disagreement (mass) with data point at t=",
                              errstr1, 'dim:', dim, "sig =", sig)
                        print("")
                else:
                    print("disagreement at", sig, "sigma")
                    print(errstr1, 'dim:', dim)
                    print("")

                raise FitRangeInconsistency
        else:
            # no differences found; print the result
            # skip effective mass points for I=0 fits (const+exp)
            if ISOSPIN == 2 or len(fit_range) != 1.0:
                if not noprint:
                    printres(themin[0], pval, syserr, fit_range, maxrange)
    if themin is not None:
        #print('p-value weighted median =', gvar.gvar(avg_gvar(median),
        #                                              median[0].sdev))
        # print("p-value weighted mean =", avg_dim)
        ret = (
            themin[0], themin[1], [themin[2]], dropsdev(themin[3]))
    else:
        ret = (gvar.gvar(np.nan, np.nan), np.nan,
               [[[]]], np.nan*dropsdev(effmass))
    print("min:", ret[0])
    return ret

def dropsdev(arr):
    """From array of gvar items, drop sdev info"""
    ret = []
    if list(arr):
        ret = [i.val for i in arr]
    return np.array(ret)

@PROFILE
def avg_gvar(arr):
    """Average array of gvar objects"""
    ret = em.acmean([i.val for i in arr])
    ret = np.asarray(ret)
    return ret
