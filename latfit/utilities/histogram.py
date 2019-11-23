#!/usr/bin/python3
"""Make histograms from fit results over fit ranges"""
import sys
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gvar
from latfit.utilities import exactmean as em
from latfit.analysis.errorcodes import FitRangeInconsistency


ISOSPIN = None

def main():
    """Make the histograms.
    """
    if len(sys.argv[1:]) == 1 and (
            'phase_shift' in sys.argv[1] or\
            'energy' in sys.argv[1]):
        fname = sys.argv[1]
        newfn = None
        if 'phase_shift' in fname:
            energyfn = re.sub('phase_shift', 'energy', fname)
            phasefn = fname
        elif 'energy' in fname:
            phasefn = re.sub('energy', 'phase_shift', fname)
            energyfn = fname
        min_en1 = make_hist(energyfn, nosave=True)
        min_ph1 = make_hist(phasefn, nosave=True)
        min_res = print_compiled_res(min_en1, min_ph1)
        tot_pos = []
        min_ph = min_ph1
        min_en = min_en1
        tot_pos.append(min_res)
        tadd = 0
        while min_en and min_ph:
            tadd += 1
            min_en = make_hist(energyfn, nosave=True, tadd=tadd)
            min_ph = make_hist(phasefn, nosave=True, tadd=tadd)
            tot_pos.append(print_compiled_res(min_en, min_ph))
        tot_neg = []
        tadd = 0
        min_ph = min_ph1
        min_en = min_en1
        while min_en and min_ph:
            tadd -= 1
            min_en = make_hist(energyfn, nosave=True, tadd=tadd)
            min_ph = make_hist(phasefn, nosave=True, tadd=tadd)
            tot_neg.append(print_compiled_res(min_en, min_ph))
        print_tot_pos(tot_pos)
        print_tot_neg(tot_neg)
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, nosave=False)
        print("minimized error results:", min_res)

def print_tot_pos(tot):
    """Print results vs. tmin"""
    tadd = -1
    for i in tot:
        tadd += 1
        print("tmin = tmin_start +", tadd)
        print(i)

def print_tot_neg(tot):
    """Print results vs. tmax"""
    tadd = 0
    for i in tot:
        tadd -= 1
        print("tmax = tmax_start -", -tadd)
        print(i)

def print_compiled_res(min_en, min_ph):
    """Print the compiled results"""
    min_en = [str(i) for i in min_en]
    min_ph = [str(i) for i in min_ph]
    min_res = [list(i) for i in zip(min_en, min_ph)]
    print("minimized error results:", min_res)
    return min_res


def trunc(val):
    """Truncate the precision of a number
    using gvar"""
    if isinstance(val, int):
        ret = val
    else:
        ret = float(str(gvar.gvar(val))[:-3])
    return ret

#def fill_pvalue_arr(pdat_freqarr, desired_length):
    #"""Extend pdat_freqarr to desired length"""
    #ext = desired_length-len(pdat_freqarr)
    #if ext > 0:
    #    pdat_freqarr = list(pdat_freqarr)
    #    for _ in range(ext):
    #        pdat_freqarr.append(1)
    #    assert len(pdat_freqarr) == desired_length
    #return np.asarray(pdat_freqarr)

def fill_pvalue_arr(pdat_freqarr, exclarr):
    """the pvalue should be 1.0 if the fit range is length 1 (forced fit)
    prelim_loop = zip(freq, errlooparr, exclarr)
    """
    ret = list(pdat_freqarr)
    if len(pdat_freqarr) != len(exclarr):
        for i, _ in enumerate(exclarr):
            if len(exclarr[i]) == 1.0:
                ret.insert(i, 1.0)
        try:
            assert len(ret) == len(exclarr), str(ret)+" "+str(exclarr)
        except AssertionError:
            for i in exclarr:
                print(i)
            print(len(exclarr))
            print(len(ret))
            raise
    ret = np.array(ret)
    return ret

def pvalue_arr(spl, fname):
    """Get pvalue array (over fit ranges)"""
    pvalfn = re.sub(spl, 'pvalue', fname)
    pvalfn = re.sub('shift_', '', pvalfn) # todo: make more general

    # get file name for pvalue
    with open(pvalfn, 'rb') as fn1:
        pdat = pickle.load(fn1)
        pdat = np.array(pdat)
        pdat = np.real(pdat)
        try:
            # pdat_avg, pdat_err, pdat_freqarr, pdat_excl = pdat
            _, _, pdat_freqarr, _ = pdat
        except ValueError:
            print("not the right number of values to unpack.  expected 3")
            print("but shape is", pdat.shape)
            print("failing on file", pvalfn)
            sys.exit(1)
    return pdat_freqarr

def err_arr(fname, freqarr, avg):
    """Get the error array"""
    # get file name for error
    if 'I' not in fname:
        errfn = fname.replace('.p', "_err.p", 1)
    else:
        errfn = re.sub(r'_mom(\d+)_', '_mom\\1_err_', fname)
        errfn2 = re.sub(r'_mom(\d+)_', '_err_mom\\1_', fname)
    #print("file with stat errors:", errfn)
    try:
        with open(errfn, 'rb') as fn1:
            errdat = pickle.load(fn1)
            errdat = np.real(np.array(errdat))
    except FileNotFoundError:
        with open(errfn2, 'rb') as fn1:
            errdat = pickle.load(fn1)
            errdat = np.real(np.array(errdat))
    assert np.array(errdat).shape, "error array not found"

    print('shape:', freqarr.shape, avg)
    print('shape2:', errdat.shape)
    return errdat

def plot_title(fname, dim):
    """Get title for histograms"""
    # get title
    title = re.sub('_', " ", fname)
    title = re.sub(r'\.p', '', title)
    for i in range(10):
        if not i:
            continue
        if str(i) in title:
            title = re.sub('0', '', title)
            title = re.sub('mom', 'p', title)
            break
    title = re.sub('x', 'Energy (lattice units)', title)
    # plot title
    title_dim = title+' state:'+str(dim)
    plt.title(title_dim)

def slice_energy_and_err(freqarr, errdat, dim):
    """Slice the energy array and error array for a particular dimension"""
    freq = np.array([np.real(i) for i in freqarr[:, dim]])
    errlooparr = errdat[:, dim] if len(errdat.shape) > 1 else errdat
    return freq, errlooparr

def setup_medians_loop(freq, pdat_freqarr, errlooparr, exclarr):
    """Setup loop over results to get medians/printable results"""
    print("val(err); pvalue; ind diff; median difference;",
          " avg difference; fit range")
    pdat_median = np.median(pdat_freqarr)
    median_diff = np.inf
    median_diff2 = np.inf
    #print(freqarr[:, dim], errlooparr)
    prelim_loop = zip(freq, errlooparr, exclarr)
    loop = sorted(zip(freq, pdat_freqarr, errlooparr, exclarr),
                  key=lambda elem: elem[2], reverse=True)
    medians = (median_diff, median_diff2, pdat_median)

    return loop, medians

def setup_make_hist(fname):
    """Get some initial variables from file fname
    including the processed string
    the array of energies, the average energy
    and the fit ranges used
    """
    if 'I2' in fname:
        ISOSPIN = 2
    with open(fname, 'rb') as fn1:
        dat = pickle.load(fn1)
        try:
            avg, err, freqarr, exclarr = dat
        except ValueError:
            print("value error for file:", fname)
            raise
        freqarr = np.real(np.array(freqarr))
        exclarr = np.asarray(exclarr)
        avg = gvar.gvar(avg, err)
    spl = fname.split('_')[0]
    return spl, freqarr, avg, exclarr

def get_raw_arrays(fname):
    """Get the raw arrays from the fit output files"""
    spl, freqarr, avg, exclarr = setup_make_hist(fname)

    pdat_freqarr = pvalue_arr(spl, fname)

    pdat_freqarr = fill_pvalue_arr(pdat_freqarr, exclarr)

    errdat = err_arr(fname, freqarr, avg)
    return freqarr, exclarr, pdat_freqarr, errdat, avg

def get_medians_and_plot_syserr(loop, freqarr, freq, medians):
    """Get medians of various quantities (over fit ranges)
    loop:
    (freq, pdat_freqarr, errlooparr, exclarr)
    (energy, pvalue, error, fit range)
    """
    median_diff, median_diff2, pdat_median = medians
    median_err = []
    half = 0
    usegevp = gevpp(freqarr)
    for i, j, k, _ in loop: # drop the fit range from the loop
        # skip the single time slice points for GEVP
        if j == 1.0 and usegevp:
            pass
            #continue
        if abs(j - pdat_median) <= median_diff: # find median pvalue
            median_diff = abs(j-pdat_median)
            freq_median = i
        elif abs(j - pdat_median) <= median_diff2: # used to find median pval
            median_diff2 = abs(j-pdat_median)
            half = i
        median_err.append([gvar.gvar(np.real(i), np.real(k)), j])
        #print(median_err[-1], j)
    if median_diff != 0:
        freq_median = (freq_median+half)/2
    median = systematic_err_est(freq, median_err, freq_median)
    return (median_err, median), freq_median

def systematic_err_est(freq, median_err, freq_median):
    """Standard deviation of results over fit ranges
    is an estimate of the systematic error.
    Also, annotate the plot with this information.
    """
    # standard deviation
    try:
        sys_err = em.acstd(freq, ddof=1)
    except ZeroDivisionError:
        print("zero division error:")
        print(np.array(median_err))
        sys.exit(1)
    plt.annotate("standard dev (est of systematic)="+str(sys_err),
                 xy=(0.05, 0.7),
                 xycoords='axes fraction')
    median = gvar.gvar(freq_median, sys_err)
    return median

def plot_hist(freq, errlooparr):
    """Plot (plt) the histogram (but do not show)"""
    # print(freq, em.acmean(freq))
    hist, bins = np.histogram(freq, bins=10)
    hist = addoffset(hist)
    # print(hist)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center',
            width=0.7 * (bins[1] - bins[0]),
            xerr=getxerr(freq, center,
                         np.asarray(errlooparr, dtype=float)))

def make_hist(fname, nosave=False, tadd=0):
    """Make histograms
    """
    freqarr, exclarr, pdat_freqarr, errdat, avg = get_raw_arrays(fname)
    ret = []
    for dim in range(freqarr.shape[-1]):
        freq, errlooparr = slice_energy_and_err(freqarr, errdat, dim)
        save_str = re.sub(r'.p$', '_state'+str(dim)+'.pdf', fname)
        # setup the loop which obtains medians/printable results
        loop, medians = setup_medians_loop(freq, pdat_freqarr,
                                           errlooparr, exclarr)
        with PdfPages(save_str) as pdf:

            plt.ylabel('count')

            # plot the title
            plot_title(fname, dim)

            # plot the histogram
            plot_hist(freq, errlooparr)

            # loop to obtain medians/printable results
            # plot the systematic error
            median_store, freq_median = get_medians_and_plot_syserr(
                loop, freqarr, freq, medians)

            # plot median fit result (median energy)
            plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                         xycoords='axes fraction')

            # prints the sorted results
            try:
                output_loop.tadd = tadd
                themin = output_loop(median_store, freqarr, avg[dim], dim,
                                     build_sliced_fitrange_list(
                                         median_store, freq, exclarr, dim))
            except FitRangeInconsistency:
                continue

            if themin is not None:
                ret.append(themin)

            print("saving plot as filename:", save_str)

            pdf.savefig()

            if not nosave:
                plt.show()
    return ret

def build_sliced_fitrange_list(median_store, freq, exclarr, dim):
    """Get all the fit ranges for a particular dimension"""
    ret = []
    for _, i in enumerate(median_store[0]):
        effmass = i[0].val
        index = list(freq).index(effmass)
        fitrange = exclarr[index]
        # fitrange = np.array(fitrange)
        # dimfit = fit_range_dim(fitrange, dim)
        dimfit = fitrange
        ret.append(dimfit)
    return ret

def fit_range_dim(lexcl, dim):
    """Get the fit range for a particular dimension"""
    ret = np.asarray(lexcl)
    if len(ret.shape) > 1 or isinstance(ret[0], list):
        ret = ret[dim]
    else:
        print(ret)
    return ret

def gevpp(freqarr):
    """Are we using the GEVP?"""
    ret = False
    if hasattr(freqarr.shape, '__iter__'):
        ret = freqarr.shape[-1] > 1
    return ret

def global_tmin(fit_range_arr, dim):
    """Find the global tmin for this dimension:
    the minimum t for a successful fit"""
    tmin = np.inf
    for i in fit_range_arr:
        try:
            dimi = i[dim]
        except IndexError:
            if isinstance(i[0], float):
                dimi = i[0]
            else:
                print('i', i)
                raise
        if not isinstance(dimi, float):
            try:
                nmin = min(dimi)
                nmin = int(nmin)
            except TypeError:
                print('i',i , dimi)
                raise
        else:
            nmin = dimi
        tmin = min(tmin, nmin)
    return tmin

def global_tmax(fit_range_arr, dim):
    """Find the global tmax for this dimension:
    the maximum t for a successful fit"""
    tmax = 0
    for i in fit_range_arr:
        try:
            dimi = i[dim]
        except IndexError:
            if isinstance(i[0], float):
                dimi = i[0]
            else:
                print('i', i)
                raise
        if not isinstance(dimi, float):
            try:
                nmax = max(dimi)
                nmax = int(nmax)
            except TypeError:
                print('i',i , dimi)
                raise
        else:
            nmax = dimi
        tmax = max(tmax, nmax)
    return tmax


def output_loop(median_store, freqarr, avg_dim, dim, fit_range_arr):
    """The print loop
    """
    median_err, median = median_store
    maxdiff = 0
    usegevp = gevpp(freqarr)
    used = set()

    tmin_allowed = 0
    tmax_allowed = np.inf
    if output_loop.tadd > 0:
        tmin_allowed = global_tmin(fit_range_arr, dim) + output_loop.tadd
    if output_loop.tadd < 0:
        tmax_allowed = global_tmax(fit_range_arr, dim) + output_loop.tadd
    tmax = tmax_allowed + 1

    themin = None

    for i, (effmass, pval) in enumerate(median_err):

        # don't print the same thing twice
        if str((effmass, pval)) in used:
            continue
        used.add(str((effmass, pval)))

        fit_range = fit_range_arr[i]

        # length cut for safety (better systematic error control
        # to include more data in a given fit range; trade stat for system.)
        # only apply if the data is plentiful (I=2)
        if usegevp:
            lencut = not hasattr(fit_range[0], '__iter__')
            if not lencut:
                lencut = any([len(i) < 3 for i in fit_range])
                lencut = False if ISOSPIN != 2 else lencut
            if lencut:
                continue

        # global tmax cut
        if max(fit_range[dim]) > tmax_allowed:
            continue

        # global min cut
        if min(fit_range[dim]) < tmin_allowed:
            continue

        # running tmax cut
        # if max([max(j) for j in fit_range]) >= tmax:
        if max(fit_range[dim]) >= tmax:
            continue

        # mostly obsolete params
        pval = trunc(pval)
        median_diff = effmass-median
        median_diff = gvar.gvar(abs(median_diff.val),
                                max(effmass.sdev, median.sdev))
        avg_diff = effmass-avg_dim
        avg_diff = gvar.gvar(abs(avg_diff.val),
                             max(effmass.sdev, avg_dim.sdev))


        # compare this result to all other results
        ind_diff, errstr, tmax_ind = diff_ind(effmass, np.array(median_err)[:, 0],
                                              fit_range_arr, dim, tmax)

        if i:
            themin = effmass
        if i > 1:
            minprev = themin

        # if the max difference is not zero
        if ind_diff.val:

            fake_err = False
            if isinstance(errstr, float):
                fake_err = errfake(fit_range[dim], errstr)
            
            # find the stat. significance
            if ind_diff.sdev:
                sig = np.abs(ind_diff.val)/ind_diff.sdev
                if not fake_err:
                    maxdiff = max(maxdiff, sig)
            else:
                sig = np.inf
                if not fake_err:
                    maxdiff = np.inf

            # print the result
            print(effmass, pval, fit_range)

            # keep track of largest errors; print the running max 
            if maxdiff == sig:
                print("")
                print(ind_diff)
                print("")

            # fit range inconsistency found; error handling below
            try:
                assert sig < 1.5 or fake_err
            except AssertionError:

                themin = minprev

                # disagreement is between two subsets
                # of multiple data points
                disagree_multi_multi_point = len(
                    fit_range) > 1 and not isinstance(errstr, float)

                # I=2 fits to a constant,
                # so we must look at all disagreements
                if (disagree_multi_multi_point or ISOSPIN == 2):
                    if isinstance(errstr, float):
                        if errstr:
                            print("disagreement (mass) with data point at t=",
                                    errstr, 'dim:', dim, "sig =", sig)
                            tmax1 = max(fit_range[dim])
                            tmax1 = max(tmax1, errstr)
                            print("current tmax1:", tmax1)
                            tmax = min(tmax1, tmax)
                            print("current tmax=", tmax)
                            print("")
                    else:
                        #tmax1 = max([max(j) for j in fit_range[dim]])
                        tmax1 = max(fit_range[dim])
                        tmax1 = max(tmax1, tmax_ind)
                        print("current tmax1:", tmax1)
                        tmax = min(tmax1, tmax)
                        print("disagreement at", sig, "sigma")
                        print(errstr, 'dim:', dim)
                        print("current tmax=", tmax)
                        #raise FitRangeInconsistency
                        print("")
        else:
            # no differences found; print the result
            # skip effective mass points for I=0 fits (const+exp)
            if ISOSPIN == 2 or len(fit_range) != 1.0:
                print(effmass, pval, fit_range)
    print('p-value weighted median =', str(median))
    print("p-value weighted mean =", avg_dim)
    return themin
output_loop.tadd = 0

def errfake(fit_range_dim, errstr):
    """Is the disagreement with an effective mass point outside of this
    dimension's fit window?  Then regard this error as spurious"""
    tmin = min(fit_range_dim)
    tmax = max(fit_range_dim)
    ret = False if errstr >= tmin and errstr <= tmax else True
    if not ret:
        print(tmin, tmax, errstr, fit_range_dim)
    return ret
    

def addoffset(hist):
    """Add an offset to each histogram so the horizontal error bars
    don't overlap
    """
    dup = np.zeros(len(hist))
    hist = np.array(hist, dtype=np.float)
    uniqs = [0 for i in range(len(hist))]
    print(hist)
    for i, count in enumerate(hist):
        for j, count2 in enumerate(hist):
            if i >= j or dup[j]:
                continue
            if count == count2:
                dup[i] += 1
                dup[j] += 1
                uniqs[j] = dup[i]
    print(uniqs)
    for i, _ in enumerate(dup):
        if dup[i]:
            offset = 0.1*uniqs[i]/dup[i]
            hist[i] += offset
    print(hist)
    return hist

ISOSPIN = 2

def diff_ind(res, arr, fit_range_arr, dim, tmax):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxsig = 0
    maxdiff = 0
    maxerr = 0
    errstr = ''
    err2str = ''
    tmax_ind = 0
    for i, gres in enumerate(arr):
        # length cut, exactly like the calling function
        lencut = not hasattr(fit_range_arr[i][0], '__iter__')
        if not lencut:
            lencut = any([len(i) < 3 for i in fit_range_arr[i]])
            lencut = False if ISOSPIN != 2 else lencut
        if lencut:
            continue

        # tmax cut
        if hasattr(fit_range_arr[i][0], '__iter__'):
            #tmax1 = max([max(j) for j in fit_range_arr[i]])
            tmax1 = max(fit_range_arr[i][dim])
        else:
            tmax1 = fit_range_arr[i][0]
            err2str = str((tmax1, gres))
        if tmax1 >= tmax:
            continue

        if len(fit_range_arr[i]) == 1 and ISOSPIN != 2:
            continue

        # cuts are passed, calculate the discrepancy
        err = max(res.sdev, gres.sdev)
        diff = abs(res.val-gres.val)
        if diff:
            sig = diff/err
        else:
            sig = 0
        maxsig = max(sig, maxsig)

        if maxsig == sig and maxsig:
            maxdiff = np.abs(diff)
            maxerr = err
            tmax_ind = max(tmax1, tmax_ind)
            if len(fit_range_arr[i]) > 1:
                errstr = "disagree:"+str(i)+" "+str(gres)+" "+str(
                    fit_range_arr[i])
            else:
                errstr = float(fit_range_arr[i][0])
            if maxerr >= maxdiff:
                maxdiff = 0
                maxerr = 0
                maxsig = 0
                tmax_ind = 0
    if maxdiff:
        assert errstr, str(errstr)
        assert tmax_ind < np.inf, str(tmax_ind)
        #if maxsig > 1.5:
            #print(err2str)
    ret = gvar.gvar(maxdiff, maxerr)
    return ret, errstr, tmax_ind


def getxerr(freq, center, errdat_dim):
    """Get horiz. error bars"""
    err = np.zeros(len(center), dtype=np.float)
    for k, cent in enumerate(center):
        mindiff = np.inf
        flag = False
        for _, pair in enumerate(zip(freq, errdat_dim)):
            i, j = pair
            mindiff = min(abs(cent-i), abs(mindiff))
            if mindiff == abs(cent-i):
                flag = True
                err[k] = j
        assert flag, "bug"
    assert not isinstance(err[0], str), "xerr needs conversion"
    assert isinstance(err[0], float), "xerr needs conversion"
    assert isinstance(err[0], np.float), "xerr needs conversion"
    print("xerr =", err)
    return err


if __name__ == '__main__':
    main()
