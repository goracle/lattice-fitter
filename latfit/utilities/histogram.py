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


def main():
    """Make the histograms.
    """
    for fname in sys.argv[1:]:
        make_hist(fname)

def trunc(val):
    """Truncate the precision of a number
    using gvar"""
    if isinstance(val, int):
        ret = val
    else:
        ret = float(str(gvar.gvar(val))[:-3])
    return ret

def fill_pvalue_arr(pdat_freqarr, desired_length):
    """Extend pdat_freqarr to desired length"""
    ext = desired_length-len(pdat_freqarr)
    if ext > 0:
        pdat_freqarr = list(pdat_freqarr)
        for _ in range(ext):
            pdat_freqarr.append(1)
        assert len(pdat_freqarr) == desired_length
    return np.asarray(pdat_freqarr)

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
    #print("file with stat errors:", errfn)
    with open(errfn, 'rb') as fn1:
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
    with open(fname, 'rb') as fn1:
        dat = pickle.load(fn1)
        dat = np.array(dat)
        dat = np.real(dat)
        avg, err, freqarr, exclarr = dat
        exclarr = np.asarray(exclarr)
        avg = gvar.gvar(avg, err)
    spl = fname.split('_')[0]
    return spl, freqarr, avg, exclarr

def get_raw_arrays(fname):
    """Get the raw arrays from the fit output files"""
    spl, freqarr, avg, exclarr = setup_make_hist(fname)

    pdat_freqarr = pvalue_arr(spl, fname)

    pdat_freqarr = fill_pvalue_arr(pdat_freqarr, len(freqarr))

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
    for i, j, k, _ in loop: # drop the fit range from the loop
        # skip the single time slice points for GEVP
        if j == 1.0 and hasattr(freqarr.shape, '__iter__') and\
           hasattr(freqarr.shape[-1], '__iter__') and\
           len(freqarr.shape[-1]) > 1:
            continue
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

def make_hist(fname):
    """Make histograms
    """
    freqarr, exclarr, pdat_freqarr, errdat, avg = get_raw_arrays(fname)
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
            output_loop(median_store, freqarr, avg[dim],
                        build_sliced_fitrange_list(median_store, freq,
                                                   exclarr, dim))

            print("saving plot as filename:", save_str)

            pdf.savefig()

            plt.show()

def build_sliced_fitrange_list(median_store, freq, exclarr, dim):
    """Get all the fit ranges for a particular dimension"""
    ret = []
    for _, i in enumerate(median_store[0]):
        effmass = i[0].val
        index = list(freq).index(effmass)
        fitrange = exclarr[index]
        fitrange = np.array(fitrange)
        dimfit = fit_range_dim(fitrange, dim)
        ret.append(dimfit)
    return ret

def fit_range_dim(lexcl, dim):
    """Get the fit range for a particular dimension"""
    ret = np.asarray(lexcl)
    if len(ret.shape) > 1:
        ret = ret[dim]
    return ret

def output_loop(median_store, freqarr, avg_dim, fit_range_arr):
    """The print loop
    """
    median_err, median = median_store
    maxdiff = 0
    for i, (effmass, pval) in enumerate(median_err):
        fit_range = fit_range_arr[i]
        # skip the single time slice points for GEVP
        if pval == 1.0 and hasattr(
                freqarr.shape[-1], '__iter__') and len(
                    freqarr.shape[-1]) > 1:
            continue
        pval = trunc(pval)
        median_diff = effmass-median
        median_diff = gvar.gvar(abs(median_diff.val),
                                max(effmass.sdev, median.sdev))
        avg_diff = effmass-avg_dim
        avg_diff = gvar.gvar(abs(avg_diff.val),
                             max(effmass.sdev, avg_dim.sdev))

        ind_diff = diff_ind(effmass, np.array(median_err)[:, 0])

        if abs(avg_diff.val) > abs(avg_diff.sdev) or abs(
                median_diff.val) > abs(median_diff.sdev):
            print(effmass, pval, fit_range)
            #print("")
            #print("diffs", ind_diff, median_diff, avg_diff)
            #print("")
        elif ind_diff.val or ind_diff.sdev:
            print(effmass, pval, fit_range)
            maxdiff = max(
                maxdiff, np.abs(ind_diff.val-ind_diff.sdev)/ind_diff.sdev)
            if maxdiff == np.abs(
                    ind_diff.val-ind_diff.sdev)/ind_diff.sdev:
                print("")
                print(ind_diff)
                print("")
        else:
            print(effmass, pval, fit_range)
    print('p-value weighted median =', str(median))
    print("p-value weighted mean =", avg_dim)

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


def diff_ind(res, arr):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxdiff = 0
    maxerr = 0
    for gres in arr:
        diff = abs(res.val-gres.val)
        maxdiff = max(diff, maxdiff)
        if maxdiff == diff:
            maxerr = max(res.sdev, gres.sdev)
            if maxerr >= maxdiff:
                maxdiff = 0
                maxerr = 0
    return gvar.gvar(maxdiff, maxerr)


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
