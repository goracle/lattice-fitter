"""Read in fit data from files, gather and prune"""
import sys
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import gvar
from latfit.jackknife_fit import jack_mean_err
import latfit.utilities.exactmean as em
from latfit.utilities.postfit.bin_analysis import find_best
from latfit.utilities.postfit.fitwin import wintoosmall
from latfit.utilities.postfit.compare_print import allow_cut, output_loop
from latfit.utilities.postfit.compare_print import printres, REVERSE
from latfit.analysis.errorcodes import FitRangeInconsistency

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

@PROFILE
def setup_make_hist(fname):
    """Get some initial variables from file fname
    including the processed string
    the array of energies, the average energy
    and the fit ranges used
    """
    with open(fname, 'rb') as fn1:
        dat = pickle.load(fn1)
        try:
            avg, err, freqarr, exclarr = dat
        except ValueError:
            print("value error for file:", fname)
            raise
        freqarr = np.real(np.array(freqarr))
        print("using results from file", fname,
              "shape", freqarr.shape)
        exclarr = np.asarray(exclarr)
        avg = gvar.gvar(avg, err)
    spl = fname.split('_')[0]
    return spl, freqarr, avg, exclarr


@PROFILE
def get_raw_arrays(fname):
    """Get the raw arrays from the fit output files"""
    spl, freqarr, avg, exclarr = setup_make_hist(fname)

    pdat_freqarr = pvalue_arr(spl, fname)

    pdat_freqarr = fill_pvalue_arr(pdat_freqarr, exclarr)

    errdat = err_arr(fname)
    return freqarr, exclarr, pdat_freqarr, errdat, avg


@PROFILE
def setup_medians_loop(freq, pdat_freqarr, errlooparr, exclarr):
    """Setup loop over results to get medians/printable results
    """
    reset_header()
    pdat_median = np.median(pdat_freqarr)
    median_diff = np.inf
    median_diff2 = np.inf
    # print(freqarr[:, dim], errlooparr)
    # prelim_loop = zip(freq, errlooparr, exclarr)
    assert len(freq) == len(errlooparr)
    loop = sorted(zip(freq, pdat_freqarr, errlooparr, exclarr),
                  key=lambda elem: elem[2], reverse=REVERSE)
    medians = (median_diff, median_diff2, pdat_median)

    return loop, medians


@PROFILE
def get_medians_and_plot_syserr(loop, freqarr, freq,
                                medians, dim, nosave=False):
    """Get medians of various quantities (over fit ranges)
    loop:
    (freq, pdat_freqarr, errlooparr, exclarr)
    (energy, pvalue, error, fit range)
    """
    median_diff, median_diff2, pdat_median = medians
    median_err = []
    half = 0
    usegevp = gevpp(freqarr)
    for effmass, pval, err, drop in loop: # drop the fit range from the loop
        # skip the single time slice points for GEVP
        if pval == 1.0 and usegevp:
            pass
            #continue
        # find median pvalue
        if abs(pval - pdat_median) <= median_diff:
            median_diff = abs(pval-pdat_median)
            freq_median = effmass
        elif abs(pval - pdat_median) <= median_diff2:
            median_diff2 = abs(pval-pdat_median)
            half = effmass
        # check jackknife error and superjackknife error are somewhat close
        emean, sjerr = jack_mean_err(effmass, acc_sum=False)
        efferr = np.array([gvar.gvar(i, err) for i in effmass])
        try:
            if len(drop) == 1:
                assert np.allclose(jkerr2(effmass), err, rtol=1e-12)
            else:
                assert np.allclose(sjerr, err, rtol=1e-12)
        except AssertionError:
            print('superjack err =', jkerr(effmass))
            print('saved err =', err)
            print('jackknife error =', jkerr2(effmass))
            print(drop)
            continue
        if allow_cut(gvar.gvar(emean, sjerr), dim, cutstat=False):
            continue
        # print("appending:", gvar.gvar(emean, err))
        median_err.append([efferr, pval, emean])
        #print(median_err[-1], j)
    if median_diff != 0:
        freq_median = (freq_median+half)/2
    median = systematic_err_est(freq, median_err, freq_median, nosave=nosave)
    return (median_err, median), freq_median

@PROFILE
def make_hist(fname, nosave=False, allowidx=None):
    """Make histograms"""
    freqarr, exclarr, pdat_freqarr, errdat, avg = get_raw_arrays(fname)
    ret = {}
    print("fname", fname)

    for dim in range(freqarr.shape[-1]):
        freq, errlooparr = slice_energy_and_err(
            freqarr, errdat, dim)
        save_str = re.sub(r'.p$',
                          '_state'+str(dim)+'.pdf', fname)
        # setup the loop which obtains medians/printable results
        loop, medians = setup_medians_loop(freq, pdat_freqarr,
                                           errlooparr, exclarr)
        with PdfPages(save_str) as pdf:

            if not nosave:
                plt.ylabel('count')

            # plot the title
            if not nosave:
                plot_title(fname, dim)

            # plot the histogram
            if not nosave:
                plot_hist(freq, errlooparr)

            # loop to obtain medians/printable results
            # plot the systematic error
            median_store, freq_median = get_medians_and_plot_syserr(
                loop, freqarr, freq, medians, dim, nosave=nosave)
            if not list(median_store[0]):
                print("no consistent results to compare, dim:", dim)
                continue

            fit_range_arr = build_sliced_fitrange_list(median_store, freq, exclarr)
            if wintoosmall(fit_range_arr=fit_range_arr):
                print("fit window too small")
                break

            # plot median fit result (median energy)
            if not nosave:
                plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                             xycoords='axes fraction')

            # prints the sorted results
            try:
                themin, sys_err, fitr = output_loop(
                    median_store, avg[dim], (dim, allowidx),
                    fit_range_arr)
            except FitRangeInconsistency:
                continue

            if themin == gvar.gvar(0, np.inf):
                print(find_best.sel,
                      "min not found for dim:", dim)
                continue
            ret[dim] = (themin, sys_err, fitr)

            if not nosave:
                print("saving plot as filename:", save_str)

            if not nosave:
                pdf.savefig()

            if not nosave:
                plt.show()
    return fill_conv_dict(ret, freqarr.shape[-1])

@PROFILE
def build_sliced_fitrange_list(median_store, freq, exclarr):
    """Get all the fit ranges for a particular dimension"""
    ret = []
    freq = np.mean(freq, axis=1)
    for _, i in enumerate(median_store[0]):
        emean = i[2]
        effmass = emean
        #effmass = np.mean([j.val for j in i[0]], axis=0)
        index = list(freq).index(effmass)
        fitrange = exclarr[index]
        # fitrange = np.array(fitrange)
        # dimfit = fit_range_dim(fitrange, dim)
        dimfit = fitrange
        ret.append(dimfit)
    return ret

@PROFILE
def jkerr(arr):
    """jackknife error"""
    arr = np.asarray(arr)
    return jack_mean_err(arr)[1]

@PROFILE
def jkerr2(arr):
    """jackknife error"""
    return em.acstd(arr)*np.sqrt(len(arr)-1)


#DIMWIN = [(9,13), (7,11), (9,13), (7,11)]

@PROFILE
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

@PROFILE
def pvalue_arr(spl, fname):
    """Get pvalue array (over fit ranges)"""
    pvalfn = re.sub(spl, 'pvalue', fname)
    pvalfn = re.sub('shift_', '', pvalfn)

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
    pdat_freqarr = np.array([np.mean(i, axis=0) for i in pdat_freqarr])
    return pdat_freqarr

@PROFILE
def fill_conv_dict(todict, dimlen):
    """Convert"""
    ret = []
    if todict:
        for i in range(dimlen):
            if i not in todict:
                ret.append((gvar.gvar(np.nan, np.nan),
                            np.nan, ([[]], (np.nan, np.nan))))
            else:
                ret.append(todict[i])
    return ret

@PROFILE
def enph_filenames(fname):
    """Get energy and phase shift file names"""
    if 'phase_shift' in fname:
        energyfn = re.sub('phase_shift', 'energy', fname)
        phasefn = fname
    elif 'energy' in fname:
        phasefn = re.sub('energy', 'phase_shift', fname)
        energyfn = fname
    return energyfn, phasefn

@PROFILE
def err_arr(fname):
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
            print("file used for error:", errfn,
                  'shape:', errdat.shape)
    except FileNotFoundError:
        with open(errfn2, 'rb') as fn1:
            errdat = pickle.load(fn1)
            errdat = np.real(np.array(errdat))
            print("file used for error:", errfn2,
                  'shape:', errdat.shape)
    assert np.array(errdat).shape, "error array not found"

    #print('shape:', freqarr.shape, avg)
    #print('shape2:', errdat.shape)
    return errdat

@PROFILE
def slice_energy_and_err(freqarr, errdat, dim):
    """Slice the energy array and error array
    for a particular dimension"""
    freq = np.array([np.real(i) for i in freqarr[:, :, dim]])
    freq_new = np.zeros(freq.shape)
    for i, item in enumerate(freq):
        add = []
        for j in item:
            try:
                add.append(np.real(j))
            except TypeError:
                print(j)
                raise
        freq_new[i] = np.array(add)
        freq = freq_new
    #freq = em.acmean(freq, axis=1)
    errlooparr = errdat[:, dim] if len(
        errdat.shape) > 1 else errdat
    assert len(errlooparr) == len(freq), (len(freq), len(errlooparr))
    return freq, errlooparr

@PROFILE
def fit_range_dim(lexcl, dim):
    """Get the fit range for a particular dimension"""
    ret = np.asarray(lexcl)
    if len(ret.shape) > 1 or isinstance(ret[0], list):
        ret = ret[dim]
    else:
        print(ret)
    return ret


@PROFILE
def gevpp(freqarr):
    """Are we using the GEVP?"""
    ret = False
    shape = freqarr.shape
    if len(freqarr.shape) == 3:
        shape = shape[0::2]
    if hasattr(shape, '__iter__'):
        ret = shape[-1] > 1
    return ret

@PROFILE
def reset_header():
    """Reset header at the beginning of the output loop"""
    printres.prt = False

@PROFILE
def systematic_err_est(freq, median_err, freq_median, nosave=False):
    """Standard deviation of results over fit ranges
    is an estimate of the systematic error.
    Also, annotate the plot with this information.
    """
    # standard deviation
    try:
        sys_err = em.acstd(freq, axis=0, ddof=1)
    except ZeroDivisionError:
        print("zero division error:")
        print(np.array(median_err))
        sys.exit(1)
    if not nosave:
        plt.annotate("standard dev (est of systematic)="+str(
            em.acmean(sys_err, axis=0)),
                     xy=(0.05, 0.7),
                     xycoords='axes fraction')
    assert list(freq_median.shape) == list(sys_err.shape),\
        (sys_err.shape, freq_median.shape)
    median = [gvar.gvar(i, j) for i, j in zip(freq_median, sys_err)]
    median = np.array(median)
    return median

@PROFILE
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

@PROFILE
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

@PROFILE
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

@PROFILE
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
