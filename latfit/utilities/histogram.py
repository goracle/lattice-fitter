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
from latfit.utilities import read_file as rf


ISOSPIN = 2
LENMIN = 3


def main(nosave=True):
    """Make the histograms.
    """
    if len(sys.argv[1:]) == 1 and (
            'phase_shift' in sys.argv[1] or\
            'energy' in sys.argv[1]) and nosave:
        fname = sys.argv[1]
        if 'phase_shift' in fname:
            energyfn = re.sub('phase_shift', 'energy', fname)
            phasefn = fname
        elif 'energy' in fname:
            phasefn = re.sub('energy', 'phase_shift', fname)
            energyfn = fname
        min_en1 = make_hist(energyfn, nosave=nosave)
        min_ph1 = make_hist(phasefn, nosave=nosave)
        min_res, test = print_compiled_res(min_en1, min_ph1)
        tot = []
        min_ph = min_ph1
        min_en = min_en1
        tot.append(min_res)
        success_tadd_tsub = []
        if test:
            success_tadd_tsub.append((0, 0))
        tadd = 0
        while tadd < 22:
            tsub = 0
            while np.abs(tsub) < 22:
                if tsub or tadd:
                    min_en = make_hist(
                        energyfn, nosave=nosave, tadd=tadd, tsub=tsub)
                    min_ph = make_hist(
                        phasefn, nosave=nosave, tadd=tadd, tsub=tsub)
                    tsub -= 1
                else:
                    tsub -= 1
                    continue
                if min_en and min_ph: # check this
                    toapp, test = print_compiled_res(min_en, min_ph)
                    if test:
                        success_tadd_tsub.append((tadd, tsub))
                    else:
                        break
                    tot.append(toapp)
                else:
                    break
            tadd += 1
        print("Successful (tadd, tsub):")
        for i in success_tadd_tsub:
            print(i)
        print_tot(tot)
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, nosave=nosave)
        print("minimized error results:", min_res)

def print_tot(tot):
    """Print results vs. tmin"""
    tot_new = []
    lenmax = 0
    for i in tot:
        i = list(i)
        lenmax = max(lenmax, len(i))
        if i and len(i) == lenmax:
            tot_new.append(i)
        if i:
            topr = drop_fit_range_info(i)
            if topr:
                print(topr)
    tot = tot_new
    for dim, _ in enumerate(tot[0]):
        plot_t_dep(tot, dim, 1, 'Phase shift', 'degrees')
        plot_t_dep(tot, dim, 0, 'Energy', 'lattice units')

def drop_fit_range_info(ilist):
    """Drop fit range data from a 'tot' list item
    (useful for prints)
    """
    ret = [[i[0][0], i[1][0]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    return ret

def plot_t_dep(tot, dim, item_num, title, units):
    """Plot the tmin dependence of an item"""
    print("plotting t dependence of dim", dim, "item:", title)
    tot_new = [i[dim][item_num] for i in tot if not np.isnan(
        gvar.gvar(i[dim][item_num][0]).val)]
    try:
        check_fitwin_continuity(tot_new)
    except AssertionError:
        print("fit windows are not continuous for dim, item:", dim, title)
        tot_new = []
    if tot_new:
        plot_t_dep_totnew(tot_new, dim, title, units)
    else:
        print("not enough consistent results for a complete set of plots")
        sys.exit(1)

def check_fitwin_continuity(tot_new):
    """Check fit window continuity
    up to the minimal separation of tmin, tmax"""
    maxtmax = {}

    # singleton cut
    assert len(tot_new) > 1, tot_new

    for _, fitwin in tot_new:
        if fitwin[0] in maxtmax:
            maxtmax[fitwin[0]] = max(maxtmax[fitwin[0]], fitwin[1])
        else:
            maxtmax[fitwin[0]] = fitwin[1]
    tmin_cont = np.arange(min(maxtmax), max(maxtmax)+1)

    # check tmin is continuous
    assert not set(tmin_cont)-set(maxtmax), list(maxtmax)
    cwin = generate_continuous_windows(maxtmax, minsep=LENMIN-1)
    for tmin in maxtmax:
        check_set = set()
        for _, fitwin in tot_new:
            if fitwin[0] == tmin:
                check_set.add(fitwin)

        # check tmax is continuous
        assert not cwin[tmin]-check_set,\
            (cwin[tmin], check_set, cwin[tmin]-check_set)
        assert not check_set-cwin[tmin],\
            (cwin[tmin], check_set, check_set-cwin[tmin])

def generate_continuous_windows(maxtmax, minsep=LENMIN-1):
    """Generate the set of fit windows
    which is necessary for successful continuity"""
    ret = {}
    for tmin in maxtmax:
        ret[tmin] = set()
        numwin = maxtmax[tmin]-(tmin+minsep)+1
        numwin = int(numwin)
        for i in range(numwin):
            ret[tmin].add((tmin, tmin+minsep+i))
    return ret

def plot_t_dep_totnew(tot_new, dim, title, units):
    """Plot something (not nothing)"""
    yarr = []
    yerr = []
    xticks_min = []
    xticks_max = []
    itemprev = None
    fitwinprev = None
    for item, fitwin in tot_new:
        item = gvar.gvar(item)
        trfitwin = (fitwin[0], fitwin[1] + 1)
        if item == itemprev and trfitwin == fitwinprev and len(tot_new) > 10:
            # decreasing tmax while holding tmin fixed will usually
            # not change the min, so don't plot these
            fitwinprev = fitwin
            itemprev = item
            print("omitting (item, dim, val(err), fitwindow):",
                  title, dim, item, fitwin)
            continue
        if np.isnan(item.val):
            continue
        yarr.append(item.val)
        yerr.append(item.sdev)
        xticks_min.append(str(fitwin[0]))
        xticks_max.append(str(fitwin[1]))
        fitwinprev = fitwin
        itemprev = item
    xarr = list(range(len(xticks_min)))
    assert len(xticks_min) == len(xticks_max)

    fname = sys.argv[1]

    try:
        tmin = int(fname.split('tmin')[1].split('.')[0])
    except IndexError:
        tmin = rf.earliest_time(fname)

    save_str = re.sub('phase_shift_', '', fname)
    save_str = re.sub('energy_', '', save_str)
    save_str = re.sub(' ', '_', save_str)
    save_str = re.sub(r'.p$', '_tdep_'+title+'_state'+str(
        dim)+'.pdf', save_str)

    with PdfPages(save_str) as pdf:
        ax1 = plt.subplot(1, 1, 1)
        ax1.errorbar(xarr, yarr, yerr=yerr, linestyle="None")
        ax1.set_xticks(xarr)
        ax1.set_xticklabels(xticks_min)
        ax1.set_xlabel('fit window tmin (lattice units)')
        ax1.set_ylabel(title+' ('+units+')')

        ax2 = ax1.twiny()
        ax2.set_xticks(xarr)
        ax2.set_xticklabels(xticks_max)
        ax2.set_xlabel('fit window tmax (lattice units)')
        ax2.set_xlim(ax1.get_xlim())

        # ok
        #ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        #ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        #ax2.spines['bottom'].set_position(('outward', 36))

        ax2.set_title(title+' vs. '+'fit window; state '+str(
            dim)+","+r' $t_{min,param}=$'+str(tmin), y=1.12)
        plt.subplots_adjust(top=0.85)
        print("saving fig:", save_str)
        pdf.savefig()
    plt.show()

def print_compiled_res(min_en, min_ph):
    """Print the compiled results"""
    min_enf = [(str(i), j) for i, j in min_en]
    min_phf = [(str(i), j) for i, j in min_ph]

    min_en = [str(i) for i, _ in min_en]
    min_ph = [str(i) for i, _ in min_ph]

    min_res = [list(i) for i in zip(min_en, min_ph) if list(i)]
    min_res_pr = [i for i in min_res if 'nan' not in str(i[0])]
    test = False
    if min_res_pr:
        print("minimized error results:", min_res_pr)
        test = True
    ret = list(zip(min_enf, min_phf))
    return ret, test


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
    return pdat_freqarr

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
    except FileNotFoundError:
        with open(errfn2, 'rb') as fn1:
            errdat = pickle.load(fn1)
            errdat = np.real(np.array(errdat))
    assert np.array(errdat).shape, "error array not found"

    #print('shape:', freqarr.shape, avg)
    #print('shape2:', errdat.shape)
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
    reset_header()
    pdat_median = np.median(pdat_freqarr)
    median_diff = np.inf
    median_diff2 = np.inf
    # print(freqarr[:, dim], errlooparr)
    # prelim_loop = zip(freq, errlooparr, exclarr)
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

    errdat = err_arr(fname)
    return freqarr, exclarr, pdat_freqarr, errdat, avg

def get_medians_and_plot_syserr(loop, freqarr, freq, medians, nosave=False):
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
    median = systematic_err_est(freq, median_err, freq_median, nosave=nosave)
    return (median_err, median), freq_median

def systematic_err_est(freq, median_err, freq_median, nosave=False):
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
    if not nosave:
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

def make_hist(fname, nosave=False, tadd=0, tsub=0):
    """Make histograms"""
    freqarr, exclarr, pdat_freqarr, errdat, avg = get_raw_arrays(fname)
    ret = {}
    for dim in range(freqarr.shape[-1]):
        freq, errlooparr = slice_energy_and_err(freqarr, errdat, dim)
        save_str = re.sub(r'.p$', '_state'+str(dim)+'.pdf', fname)
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
                loop, freqarr, freq, medians, nosave=nosave)

            # plot median fit result (median energy)
            if not nosave:
                plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                             xycoords='axes fraction')

            # prints the sorted results
            try:
                output_loop.tadd = tadd
                output_loop.tsub = tsub
                themin, fitwindow = output_loop(
                    median_store, freqarr, avg[dim], dim,
                    build_sliced_fitrange_list(
                        median_store, freq, exclarr))
            except FitRangeInconsistency:
                continue

            if themin is not None:
                ret[dim] = (themin, fitwindow)

            if not nosave:
                print("saving plot as filename:", save_str)

            if not nosave:
                pdf.savefig()

            if not nosave:
                plt.show()
    return fill_conv_dict(ret, freqarr.shape[-1])

def fill_conv_dict(todict, dimlen):
    """Convert"""
    ret = []
    for i in range(dimlen):
        if i not in todict:
            ret.append((gvar.gvar(np.nan, np.nan), (0, np.inf)))
        else:
            ret.append(todict[i])
    return ret

def build_sliced_fitrange_list(median_store, freq, exclarr):
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

def global_tmin(fit_range_arr):
    """Find the global tmin for this dimension:
    the minimum t for a successful fit"""
    tmin = np.inf
    for i in fit_range_arr:
        if len(i) > 1:
            tee = min([min(j) for j in i])
        else:
            tee = i[0]
        tmin = min(tee, tmin)
    return tmin

def global_tmax(fit_range_arr):
    """Find the global tmax for this dimension:
    the maximum t for a successful fit"""
    tmax = 0
    for i in fit_range_arr:
        if len(i) > 1:
            tee = max([max(j) for j in i])
        else:
            tee = i[0]
        tmax = max(tmax, tee)
    return tmax


def output_loop(median_store, freqarr, avg_dim, dim, fit_range_arr):
    """The print loop
    """
    median_err, median = median_store
    maxdiff = 0
    usegevp = gevpp(freqarr)
    used = set()

    #tmin_allowed = 0
    #tmax_allowed = np.inf
    tmin_allowed = global_tmin(fit_range_arr) + output_loop.tadd
    tmax_allowed = global_tmax(fit_range_arr) + output_loop.tsub

    fitwindow = (tmin_allowed, tmax_allowed)
    #if output_loop.tsub:
    #    print(fitwindow)
        #print(global_tmax(fit_range_arr, dim))
        #print(output_loop.tadd, output_loop.tsub)
    themin = None
    minprev = None

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
                lencut = any([len(i) < LENMIN for i in fit_range])
                lencut = False if ISOSPIN != 2 else lencut
            if lencut:
                continue

        # global tmax cut
        if max([max(i) for i in fit_range]) > tmax_allowed:
            continue

        # global min cut
        if min([min(i) for i in fit_range]) < tmin_allowed:
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
        ind_diff, errstr, tmax_ind = diff_ind(
            effmass, np.array(median_err)[:, 0], fit_range_arr, fitwindow)

        if themin is None:
            themin = effmass
        else:
            minprev = themin
            themin = effmass

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
            printres(effmass, pval, fit_range)

            # keep track of largest errors; print the running max
            if maxdiff == sig:
                print("")
                print(ind_diff)
                print("")

            # disagreement is between two subsets
            # of multiple data points
            disagree_multi_multi_point = len(
                fit_range) > 1 and not isinstance(errstr, float)

            # fit range inconsistency found; error handling below
            # I=2 fits to a constant,
            # so we must look at all disagreements
            try:
                assert sig < 1.5 or fake_err or not (
                    disagree_multi_multi_point or ISOSPIN == 2)
            except AssertionError:

                themin = minprev

                if isinstance(errstr, float):
                    if errstr:
                        print("disagreement (mass) with data point at t=",
                              errstr, 'dim:', dim, "sig =", sig)
                        tmax1 = max(fit_range[dim])
                        tmax1 = max(tmax1, errstr)
                        print("current tmax1:", tmax1)
                        tmax_allowed = min(tmax1, tmax_allowed)
                        print("current tmax=", tmax_allowed)
                        print("")
                else:
                    #tmax1 = max([max(j) for j in fit_range[dim]])
                    tmax1 = max(fit_range[dim])
                    tmax1 = max(tmax1, tmax_ind)
                    print("current tmax1:", tmax1)
                    tmax_allowed = min(tmax1, tmax_allowed)
                    print("disagreement at", sig, "sigma")
                    print(errstr, 'dim:', dim)
                    print("current tmax=", tmax_allowed)
                    print("")

                raise FitRangeInconsistency
        else:
            # no differences found; print the result
            # skip effective mass points for I=0 fits (const+exp)
            if ISOSPIN == 2 or len(fit_range) != 1.0:
                printres(effmass, pval, fit_range)
    if themin is not None:
        print('p-value weighted median =', str(median))
        print("p-value weighted mean =", avg_dim)
    return themin, fitwindow
output_loop.tadd = 0
output_loop.tsub = 0

def printres(effmass, pval, fit_range):
    """Print the result (and a header for the first result printed)"""
    if not printres.prt:
        print("val(err); pvalue; ind diff; median difference;",
              " avg difference; fit range")
        printres.prt = True
    print(effmass, pval, fit_range)
printres.prt = False

def reset_header():
    """Reset header at the beginning of the output loop"""
    printres.prt = False


def errfake(frdim, errstr):
    """Is the disagreement with an effective mass point outside of this
    dimension's fit window?  Then regard this error as spurious"""
    tmin = min(frdim)
    tmax = max(frdim)
    ret = errstr < tmin or errstr > tmax
    if not ret:
        print(tmin, tmax, errstr, frdim)
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

def diff_ind(res, arr, fit_range_arr, fitwindow):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxsig = 0
    maxdiff = 0
    maxerr = 0
    errstr = ''
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
            tmax = max([max(j) for j in fit_range_arr[i]])
            tmin = max([max(j) for j in fit_range_arr[i]])
        else:
            tmax = fit_range_arr[i][0]
            tmin = fit_range_arr[i][0]
        if tmax > fitwindow[1] or tmin < fitwindow[0]:
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
            tmax_ind = max(tmax, tmax_ind)
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



def check_ids_hist():
    """Check the ensemble id file to be sure
    not to run processing parameters from a different ensemble"""
    ids_check = [ISOSPIN, LENMIN]
    ids_check = np.asarray(ids_check)
    ids = pickle.load(open('ids_hist.p', "rb"))
    assert np.all(ids == ids_check),\
        "wrong ensemble. [ISOSPIN, LENMIN]"+\
        " should be:"+str(ids)+" instead of:"+str(ids_check)
    return ids


if __name__ == '__main__':
    try:
        open('ids_hist.p', 'rb')
    except FileNotFoundError:
        print("be sure to set ISOSPIN, LENMIN",
              "correctly")
        print("edit histogram.py to remove the hold then rerun")
        # the hold
        sys.exit(1)
        IDS_HIST = [ISOSPIN, LENMIN]
        IDS_HIST = np.asarray(IDS_HIST)
        pickle.dump(IDS_HIST, open('ids_hist.p', "wb"))
    check_ids_hist()
    main()
