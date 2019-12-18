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
from latfit.config import RANGE_LENGTH_MIN, ISOSPIN
from latfit.jackknife_fit import jack_mean_err
from latfit.utilities.postprod.h5jack import TDIS_MAX

try:
    PROFILE = profile  # throws an exception when PROFILE isn't defined
except NameError:
    def profile(arg2):
        """Line profiler default."""
        return arg2
    PROFILE = profile

LENMIN = 3
assert LENMIN == RANGE_LENGTH_MIN
SYS_ALLOWANCE = None
#SYS_ALLOWANCE = [['0.44042(28)', '-3.04(21)'], ['0.70945(32)', '-14.57(28)'], ['0.8857(39)', '-19.7(4.7)']]

@PROFILE
def geterr(allow):
    ret = allow
    if allow is not None:
        ret = []
        for i, item in enumerate(allow):
            print(item)
            ret.append([gvar.gvar(item[j]).sdev for j,
                        _ in enumerate(item)])
        ret = np.array(ret)
    return ret

SYS_ALLOWANCE = geterr(SYS_ALLOWANCE)

def arithseq(fitrange):
    """Check if arithmetic sequence"""
    ret = True
    for fitr in fitrange:
        minp = fitr[0]
        nextp = fitr[1]
        step = nextp-minp
        maxp = fitr[-1]
        rchk = np.arange(minp, maxp+step, step)
        if list(rchk) != list(fitr):
            ret = False
    return ret


@PROFILE
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
        min_en1 = make_hist(energyfn, nosave=nosave, allowidx=0)
        min_ph1 = make_hist(phasefn, nosave=nosave, allowidx=1)
        min_res, test, min_res_pr = print_compiled_res(
            min_en1, min_ph1)
        tot = []
        tot_pr = []
        min_ph = min_ph1
        min_en = min_en1
        tot.append(min_res)
        tot_pr.append(min_res_pr)
        success_tadd_tsub = []
        if test:
            success_tadd_tsub.append((0, 0))
        tadd = 0
        breakadd = False
        while tadd < TDIS_MAX:
            if breakadd:
                break
            tsub = 0
            while np.abs(tsub) < TDIS_MAX:
                if tsub or tadd:
                    min_en = make_hist(
                        energyfn, nosave=nosave,
                        tadd=tadd, tsub=tsub, allowidx=0)
                    min_ph = make_hist(
                        phasefn, nosave=nosave,
                        tadd=tadd, tsub=tsub, allowidx=1)
                    tsub -= 1
                else:
                    tsub -= 1
                    continue
                if min_en and min_ph: # check this
                    toapp, test, toapp_pr = print_compiled_res(
                        min_en, min_ph)
                    if test:
                        success_tadd_tsub.append((tadd, tsub))
                    else:
                        if tsub == -1:
                            breakadd = True
                        break
                    tot.append(toapp)
                    tot_pr.append(toapp_pr)
                else:
                    break
            tadd += 1
        print("Successful (tadd, tsub):")
        for i in success_tadd_tsub:
            print(i)
        print_sep_errors(tot_pr)
        print_tot(tot)
    else:
        for fname in sys.argv[1:]:
            min_res = make_hist(fname, nosave=nosave)
        print("minimized error results:", min_res)

def print_sep_errors(tot_pr):
    """Print results with separate errors"""
    for i in tot_pr:
        i = list(i)
        if i:
            fitwin, i = i
            print("fitwin", fitwin)
            print(i)

@PROFILE
def print_tot(tot):
    """Print results vs. tmin"""
    tot_new = []
    lenmax = 0
    print("summary results:")
    for i in tot:
        i = list(i)
        lenmax = max(lenmax, len(i))
        if i and len(i) == lenmax:
            tot_new.append(i)
        if i:
            topr = drop_extra_info(i)
            if topr:
                fitwin, fit_range, res = topr
                print('fitwin =', fitwin)
                if fit_range is not None:
                    print("fit range:", fit_range)
                for i, item in enumerate(res):
                    print("dim", i, ":", item)
    tot = tot_new
    for dim, _ in enumerate(tot[0]):
        plot_t_dep(tot, dim, 1, 'Phase shift', 'degrees')
        plot_t_dep(tot, dim, 0, 'Energy', 'lattice units')

@PROFILE
def drop_extra_info(ilist):
    """Drop fit range data from a 'tot' list item
    (useful for prints)
    i[0][0]==energy
    i[1][0]==phase
    (in general i[j][0] == result)
    i[j][1] == sys_err
    i[j][2] == (fit_range, fit window)
    """
    # find the common fit range, for this fit window if it exists
    fit_range = None
    for i in ilist:
        fitr1 = i[0][2][0]
        fitr2 = i[1][2][0]
        if fit_range_equality(fitr1, fitr2):
            if fit_range is None:
                fit_range = fitr1
            else:
                if not fit_range_equality(fitr1, fit_range):
                    fit_range = None
                    break
        else:
            fit_range = None
            break
    # printable results
    ret = []
    if fit_range is None: 
        # ret is (energy, fit range), (phase, fit range)
        for i in ilist:
            if 'nan' in str(i[0][0]):
                toapp = []
            else:
                fitr1 = i[0][2][0]
                fitr2 = i[1][2][0]
                # check for sub consistency of fit ranges
                if fit_range_equality(fitr1, fitr2):
                    toapp = ([i[0][0], i[1][0]], fitr1)
                else:
                    assert None, (fitr1, fitr2)
                    toapp = [(i[0][0], fitr1), (i[1][0], fitr2)]
            ret.append(toapp)
    else:
        # ret is ((energy, phase)..., fit range)
        ret = [[i[0][0], i[1][0]] if 'nan' not in str(
            i[0][0]) else [] for i in ilist]

    # get common fit window, make sure there is a common fit window
    fitwin = [[i[0][2][1], i[1][2][1]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    fitw = None
    for i in fitwin:
        if fitw is None:
            fitw = i
        else:
            assert list(fitw) == list(i), (fitw, i)
    if hasattr(fitw[0], '__iter__'):
        assert list(fitw[0]) == list(fitw[1])
        fitw = fitw[0]
    return (fitw, fit_range, ret)

@PROFILE
def plot_t_dep(tot, dim, item_num, title, units):
    """Plot the tmin dependence of an item"""
    print("plotting t dependence of dim", dim, "item:", title)
    tot_new = [i[dim][item_num] for i in tot if not np.isnan(
        gvar.gvar(i[dim][item_num][0]).val)]
    try:
        check_fitwin_continuity(tot_new)
    except AssertionError:
        print("fit windows are not continuous for dim, item:", dim, title)
        raise
        tot_new = []
    if tot_new:
        plot_t_dep_totnew(tot_new, dim, title, units)
    else:
        print("not enough consistent results for a complete set of plots")
        sys.exit(1)

@PROFILE
def fit_range_equality(fitr1, fitr2):
    """Check if two fit ranges are the same"""
    ret = True
    assert len(fitr1) == len(fitr2), (fitr1, fitr2)
    for i, j in zip(fitr1, fitr2):
        if list(i) != list(j):
            ret = False
    return ret

@PROFILE
def check_fitwin_continuity(tot_new):
    """Check fit window continuity
    up to the minimal separation of tmin, tmax"""
    maxtmax = {}

    # singleton cut
    assert len(tot_new) > 1, tot_new

    for _, _, fitwin in tot_new:
        fitwin = fitwin[1]
        if fitwin[0] in maxtmax:
            maxtmax[fitwin[0]] = max(maxtmax[fitwin[0]],
                                     fitwin[1])
        else:
            maxtmax[fitwin[0]] = fitwin[1]
    tmin_cont = np.arange(min(maxtmax), max(maxtmax)+1)

    # check tmin is continuous
    assert not set(tmin_cont)-set(maxtmax), list(maxtmax)
    cwin = generate_continuous_windows(maxtmax,
                                       minsep=LENMIN-1)
    for tmin in maxtmax:
        check_set = set()
        for _, _, fitwin in tot_new:
            fitwin = fitwin[1]
            if fitwin[0] == tmin:
                check_set.add(fitwin)

        # check tmax is continuous
        assert not cwin[tmin]-check_set,\
            (cwin[tmin]-check_set)
            #(cwin[tmin], check_set, cwin[tmin]-check_set)
        assert not check_set-cwin[tmin],\
            (cwin[tmin], check_set, check_set-cwin[tmin])

@PROFILE
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

@PROFILE
def plot_t_dep_totnew(tot_new, dim, title, units):
    """Plot something (not nothing)"""
    yarr = []
    yerr = []
    xticks_min = []
    xticks_max = []
    itemprev = None
    fitwinprev = None
    for item, _, fitwin in tot_new:
        fitwin = fitwin[1]
        item = gvar.gvar(item)
        trfitwin = (fitwin[0], fitwin[1] + 1)
        if item == itemprev and trfitwin == fitwinprev and len(
                tot_new) > 10:
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

def round_to_n(val, places):
    """Round to two sigfigs"""
    # from
    # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    ret = round(val, -int(np.floor(np.log10(val))) + (places - 1))
    return ret

def round_wrt(err1, err2):
    """Round err2 with respect to err1"""
    err1 = round_to_n(err1, 2)
    err1 = str(err1)
    places = len(err1.split('.')[1])
    ret = round(err2, places)
    return ret


def other_err_str(val, err1, err2):
    """Get string for other error from a given gvar"""
    if isinstance(val, np.float) and not np.isnan(val) and not np.isnan(err2):
        err2 = round_wrt(err1, err2)
        err = gvar.gvar(val, err2)
        places = place_diff_gvar(gvar.gvar(val, err1),
                                 gvar.gvar(val, err2))
        assert places >= 0, (val, err1, err2, places)
        try:
            ret = str(err).split('(')[1][:-1]
        except IndexError:
            print('va', val)
            print('er', err2)
            raise
        if places and err2:
            assert places == 1
            ret = '0'+ret[0]
        ret = '(' + ret + ')'
    else:
        ret = ''
    return ret

def place_diff_gvar(gvar1, gvar2):
    """Find difference in places between gvar1 and gvar2"""
    one = str(gvar1)
    two = str(gvar2)
    one = len(one.split('(')[0])
    two = len(two.split('(')[0])
    return two-one

def tot_to_stat(res, sys_err):
    """Get result object which has separate stat
    and sys errors"""
    err = res.sdev
    assert err > sys_err, (err, sys_err)
    err = np.sqrt(err**2-sys_err**2)
    ret = gvar.gvar(res.val, err)
    return ret

def errstr(res, sys_err):
    """Print error string"""
    if not np.isnan(res.val):
        assert res.sdev >= sys_err, (res.sdev, sys_err)
        newr = tot_to_stat(res, sys_err)
        assert newr.sdev >= sys_err, "reverse not supported"
        ret = other_err_str(newr.val, newr.sdev, sys_err)
        ret = str(newr)+ret
    else:
        ret = res
    return ret

def swap_err(gvar1, newerr):
    """Swap the errors in the gvar object"""
    err1 = gvar1.sdev
    val = gvar1.val
    ret = gvar.gvar(gvar1, newerr)
    return ret, err1

@PROFILE
def print_compiled_res(min_en, min_ph):
    """Print the compiled results"""
    min_enf = [(str(i), j, k) for i, j, k in min_en]
    min_phf = [(str(i), j, k) for i, j, k in min_ph]

    fitwin = min_en[0][2][1]
    min_en = [errstr(i, j) for i, j, _ in min_en]
    min_ph = [errstr(i, j) for i, j, _ in min_ph]

    min_res = [
        list(i) for i in zip(min_en, min_ph) if list(i)]
    min_res_pr = [
        i for i in min_res if 'nan' not in str(i[0])]
    test = False
    if min_res_pr:
        print("minimized error results:", min_res_pr)
        test = True
    ret = list(zip(min_enf, min_phf))
    return ret, test, (fitwin, min_res_pr)


@PROFILE
def trunc(val):
    """Truncate the precision of a number
    using gvar"""
    if isinstance(val, int):
        ret = val
    else:
        ret = float(str(gvar.gvar(val))[:-3])
    return ret

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
    pdat_freqarr = np.array([em.acmean(i) for i in pdat_freqarr])
    return pdat_freqarr

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
def setup_medians_loop(freq, pdat_freqarr, errlooparr, exclarr):
    """Setup loop over results to get medians/printable results
    """
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
        efferr = np.array([gvar.gvar(i, err) for i in effmass])
        try:
            assert np.allclose(jkerr(effmass), err, rtol=1e-1)
        except AssertionError:
            print(jkerr(effmass))
            print(err)
            print(jkerr2(effmass))
            print(drop)
            raise
        median_err.append([efferr, pval])
        #print(median_err[-1], j)
    if median_diff != 0:
        freq_median = (freq_median+half)/2
    median = systematic_err_est(freq, median_err, freq_median, nosave=nosave)
    return (median_err, median), freq_median

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
            em.acmean(sys_err,axis=0)),
                     xy=(0.05, 0.7),
                     xycoords='axes fraction')
    assert list(freq_median.shape) == list(sys_err.shape),\
        (sys_err.shape, freq_median.shape)
    median = [gvar.gvar(i, j) for i, j in zip(freq_median, sys_err)]
    median = np.array(median)
    return median

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
def make_hist(fname, nosave=False, tadd=0, tsub=0, allowidx=None):
    """Make histograms"""
    freqarr, exclarr, pdat_freqarr, errdat, avg = get_raw_arrays(fname)
    ret = {}
    print("tadd, tsub", tadd, tsub)
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
                loop, freqarr, freq, medians, nosave=nosave)

            # plot median fit result (median energy)
            if not nosave:
                plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                             xycoords='axes fraction')

            # prints the sorted results
            try:
                output_loop.tadd = tadd
                output_loop.tsub = tsub
                themin, sys_err, fitwindow = output_loop(
                    median_store, freqarr, avg[dim],
                    (dim, allowidx),
                    build_sliced_fitrange_list(
                        median_store, freq, exclarr))
            except FitRangeInconsistency:
                continue

            if themin is not None:
                ret[dim] = (themin, sys_err, fitwindow)

            if not nosave:
                print("saving plot as filename:", save_str)

            if not nosave:
                pdf.savefig()

            if not nosave:
                plt.show()
    return fill_conv_dict(ret, freqarr.shape[-1])

@PROFILE
def fill_conv_dict(todict, dimlen):
    """Convert"""
    ret = []
    for i in range(dimlen):
        if i not in todict:
            ret.append((gvar.gvar(np.nan, np.nan), np.nan, (0, np.inf)))
        else:
            ret.append(todict[i])
    return ret

@PROFILE
def build_sliced_fitrange_list(median_store, freq, exclarr):
    """Get all the fit ranges for a particular dimension"""
    ret = []
    freq = em.acmean(freq, axis=1) 
    for _, i in enumerate(median_store[0]):
        effmass = em.acmean([j.val for j in i[0]])
        index = list(freq).index(effmass)
        fitrange = exclarr[index]
        # fitrange = np.array(fitrange)
        # dimfit = fit_range_dim(fitrange, dim)
        dimfit = fitrange
        ret.append(dimfit)
    return ret

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

@PROFILE
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

@PROFILE
def avg_gvar(arr):
    """Average array of gvar objects"""
    ret = em.acmean([i.val for i in arr])
    ret = np.asarray(ret)
    return ret

def lenfitw(fitwin):
    """Length of fit window"""
    return fitwin[1]-fitwin[0]+1

@PROFILE
def output_loop(median_store, freqarr, avg_dim, dim_idx, fit_range_arr):
    """The print loop
    """
    dim, allowidx = dim_idx
    median_err, median = median_store
    maxsig = 0
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

        sdev = effmass[0].sdev
        # don't print the same thing twice
        if str((effmass, pval)) in used:
            continue
        used.add(str((effmass, pval)))

        fit_range = fit_range_arr[i]

        # length cut
        if lencut(fit_range):
            continue

        # arithmetic sequence cut
        if not arithseq(fit_range):
            continue

        # fit window cut
        if fitwincut(fit_range, fitwindow):
            continue

        # mostly obsolete params
        pval = trunc(pval)
        assert list(effmass.shape) == list(median.shape), (
            list(effmass.shape), list(median.shape))
        median_diff = effmass-median
        median_diff = np.array(
            [gvar.gvar(i.val, max(sdev, i.sdev)) for i in median])
        avg_diff = effmass-avg_dim
        avg_diff = np.array(
            [gvar.gvar(i.val, max(sdev, avg_dim.sdev)) for i in avg_diff])

        # compare this result to all other results
        ind_diff, sig, errstr, syserr = diff_ind(
            effmass, np.array(median_err)[:, 0],
            fit_range_arr, fitwindow)

        assert ind_diff.sdev >= syserr

        allowable_err = 0.0
        if SYS_ALLOWANCE is not None:
            allowable_err = SYS_ALLOWANCE[dim][allowidx]
        assert isinstance(allowable_err, np.float), (
            SYS_ALLOWANCE, dim, allowidx)

        if themin is not None:
            minprev = themin
            if themin[0].sdev >= ind_diff.sdev:
                themin = (gvar.gvar(avg_gvar(effmass), ind_diff.sdev), syserr, fit_range)
        else:
            themin = (gvar.gvar(avg_gvar(effmass), ind_diff.sdev), syserr, fit_range)

        # if the max difference is not zero
        if ind_diff.val:

            #fake_err = False
            #if isinstance(errstr, float):
            #    fake_err = errfake(fit_range[dim], errstr)

            maxsig = max(maxsig, sig)

            # print the result
            printres(effmass, pval, fit_range)

            # keep track of largest errors;
            # print the running max
            if maxsig == sig:
                print("")
                print(ind_diff)
                print("")

            # disagreement is between two subsets
            # of multiple data points
            #disagree_multi_multi_point = len(
            #    fit_range) > 1 and not isinstance(errstr, float)

            # fit range inconsistency found;
            # error handling below
            # I=2 fits to a constant,
            # so we must look at all disagreements
            try:
                #assert sig < 1.5 or fake_err or not (
                #    disagree_multi_multi_point or ISOSPIN == 2)
                assert sig < 1.5 or ind_diff.sdev > allowable_err
            except AssertionError:

                if isinstance(errstr, float):
                    if errstr:
                        print("disagreement (mass) with data point at t=",
                              errstr, 'dim:', dim, "sig =", sig)
                        print("")
                else:
                    print("disagreement at", sig, "sigma")
                    print(errstr, 'dim:', dim)
                    print("")

                raise FitRangeInconsistency
        else:
            # no differences found; print the result
            # skip effective mass points for I=0 fits (const+exp)
            if ISOSPIN == 2 or len(fit_range) != 1.0:
                printres(effmass, pval, fit_range)
    if themin is not None:
        print('p-value weighted median =', gvar.gvar(avg_gvar(median),
                                                     median[0].sdev))
        print("p-value weighted mean =", avg_dim)
    else:
        themin = (None, None, None)
    return themin[0], themin[1], (themin[2], fitwindow)
output_loop.tadd = 0
output_loop.tsub = 0

@PROFILE
def printres(effmass, pval, fit_range):
    """Print the result (and a header for the first result printed)"""
    effmass1 = avg_gvar(effmass)
    effmass1 = gvar.gvar(effmass1, effmass[0].sdev)
    if not printres.prt:
        print("val(err); pvalue; ind diff; median difference;",
              " avg difference; fit range")
        printres.prt = True
    print(effmass1, pval, fit_range)
printres.prt = False

@PROFILE
def reset_header():
    """Reset header at the beginning of the output loop"""
    printres.prt = False


@PROFILE
def errfake(frdim, errstr):
    """Is the disagreement with an effective mass point outside of this
    dimension's fit window?  Then regard this error as spurious"""
    tmin = min(frdim)
    tmax = max(frdim)
    ret = errstr < tmin or errstr > tmax
    if not ret:
        print(tmin, tmax, errstr, frdim)
    return ret


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

@PROFILE
def lencut(fit_range):
    """Length cut; we require each fit range to have
    a minimum number of time slices
    length cut for safety (better systematic error control
    to include more data in a given fit range;
    trade stat for system.)
    only apply if the data is plentiful (I=2)
    """
    ret = False
    iterf = hasattr(fit_range[0], '__iter__')
    effmasspt = not iterf and len(fit_range) == 1
    if effmasspt:
        ret = True
    if not ret:
        if iterf:
            ret = any([len(i) < LENMIN for i in fit_range])
        else:
            ret = len(fit_range) < LENMIN
    return ret

@PROFILE
def fitwincut(fit_range, fitwindow):
    """Cut all fit ranges outside this fit window"""
    # tmin, tmax cut
    ret = False
    iterf = hasattr(fit_range[0], '__iter__')
    if iterf:
        tmax = max([max(j) for j in fit_range])
        tmin = min([min(j) for j in fit_range])
    else:
        tmax = max(fit_range)
        tmin = min(fit_range)
    ret = tmax > fitwindow[1] or tmin < fitwindow[0]
    if not ret and lenfitw(fitwindow) == 1:
        print(fitwindow)
        print(tmin, tmax)
        print(fit_range)
        sys.exit(1)
    return ret


@PROFILE
def discrep(res, gres, maxsys_errcurr):
    """Calculate the stat. sig of the disagreement"""
    resarr = [i.val for i in res]
    gresarr = [i.val for i in gres]
    diff = []
    for i, j in zip(resarr, gresarr):
        diff.append(abs(i-j))
    diff = np.array(diff)
    # needs super jack
    err = jkerr(diff)
    sys_err = max(0, em.acmean(diff)-1.5*err)
    #sig = statlvl(gvar.gvar(em.acmean(diff), err))
    #maxsig = max(sig, maxsigcurr)
    return diff, sys_err

@PROFILE
def statlvl(diff):
    """Calculate the statistical significance of a gvar diff"""
    if diff.val:
        if diff.sdev:
            sig = diff.val/diff.sdev
        else:
            sig = np.inf
    else:
        sig = 0
    return sig

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
def diff_ind(res, arr, fit_range_arr, fitwindow):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxdiff = None
    maxsyserr = 0
    errstr = ''
    for i, gres in enumerate(arr):

        # apply cuts
        if lencut(fit_range_arr[i]):
            continue
        if not arithseq(fit_range_arr[i]):
            continue
        if fitwincut(fit_range_arr[i], fitwindow):
            continue

        # cuts are passed, calculate the discrepancy
        diff, syserr = discrep(res, gres, maxsyserr)
        maxsyserr = max(syserr, maxsyserr)
        #maxdiff = maxarr(diff, maxdiff)
        #if np.all(diff == maxdiff):
        #    maxerr = err

        #if maxsig == sig and maxsig:
        if syserr == maxsyserr:
            maxdiff = diff
            mean = avg_gvar(gres)
            sdev = gres[0].sdev
            if len(fit_range_arr[i]) > 1:
                errstr = "disagree:"+str(i)+" "+str(
                    gvar.gvar(mean, sdev))+" "+str(fit_range_arr[i])
            else:
                errstr = float(fit_range_arr[i][0])
    mean = em.acmean(maxdiff)
    sig = statlvl(gvar.gvar(mean, jkerr(maxdiff)))
    ret = gvar.gvar(mean, np.sqrt(maxsyserr**2+res[0].sdev**2))
    return ret, sig, errstr, maxsyserr

@PROFILE
def jkerr(arr):
    """jackknife error"""
    return jack_mean_err(arr)[1]

def jkerr2(arr):
    """jackknife error"""
    return em.acstd(arr)*np.sqrt(len(arr)-1)

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
