#!/usr/bin/python3
"""Make histograms from fit results over fit ranges"""
import sys
import re
import ast
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
#SYS_ALLOWANCE = [['0.44042(28)', '-3.04(21)'], ['0.70945(32)',
# '-14.57(28)'], ['0.8857(39)', '-19.7(4.7)']]

CBEST = [[['0.21080(44)', '-0.26(12)'], ['0.45676(52)', '-13.18(60)'],
          ['0.6147(16)', '-21.7(1.6)'], ['0.7257(50)', '-28.8(9.3)']]]

def fill_best(cbest):
    """Fill the ALLOW buffers with current best known result"""
    rete = []
    retph = []
    for i in cbest:
        aph = []
        aen = []
        for j in i:
            energy = gvar.gvar(j[0])
            phase = gvar.gvar(j[1])
            aen.append(energy)
            aph.append(phase)
        rete.append(aen)
        retph.append(aph)
    return rete, retph

ALLOW_ENERGY, ALLOW_PHASE = fill_best(CBEST)

REVERSE = False
@PROFILE
def geterr(allow):
    """Process SYS_ALLOWANCE"""
    ret = allow
    if allow is not None:
        ret = []
        for _, item in enumerate(allow):
            print(item)
            ret.append([gvar.gvar(item[j]).sdev for j,
                        _ in enumerate(item)])
        ret = np.array(ret)
    return ret

SYS_ALLOWANCE = geterr(SYS_ALLOWANCE)

@PROFILE
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
def select_ph_en(sel):
    """Select which quantity we will analyze now"""
    assert sel in ('phase', 'energy'), sel
    find_best.sel = sel

@PROFILE
def main(nosave=True):
    """Make the histograms.
    """
    if len(sys.argv[1:]) == 1 and (
            'phase_shift' in sys.argv[1] or\
            'energy' in sys.argv[1]) and nosave:

        # init variables
        fname = sys.argv[1]
        tot = []
        tot_pr = []
        success_tadd_tsub = []
        tadd = 0
        breakadd = False

        # get file names
        energyfn, phasefn = enph_filenames(fname)

        # loop over fit windows
        while tadd < TDIS_MAX:
            if breakadd:
                break
            tsub = 0
            while np.abs(tsub) < TDIS_MAX:
                select_ph_en('energy')
                set_tadd_tsub(tadd, tsub)
                min_en = make_hist(
                    energyfn, nosave=nosave, allowidx=0)
                if min_en:
                    select_ph_en('phase')
                    min_ph = make_hist(
                        phasefn, nosave=nosave,
                        allowidx=1)
                if min_en and min_ph: # check this
                    toapp, test, toapp_pr = print_compiled_res(
                        min_en, min_ph)
                    if test:
                        success_tadd_tsub.append((tadd, tsub))
                    else:
                        breakadd = not tsub
                        break
                    tot.append(toapp)
                    tot_pr.append(toapp_pr)
                else:
                    breakadd = not tsub
                    break
                tsub -= 1
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
        prune_cbest()

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
                for idx, item in enumerate(res):
                    print("dim", idx, ":", item)
    tot = tot_new
    for dim, _ in enumerate(tot[0]):
        plot_t_dep(tot, dim, 0, 'Energy', 'lattice units')
        plot_t_dep(tot, dim, 1, 'Phase shift', 'degrees')
    print(plot_t_dep.coll)
    pr_best_fitwin(plot_t_dep.fitwin_votes)

@PROFILE
def pr_best_fitwin(fitwin_votes):
    """Print the best fit window
    (most minimum results)"""
    mvotes = 0
    key = None
    for i in fitwin_votes:
        mvotes = max(fitwin_votes[i], mvotes)
        if mvotes == fitwin_votes[i]:
            key = i
    assert key is not None, key
    print("best fit window:", key)

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
                    #assert None, (fitr1, fitr2)
                    toapp = [(i[0][0], fitr1), (i[1][0], fitr2)]
            ret.append(toapp)
    else:
        # ret is ((energy, phase)..., fit range)
        ret = [[i[0][0], i[1][0]] if 'nan' not in str(
            i[0][0]) else [] for i in ilist]

    # get common fit window, make sure there is a common fit window
    fitwin = [[i[0][2][1], i[1][2][1]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    fitr = [[i[0][2][0], i[1][2][0]] if 'nan' not in str(
        i[0][0]) else [] for i in ilist]
    fitw = None
    for i, j in zip(fitwin, fitr):
        i = list(i)
        j = list(j)
        if not i and not j:
            continue
        assert i or j, (i, j)
        if fitw is None:
            fitw = i
        else:
            assert list(fitw) == list(i), (fitw, i, j)
    if hasattr(fitw[0], '__iter__'):
        assert list(fitw[0]) == list(fitw[1]), fitw
        fitw = fitw[0]
    return (fitw, fit_range, ret)

@PROFILE
def plot_t_dep(tot, dim, item_num, title, units):
    """Plot the tmin dependence of an item"""
    print("plotting t dependence of dim", dim, "item:", title)
    tot_new = [i[dim][item_num] for i in tot if not np.isnan(
        gvar.gvar(i[dim][item_num][0]).val)]
    try:
        if not ALLOW_ENERGY and not ALLOW_PHASE:
            tot_new = check_fitwin_continuity(tot_new)
    except AssertionError:
        print("fit windows are not continuous for dim, item:", dim, title)
        raise
        #tot_new = []
    if list(tot_new):
        plot_t_dep_totnew(tot_new, dim, title, units)
    else:
        print("not enough consistent results for a complete set of plots")
        sys.exit(1)

@PROFILE
def fit_range_equality(fitr1, fitr2):
    """Check if two fit ranges are the same"""
    ret = True
    assert hasattr(fitr1, '__iter__'), (fitr1, fitr2)
    assert hasattr(fitr2, '__iter__'), (fitr1, fitr2)
    assert len(fitr1) == len(fitr2), (fitr1, fitr2)
    for i, j in zip(fitr1, fitr2):
        if list(i) != list(j):
            ret = False
    return ret

@PROFILE
def check_fitwin_continuity(tot_new):
    """Check fit window continuity
    up to the minimal separation of tmin, tmax"""
    continuous_tmin_singleton(tot_new)
    flag = 1
    while flag:
        try:
            continuous_tmax(tot_new)
            flag = 0
        except AssertionError as err:
            tocut = set()
            err = ast.literal_eval(str(err))
            for i in err:
                assert len(i) == 2, (i, err)
                tocut.add(i[0])
            tot_new = cut_tmin(tot_new, tocut)
            continuous_tmin_singleton(tot_new)
    return tot_new

@PROFILE
def cut_tmin(tot_new, tocut):
    """Cut tmin"""
    todel = []
    for i, (_, _, fitwin) in enumerate(tot_new):
        fitwin = fitwin[1] # cut out the fit range info
        assert isinstance(fitwin[0], np.float), fitwin
        if fitwin[0] in tocut or fitwin[0] < min(tocut):
            todel.append(i)
    ret = np.delete(tot_new, todel, axis=0)
    return ret


@PROFILE
def continuous_tmax(tot_new):
    """Check continuous tmax"""
    maxtmax = max_tmax(tot_new)
    cwin = generate_continuous_windows(maxtmax)
    print("cwin=", cwin)
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
        # sanity check
        #assert not check_set-cwin[tmin],\
            #(cwin[tmin], check_set, check_set-cwin[tmin])

@PROFILE
def continuous_tmin_singleton(tot_new):
    """Check for continous tmin, singleton cut"""

    # singleton cut
    assert len(tot_new) > 1, tot_new
    maxtmax = max_tmax(tot_new)
    tmin_cont = np.arange(min(maxtmax), max(maxtmax)+1)

    # check tmin is continuous
    assert not set(tmin_cont)-set(maxtmax), list(maxtmax)

@PROFILE
def max_tmax(tot_new):
    """Find the maximum tmax for each tmin"""
    ret = {}
    for _, _, fitwin in tot_new:
        fitwin = fitwin[1]
        if fitwin[0] in ret:
            ret[fitwin[0]] = max(ret[fitwin[0]],
                                 fitwin[1])
        else:
            ret[fitwin[0]] = fitwin[1]
    return ret


#def generate_continuous_windows(maxtmax, minsep=LENMIN-1):
@PROFILE
def generate_continuous_windows(maxtmax, minsep=LENMIN+1):
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
def quick_compare(tot_new):
    """Check final results for consistency"""
    for item, _, fitwin in tot_new:
        item = gvar.gvar(item)
        for item2, _, fitwin2 in tot_new:
            item2 = gvar.gvar(item2)
            assert consistency(item, item2), (
                item, item2, fitwin, fitwin2)

@PROFILE
def consistency(item1, item2):
    """Check two gvar items for consistency"""
    diff = np.abs(item1.val-item2.val)
    dev = max(item1.sdev, item2.sdev)
    sig = statlvl(gvar.gvar(diff, dev))
    ret = np.allclose(0, max(0, sig-1.5), rtol=1e-12)
    if not ret:
        print("sig inconsis. =", sig)
        assert sig < 10, (sig, "check the best known list for",
                          "compatibility with current set",
                          "of results being analyzed")
    return ret

@PROFILE
def plot_t_dep_totnew(tot_new, dim, title, units):
    """Plot something (not nothing)"""
    quick_compare(tot_new)
    yarr = []
    yerr = []
    xticks_min = []
    xticks_max = []
    itemprev = None
    fitwinprev = None
    itmin = [gvar.gvar(np.nan, np.inf), []]
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
        if item.sdev <= itmin[0].sdev:
            itmin = (item, fitwin)
        xticks_min.append(str(fitwin[0]))
        xticks_max.append(str(fitwin[1]))
        fitwinprev = fitwin
        itemprev = item
    if itmin[1] not in plot_t_dep.fitwin_votes:
        plot_t_dep.fitwin_votes[itmin[1]] = 0
    plot_t_dep.fitwin_votes[itmin[1]] += 1
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
        print("minimum error:", itmin[0], "fit window:", itmin[1])
        app_itmin(itmin[0])
        print("saving fig:", save_str)
        pdf.savefig()
    plt.show()
plot_t_dep.fitwin_votes = {}
plot_t_dep.coll = []

@PROFILE
def app_itmin(itmin):
    """Append next minimum result"""
    app_itmin.curr.append(str(itmin))
    if len(app_itmin.curr) == 2:
        plot_t_dep.coll.append(app_itmin.curr)
        app_itmin.curr = []
app_itmin.curr = []

@PROFILE
def round_to_n(val, places):
    """Round to two sigfigs"""
    # from
    # https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
    ret = round(val, -int(np.floor(np.log10(val))) + (places - 1))
    return ret

@PROFILE
def round_wrt(err1, err2):
    """Round err2 with respect to err1"""
    err1 = round_to_n(err1, 2)
    err1 = str(err1)
    places = len(err1.split('.')[1])
    ret = round(err2, places)
    return ret


@PROFILE
def other_err_str(val, err1, err2):
    """Get string for other error from a given gvar"""
    if isinstance(val, np.float) and not np.isnan(val) and not np.isnan(err2):
        err2 = round_wrt(err1, err2)
        err = gvar.gvar(val, err2)
        if not err2:
            places = np.inf
        else:
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
            assert places == 1, (val, err1, err2)
            ret = '0'+ret[0]
        ret = '(' + ret + ')'
    else:
        ret = ''
    return ret

@PROFILE
def place_diff_gvar(gvar1, gvar2):
    """Find difference in places between gvar1 and gvar2"""
    one = str(gvar1)
    two = str(gvar2)
    one = one.split('(')[0]
    two = two.split('(')[0]
    one = remove_period(one)
    two = remove_period(two)
    two = len(two)
    one = len(one)
    return two-one

@PROFILE
def remove_period(astr):
    """Remove decimal point"""
    return re.sub(r'.', '', astr)

@PROFILE
def tot_to_stat(res, sys_err):
    """Get result object which has separate stat
    and sys errors"""
    err = res.sdev
    assert err > sys_err, (err, sys_err)
    err = np.sqrt(err**2-sys_err**2)
    ret = gvar.gvar(res.val, err)
    return ret

@PROFILE
def errstr(res, sys_err):
    """Print error string"""
    if not np.isnan(res.val):
        assert res.sdev >= sys_err, (res.sdev, sys_err)
        newr = tot_to_stat(res, sys_err)
        if newr.sdev >= sys_err:
            ret = other_err_str(newr.val, newr.sdev, sys_err)
            ret = str(newr)+ret
        else:
            ret = other_err_str(newr.val, sys_err, newr.sdev)
            ret = swap_err_str(gvar.gvar(newr.val, sys_err), ret)
    else:
        ret = res
    return ret

@PROFILE
def swap_err_str(gvar1, errstr1):
    """Swap the gvar error string with the new one"""
    val, errs = str(gvar1).split("(") # sys err
    assert np.float(errs[:-1]) >= gvar.gvar('0'+errstr1).val, (gvar1, errstr1)
    ret = str(val)+errstr1+'('+errs
    return ret

@PROFILE
def swap_err(gvar1, newerr):
    """Swap the errors in the gvar object"""
    err1 = gvar1.sdev
    #val = gvar1.val
    ret = gvar.gvar(gvar1, newerr)
    return ret, err1

@PROFILE
def print_compiled_res(min_en, min_ph):
    """Print the compiled results"""
    min_enf = [(str(i), j, k) for i, j, k in min_en]
    min_phf = [(str(i), j, k) for i, j, k in min_ph]

    fitwin = min_en[0][2][1]
    fitwin2 = min_ph[0][2][1]
    assert list(fitwin) == list(fitwin2), (fitwin, fitwin2)
    if not (ALLOW_ENERGY or ALLOW_PHASE):
        min_en = [errstr(i, j) for i, j, _ in min_en]
        min_ph = [errstr(i, j) for i, j, _ in min_ph]
    else:
        min_en = [i for i, _, _ in min_en]
        min_ph = [i for i, _, _ in min_ph]

    min_res = [
        list(i) for i in zip(min_en, min_ph) if list(i)]
    min_res_pr = [
        i for i in min_res if 'nan' not in str(i[0])]
    update_best(min_res_pr)
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
    pdat_freqarr = np.array([np.mean(i, axis=0) for i in pdat_freqarr])
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
    assert len(freq) == len(errlooparr)
    loop = sorted(zip(freq, pdat_freqarr, errlooparr, exclarr),
                  key=lambda elem: elem[2], reverse=REVERSE)
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
def get_medians_and_plot_syserr(loop, freqarr, freq, medians, dim, nosave=False):
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
        median_err.append([efferr, pval, emean])
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
            em.acmean(sys_err, axis=0)),
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

def reset_window():
    """Reset internal fit window variables"""
    get_fitwindow.tadd = 0
    get_fitwindow.tsub = 0
    get_fitwindow.win = (None, None)

def set_tadd_tsub(tadd, tsub):
    """Store tadd, tsub to find fit window"""
    reset_window()
    get_fitwindow.tadd = tadd
    get_fitwindow.tsub = tsub

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

            # plot median fit result (median energy)
            if not nosave:
                plt.annotate("median="+str(freq_median), xy=(0.05, 0.8),
                             xycoords='axes fraction')

            # prints the sorted results
            try:
                themin, sys_err, fitr = output_loop(
                    median_store, avg[dim], (dim, allowidx),
                    build_sliced_fitrange_list(
                        median_store, freq, exclarr))
            except FitRangeInconsistency:
                continue

            if themin != gvar.gvar(0, np.inf):
                ret[dim] = (themin, sys_err, fitr)
            else:
                break

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
    if todict:
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
        tee = np.inf
        if len(i) > 1:
            tee = min([min(j) for j in i])
        tmin = min(tee, tmin)
    return tmin

@PROFILE
def global_tmax(fit_range_arr):
    """Find the global tmax for this dimension:
    the maximum t for a successful fit"""
    tmax = 0
    for i in fit_range_arr:
        tee = 0
        if len(i) > 1:
            tee = max([max(j) for j in i])
        tmax = max(tmax, tee)
    return tmax

@PROFILE
def avg_gvar(arr):
    """Average array of gvar objects"""
    ret = em.acmean([i.val for i in arr])
    ret = np.asarray(ret)
    return ret

@PROFILE
def lenfitw(fitwin):
    """Length of fit window"""
    return fitwin[1]-fitwin[0]+1

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

def match_arrs(arr, new):
    """Match array lengths by extending the new array length"""
    arr = list(arr)
    new = list(new)
    dlen = len(arr) - len(new)
    assert dlen >= 0, (arr, new, dlen)
    if dlen:
        ext = arr[-1*dlen:]
        new.extend(ext)
    return new


def compare_bests(new, curr):
    """Compare new best to current best to see if an update is needed"""
    rete = []
    retph = []
    ret = False
    for ibest in curr:
        new = match_arrs(ibest, new)
        assert len(new) == len(ibest), (new, ibest)
        aph = []
        aen = []
        for jit, kit in zip(ibest, new):
            enerr = gvar.gvar(jit[0]).sdev
            pherr = gvar.gvar(jit[1]).sdev
            enerr_new = gvar.gvar(kit[0]).sdev
            pherr_new = gvar.gvar(kit[1]).sdev
            if enerr > enerr_new or pherr > pherr_new:
                ret = True
                break
    return ret

def prune_cbest(cbest=None):
    """Prune the best known so only
    the smallest error params remain.  Print the result."""
    cbest = update_best.cbest if cbest is None else cbest
    if len(cbest) == 1:
        cnew = cbest[0]
    else:
        diml = len(cbest[0])
        cnew = None
        for ibest in cbest:
            if cnew is None:
                cnew = ibest
                continue
            else:
                assert len(cnew) == len(ibest)
            aph = []
            aen = []
            for idx, (jit, kit) in enumerate(zip(ibest, cnew)):
                enerr = gvar.gvar(jit[0]).sdev
                pherr = gvar.gvar(jit[1]).sdev
                enerr_new = gvar.gvar(kit[0]).sdev
                pherr_new = gvar.gvar(kit[1]).sdev
                if enerr < enerr_new:
                    cnew[idx][0] = str(jit[0])
                if pherr < pherr_new:
                    cnew[idx][1] = str(jit[1])
    print("pruned cbest list:", cnew)
    return cnew

def update_best(new_best):
    """Update best lists"""
    cbest = update_best.cbest
    if cbest:
        if compare_bests(new_best, update_best.cbest):
            print("adding new best known params:", new_best)
            new_best = match_arrs(update_best.cbest[0], new_best)
            update_best.cbest.append(new_best)
            prune_cbest(cbest)
            cbest = [update_best.cbest[0]]
            print("current best known params list:", cbest)
    find_best.allow_energy, find_best.allow_phase = fill_best(cbest)
update_best.cbest = CBEST

def find_best():
    """Find best known result"""
    assert find_best.sel is not None, find_best.sel
    sel = find_best.sel
    if sel == 'phase':
        best = find_best.allow_phase
    if sel == 'energy':
        best = find_best.allow_energy
    return best
find_best.sel = None
find_best.allow_phase = ALLOW_PHASE
find_best.allow_energy = ALLOW_ENERGY

@PROFILE
def allow_cut(res, dim, cutstat=True, chk_consis=True):
    """If we already have a minimized error result,
    cut all that are statistically incompatible"""
    ret = False
    best = find_best()
    if best:
        battr = allow_cut.best_attr
        if battr is None:
            battr = hasattr(gvar.gvar(best[0]), '__iter__')
            allow_cut.best_attr = battr
        if battr:
            for i in best:
                ret = ret or res_best_comp(
                    res, i, dim, cutstat=cutstat, chk_consis=chk_consis)
        else:
            ret = res_best_comp(
                res, best, dim, cutstat=cutstat, chk_consis=chk_consis)
    return ret
allow_cut.best_attr = None

@PROFILE
def res_best_comp(res, best, dim, chk_consis=True, cutstat=True):
    """Compare result with best known for consistency"""
    if hasattr(res, '__iter__'):
        sdev = res[0].sdev
        res = np.mean([i.val for i in res], axis=0)
        try:
            res = gvar.gvar(res, sdev)
        except TypeError:
            print(res)
            print(sdev)
            raise
    best = best[dim]
    # best = gvar.gvar(best)
    ret = False
    if chk_consis:
        ret = not consistency(best, res)
    if cutstat:
        ret = ret or best.sdev <= res.sdev
    return ret

@PROFILE
def update_effmass(effmass, errterm):
    """Replace the stat error with the total error in the effmass array"""
    ret = [gvar.gvar(i.val, errterm) for i in effmass]
    return ret

@PROFILE
def fitrange_skip_list(fit_range_arr, fitwindow, dim):
    """List of indices to skip"""
    # fit window cut
    ret = set()
    for idx, item in enumerate(fit_range_arr):
        if lencut(item):
            ret.add(idx)
        elif fitwincuts(item, fitwindow):
            ret.add(idx)
        elif not arithseq(item):
            ret.add(idx)
    ret = sorted(list(ret))
    return ret

@PROFILE
def cut_arr(arr, skip_list):
    """Prune the results array"""
    return np.delete(arr, skip_list, axis=0)

@PROFILE
def get_fitwindow(fit_range_arr):
    """Get fit window, print result"""
    if get_fitwindow.win == (None, None):
        tadd = get_fitwindow.tadd
        tsub = get_fitwindow.tsub
        tmin_allowed = global_tmin(fit_range_arr) + tadd
        tmax_allowed = global_tmax(fit_range_arr) + tsub
        get_fitwindow.win = (tmin_allowed, tmax_allowed)
    ret = get_fitwindow.win
    assert ret[0] is not None, ret
    print("fit window:", ret)
    return ret
get_fitwindow.tadd = 0
get_fitwindow.tsub = 0
get_fitwindow.win = (None, None)


@PROFILE
def fitrange_cuts(median_err, fit_range_arr, dim):
    """Get fit window, apply cuts"""
    fitwindow = get_fitwindow(fit_range_arr)
    skip_list = fitrange_skip_list(fit_range_arr, fitwindow, dim)
    median_err = cut_arr(median_err, skip_list)
    fit_range_arr = cut_arr(fit_range_arr, skip_list)
    return median_err, fit_range_arr
        

@PROFILE
def output_loop(median_store, avg_dim, dim_idx, fit_range_arr):
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

    # cut results outside the fit window
    median_err, fit_range_arr = fitrange_cuts(
        median_err, fit_range_arr, dim)

    sort_check(median_err, reverse=REVERSE)

    don = {}

    for idx, (effmass, pval, emean) in enumerate(median_err):

        midx = None

        sdev = effmass[0].sdev
        try:
            assert sdev, effmass
        except AssertionError:
            continue
        # skip if the error is already too big
        if themin is not None:
            if sdev >= themin[0].sdev:
                break

        fit_range = fit_range_arr[idx]
        assert not lencut(fit_range), (fit_range, idx)

        # best known cut (stat comparison only)
        if allow_cut(gvar.gvar(emean, sdev), dim, chk_consis=False):
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
            [gvar.gvar(i.val, max(sdev, avg_dim.sdev)) for i in avg_diff])

        # compare this result to all other results, up to
        # NOTE:
        # midx is because we only want to do the comparisons once.
        if list(np.array(median_err)[:, 0]):
            if (idx, midx) not in don and (midx, idx) not in don:
                ind_diff, sig, errstr1, syserr, midx, maxrange = diff_ind(
                    (effmass, emean), np.array(median_err)[:, 0::2],
                    fit_range_arr, dim)
                don[(idx, midx)] = (ind_diff, sig, errstr1, syserr, midx, maxrange)
                don[(midx, idx)] = (ind_diff, sig, errstr1, syserr, idx, maxrange)
            else:
                ind_diff, sig, errstr1, syserr, midx, maxrange = don[(idx, midx)]
        else:
            ind_diff, sig, errstr1, syserr, midx, maxrange = (
                gvar.gvar(0, 0), 0, '', 0, None, fit_range)
        assert ind_diff.sdev >= syserr

        errterm = np.sqrt(sdev**2+syserr**2)
        #effmass = update_effmass(effmass, errterm)

        noprint = False
        if themin is not None:
            if themin[0].sdev > errterm:
                themin = (gvar.gvar(emean, errterm), syserr, fit_range)
                #pvalmin = pval
                #fitrmin = fit_range
            else:
                noprint = True
        else:
            #pvalmin = pval
            #fitrmin = fit_range
            themin = (gvar.gvar(emean, errterm), syserr, fit_range)

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
        print('p-value weighted median =', gvar.gvar(avg_gvar(median),
                                                     median[0].sdev))
        print("p-value weighted mean =", avg_dim)
    else:
        themin = (gvar.gvar(0,np.inf), 0, [[]])
    return themin[0], themin[1], (themin[2], get_fitwindow(fit_range_arr))

@PROFILE
def printres(effmass1, pval, syserr, fit_range, maxrange):
    """Print the result (and a header for the first result printed)"""
    #effmass1 = avg_gvar(effmass)
    #effmass1 = gvar.gvar(effmass1, effmass[0].sdev)
    if not printres.prt:
        print("val(err); syserr; pvalue; ind diff; median difference;",
              " avg difference; fit range; disagreeing fit range")
        printres.prt = True
    print(effmass1, syserr, pval, fit_range, maxrange)
printres.prt = False

@PROFILE
def reset_header():
    """Reset header at the beginning of the output loop"""
    printres.prt = False


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

#DIMWIN = [(9,13), (7,11), (9,13), (7,11)]

@PROFILE
def fitwincuts(fit_range, fitwindow, dim=None):
    """Cut all fit ranges outside this fit window,
    also cut out fit windows that are too small
    """
    ret = False

    # skip fit windows of a small length
    if lenfitw(fitwindow) < LENMIN+1:
        ret = True
    if not ret:
        # tmin, tmax cut
        dim = None # appears to help somewhat with errors
        iterf = hasattr(fit_range[0], '__iter__')
        if dim is not None:
            assert iterf, fit_range
            fit_range = fit_range[dim]
    for i, fitr in enumerate(fit_range):
        if ret:
            break
        if i == dim:
            ret = not inside_win(fitr, fitwindow) or ret
        else:
            #dimwin = DIMWIN[i] if iterf else fitwindow
            dimwin = fitwindow
            ret = not inside_win(fitr, dimwin) or ret
    if not ret and lenfitw(fitwindow) == 1:
        print(fitwindow)
        print(fit_range)
        sys.exit(1)
    return ret

@PROFILE
def inside_win(fit_range, fitwin):
    """Check if fit range is inside the fit window"""
    assert hasattr(fit_range, '__iter__'), (fit_range, fitwin)
    iterf = hasattr(fit_range[0], '__iter__')
    if iterf:
        tmax = max([max(j) for j in fit_range])
        tmin = min([min(j) for j in fit_range])
    else:
        tmax = max(fit_range)
        tmin = min(fit_range)
    ret = tmax <= fitwin[1] and tmin >= fitwin[0]
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
def diff_ind(res, arr, fit_range_arr, dim):
    """Find the maximum difference between fit range result i
    and all the other fit ranges
    """
    maxdiff = 0
    maxsyserr = 0
    maxrange = []
    maxerr = 0
    errstr1 = ''
    midx = None
    res, emean = res
    for i, gres in enumerate(arr):

        gres, gemean = gres
        gsdev = gres[0].sdev

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
            maxrange = fit_range_arr[i]
            midx = i
            #mean = avg_gvar(gres)
            sdev = gres[0].sdev
            if len(fit_range_arr[i]) > 1:
                errstr1 = "disagree:"+str(i)+" "+str(
                    gvar.gvar(0, sdev))+" "+str(fit_range_arr[i])
            else:
                errstr1 = float(fit_range_arr[i][0])
    ret = gvar.gvar(maxdiff, maxerr)
    sig = statlvl(ret)
    return ret, sig, errstr1, maxsyserr, midx, maxrange

@PROFILE
def jkerr(arr):
    """jackknife error"""
    arr = np.asarray(arr)
    return jack_mean_err(arr)[1]

@PROFILE
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
        # sys.exit(1)
        IDS_HIST = [ISOSPIN, LENMIN]
        IDS_HIST = np.asarray(IDS_HIST)
        pickle.dump(IDS_HIST, open('ids_hist.p', "wb"))
    check_ids_hist()
    main()
